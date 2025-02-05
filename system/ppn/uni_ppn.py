import os
import json
from copy import deepcopy
from logging import getLogger
from typing import Optional, List, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    GPT2Model,
    GPT2PreTrainedModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from convlab2.util.multiwoz.state import default_state
from convlab2.util.multiwoz.dbquery import Database
from convlab2.policy.vector.vector_multiwoz import process_str_action
from convlab2.util.multiwoz.lexicalize import lexicalize_da as lexicalize_da_dict_to_tuples

from system.ppn.ppn_base import PPNBase
from system.data import (
    NLUOutput,
    DSTOutput,
    PolicyOutput,
    WordPolicyOutput,
    NLGOutput,

    PPNNLUOutput,
    PPNDSTOutput,
    PPNPolicyOutput,
    PPNNLGOutput,

    SystemInternalHistory,
)
from utils import (
    get_default_device,
    set_logger,
)
from system.utils import lexicalize_domain_slot_tuples
from system.ppn.uni_ppn_utils import (
    context_tuples2str,
    da_tuples2structured_str,
    belief_state2structured_str,
    request_state2structured_str,
    extract_updated_belief_state,
    dialogue_state2structured_strs,
)

logger = getLogger(__name__)
set_logger(logger)

class UniPPNValueModel(GPT2PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.transformer = GPT2Model(config)

        # Get hidden dim
        hidden_dim = config.n_embd # assume gpt-2 like model
        
        # self.value_head = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1),
        # )
        
        self.dropout = nn.Dropout(config.embd_pdrop)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = transformer_outputs.last_hidden_state

        batch_size = input_ids.shape[0]
        max_token_length = input_ids.shape[1]

        final_token_indices = ((~attention_mask).argmax(-1) - 1) % max_token_length
        final_token_hidden_state = hidden_state[torch.arange(batch_size), final_token_indices]

        values = self.value_head(self.dropout(final_token_hidden_state))

        return values

class UniPPN(PPNBase):
    copy_token = "copy_original"
    ppn_nlu_bos_token = "<|ppn_nlu|>"
    ppn_dst_bos_token = "<|ppn_dst|>"
    ppn_policy_bos_token = "<|ppn_policy|>"
    ppn_nlg_bos_token = "<|ppn_nlg|>"

    def __init__(
        self,
        policy_model_name: str,
        value_model_name: str,
        model_dtype: str,
        local_rank: int,
        max_context_turns: int,
        max_prompt_tokens: int,
        max_response_tokens: int,
        do_sample: bool,
        top_p: float,
        ref_policy_model_name: Optional[str] = None,
    ) -> None:
        self.local_rank = local_rank

        torch_dtype = getattr(torch, model_dtype)
        device_map = {"" : local_rank}

        self.policy_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=policy_model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        self.policy_tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
        if self.policy_tokenizer.pad_token_id is None:
            self.policy_tokenizer.pad_token_id = self.policy_tokenizer.eos_token_id

        if ref_policy_model_name is None:
            ref_policy_model_name = policy_model_name
        self.refp_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=ref_policy_model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        self.value_model = UniPPNValueModel.from_pretrained(
            pretrained_model_name_or_path=value_model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.value_tokenizer = AutoTokenizer.from_pretrained(value_model_name)
        if self.value_tokenizer.pad_token_id is None:
            self.value_tokenizer.pad_token_id = self.value_tokenizer.eos_token_id

        self.copy_token_id = self.policy_tokenizer.convert_tokens_to_ids(self.copy_token)
        self.ppn_nlu_bos_token_id = self.policy_tokenizer.convert_tokens_to_ids(self.ppn_nlu_bos_token)
        self.ppn_dst_bos_token_id = self.policy_tokenizer.convert_tokens_to_ids(self.ppn_dst_bos_token)
        self.ppn_policy_bos_token_id = self.policy_tokenizer.convert_tokens_to_ids(self.ppn_policy_bos_token)
        self.ppn_nlg_bos_token_id = self.policy_tokenizer.convert_tokens_to_ids(self.ppn_nlg_bos_token)

        self.ppn_nlu_token_ids = [self.ppn_nlu_bos_token_id]
        self.ppn_dst_token_ids = [self.ppn_dst_bos_token_id]
        self.ppn_policy_token_ids = [self.ppn_policy_bos_token_id]
        self.ppn_nlg_token_ids = [self.ppn_nlg_bos_token_id]

        for token_id, token in self.policy_tokenizer.added_tokens_decoder.items():
            if token.content.startswith("user_da."):
                self.ppn_nlu_token_ids.append(token_id)
            elif token.content.startswith("belief_state."):
                self.ppn_dst_token_ids.append(token_id)
            elif token.content.startswith("system_da."):
                self.ppn_policy_token_ids.append(token_id)
            else:
                continue

        self.max_context_turns = max_context_turns
        self.max_prompt_tokens = max_prompt_tokens
        self.max_response_tokens = max_response_tokens
        self.do_sample = do_sample
        self.top_p = top_p

        # MultiWOZ
        self.multiwoz_db = Database()
        self.multiwoz_db_domains = ['Attraction', 'Restaurant', 'Train', 'Hotel']
        self.requestable_intent = ["Request"]

    @classmethod
    def make_ppn_nlu_prompt(
            cls,
            context_tuples: List[Tuple[str, str]],
            user_da_tuples: List[Tuple[str, str, str, str]],
            max_context_turns: int,
            ppn_nlu_bos_token: str,
        ):
        prompt_text = ""

        context_str = context_tuples2str(
            context_tuples=context_tuples,
            max_context_turns=max_context_turns,
            speaker2prefix_mapping={"user": "User:", "system": "System:"},
        )
        prompt_text += f"{context_str} "

        user_da_str = da_tuples2structured_str(da_tuples=user_da_tuples)
        prompt_text += f"UserDA: {user_da_str} "

        prompt_text += ppn_nlu_bos_token
        return prompt_text

    @classmethod
    def make_ppn_dst_prompt(
            cls,
            context_tuples: List[Tuple[str, str]],
            user_da_tuples_or_none: Optional[List[Tuple[str, str, str, str]]],
            updated_belief_state: Dict[str, Dict[str, Dict[str, str]]],
            max_context_turns: int,
            ppn_dst_bos_token: str,
        ):
        prompt_text = ""

        context_str = context_tuples2str(
            context_tuples=context_tuples,
            max_context_turns=max_context_turns,
            speaker2prefix_mapping={"user": "User:", "system": "System:"},
        )
        prompt_text += f"{context_str} "

        if isinstance(user_da_tuples_or_none, list):
            user_da_str = da_tuples2structured_str(da_tuples=user_da_tuples_or_none)
            prompt_text += f"UserDA: {user_da_str} "

        # use only updated slots
        updated_bs_str = belief_state2structured_str(belief_state=updated_belief_state, exclude_empty_slots=False)
        prompt_text += f"UpdatedBeliefState: {updated_bs_str} "

        prompt_text += ppn_dst_bos_token
        return prompt_text

    @classmethod
    def make_ppn_policy_prompt(
            cls,
            context_tuples: List[Tuple[str, str]],
            dialogue_state: Dict[str, Any],
            system_da_tuples_or_str: Union[List[Tuple[str, str, str, str]], str],
            max_context_turns: int,
            multiwoz_db: Database,
            ppn_policy_bos_token: str,
        ):
        prompt_text = ""

        context_str = context_tuples2str(
            context_tuples=context_tuples,
            max_context_turns=max_context_turns,
            speaker2prefix_mapping={"user": "User:", "system": "System:"},
        )
        prompt_text += f"{context_str} "

        bs_str, rs_str, num_ents, terminated = dialogue_state2structured_strs(
            dialogue_state=dialogue_state,
            multiwoz_db=multiwoz_db,
        )
        prompt_text += (
            f"BeliefState: {bs_str} "
            f"RequestState: {rs_str} "
            f"NumEntities: {num_ents} "
            f"Terminated: {terminated} "
        )

        if isinstance(system_da_tuples_or_str, list):
            original_da_str = da_tuples2structured_str(da_tuples=system_da_tuples_or_str)
            prompt_text += f"SystemDA: {original_da_str} "
        else:
            original_da_str = system_da_tuples_or_str
            prompt_text += f"System: {original_da_str} "

        prompt_text += ppn_policy_bos_token
        return prompt_text

    @classmethod
    def make_ppn_nlg_prompt(
            cls,
            context_tuples: List[Tuple[str, str]],
            system_da_tuples: List[Tuple[str, str, str, str]],
            system_response: str,
            max_context_turns: int,
            ppn_nlg_bos_token: str,
        ):
        context_str = context_tuples2str(
            context_tuples=context_tuples,
            max_context_turns=max_context_turns,
            speaker2prefix_mapping={"user": "User:", "system": "System:"},
        )
        system_da_str = da_tuples2structured_str(da_tuples=system_da_tuples)
        prompt_text = (
            f"{context_str} "
            f"SystemDA: {system_da_str} "
            f"System: {system_response} "
            f"{ppn_nlg_bos_token}"
        )
        return prompt_text

    def _make_context_tuples(self, system_internal_history: SystemInternalHistory) -> str:
        context_tuples = []
        for turn_id in range(system_internal_history.num_turns):
            context_tuples += [
                ("user", system_internal_history.user_input_history[turn_id].user_utterance),
                ("system", system_internal_history.system_output_history[turn_id].system_response),
            ]
        context_tuples += [
            ("user", system_internal_history.user_input_history[-1].user_utterance)
        ]
        return context_tuples

    def _generate_response(self, prompt_text: str, bad_words_ids: List[List[int]]) -> Dict[str, Union[str, torch.Tensor]]:
        # 1. make input_ids
        default_truncation_side = self.policy_tokenizer.truncation_side
        self.policy_tokenizer.truncation_side = "left"
        model_inputs = self.policy_tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False,
            truncation=True, max_length=self.max_prompt_tokens,
        )
        self.policy_tokenizer.truncation_side = default_truncation_side

        # 2. generate response
        output_ids = self.policy_model.generate(
            **model_inputs.to(torch.device("cuda", self.local_rank)),
            max_new_tokens=self.max_response_tokens,
            do_sample=self.do_sample,
            num_beams=1,
            temperature=1.0,
            top_p=self.top_p,
            pad_token_id=self.policy_tokenizer.eos_token_id,
            eos_token_id=self.policy_tokenizer.eos_token_id,
            bad_words_ids=bad_words_ids
        )
        prompt_ids = model_inputs.input_ids[0]
        response_ids = output_ids[0, prompt_ids.shape[0]:]
        response_text = self.policy_tokenizer.decode(response_ids, skip_special_tokens=False)

        # 3. compute logprobs
        with torch.no_grad():
            logits = self.policy_model(input_ids=output_ids[:, :-1]).logits[0] # Size([num_tokens, vocab_size])
            ref_logits = self.refp_model(input_ids=output_ids[:, :-1]).logits[0] # Size([num_tokens, vocab_size])
            labels = output_ids[0, 1:] # Size([num_tokens])

            full_logprobs = F.log_softmax(logits, dim=-1)
            ref_full_logprobs = F.log_softmax(ref_logits, dim=-1)

            logprobs = full_logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            ref_logprobs = ref_full_logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        # 4. evaluate value
        value_model_inputs = self.value_tokenizer(
            prompt_text+self.value_tokenizer.eos_token, return_tensors="pt", add_special_tokens=False,
            truncation=True, max_length=self.max_prompt_tokens,
        )
        with torch.no_grad():
            value = self.value_model(**value_model_inputs.to(torch.device("cuda", self.local_rank)))
            value = value.view(-1) # Size([1,1]) -> Size([1])
            state_ids = value_model_inputs.input_ids[0]

        # 5. check response
        is_training_example = True
        # if response_ids.shape[0] <= 1:
        #     is_training_example = False

        return {
            "prompt_text": prompt_text, "response_text": response_text,
            "prompt_ids": prompt_ids.cpu(), "response_ids": response_ids.cpu(), 
            "logprobs": logprobs.cpu(), "ref_logprobs": ref_logprobs.cpu(),
            "full_logprobs": full_logprobs.cpu(), "ref_full_logprobs": ref_full_logprobs.cpu(),
            "state_ids": state_ids.cpu(), "value": value.cpu(),
            "is_training_example": is_training_example,
        }

    def ppn_nlu(self, nlu_output: NLUOutput, system_internal_history: SystemInternalHistory) -> PPNNLUOutput:
        # 1 make prompt text
        prompt_text = self.make_ppn_nlu_prompt(
            context_tuples=self._make_context_tuples(system_internal_history),
            user_da_tuples=nlu_output.user_action,
            max_context_turns=self.max_context_turns,
            ppn_nlu_bos_token=self.ppn_nlu_bos_token,
        )

        # 2. generate response
        other_ppn_token_ids = [[self.ppn_nlu_bos_token_id]] + [
            [t_id] for t_id in self.ppn_dst_token_ids + self.ppn_policy_token_ids + self.ppn_nlg_token_ids
        ]
        generation_result = self._generate_response(prompt_text=prompt_text, bad_words_ids=other_ppn_token_ids)

        # 3. recover da tuples
        da_dict = {}
        current_key_id = None
        do_copy = False
        for token_id in generation_result["response_ids"].tolist():
            if token_id == self.policy_tokenizer.eos_token_id: # EOS token
                break
            elif token_id == self.copy_token_id: # Copy the original
                if current_key_id is None:
                    # If no key has been started yet, accept copy token
                    do_copy = True
                    break
                else:
                    # If a key has been started, ignore the copy token
                    continue
            elif token_id in self.ppn_nlu_token_ids: # Start of a new key
                current_key_id = token_id
                da_dict[current_key_id] = []
            elif current_key_id is not None: # Append token to the current key
                da_dict[current_key_id].append(token_id)
            else: # Skip the token if any key has not been started yet (e.g., the first whitespace token)
                continue

        if do_copy:
            da_tuples = deepcopy(nlu_output.user_action)
        else:
            da_tuples = []
            for key_id, value_ids in da_dict.items():
                token = self.policy_tokenizer.convert_ids_to_tokens(key_id)
                try:
                    _, intent, domain, slot = token.split(".")
                except ValueError as e:
                    print(generation_result["response_ids"])
                    print(token)
                    raise e
                value = self.policy_tokenizer.decode(value_ids, skip_special_tokens=True).strip()
                da_tuples.append((intent, domain, slot, value))

        # 4. make ppn_nlu_output
        ppn_nlu_output = PPNNLUOutput(
            module_name=nlu_output.module_name,
            user_action=da_tuples,
            is_training_example=generation_result.pop("is_training_example"),
            trajectory=generation_result,
        )

        return ppn_nlu_output

    def ppn_dst(self, dst_output: DSTOutput, system_internal_history: SystemInternalHistory) -> PPNDSTOutput:
        # 1. make prompt text
        user_da_tuples = system_internal_history.get_user_action(turn_id=system_internal_history.num_turns)

        if system_internal_history.num_turns > 0:
            prev_dialogue_state = system_internal_history.get_dialogue_state(turn_id=system_internal_history.num_turns-1)
        else:
            prev_dialogue_state = default_state()

        updated_bs = extract_updated_belief_state(
            prev_belief_state=prev_dialogue_state["belief_state"],
            belief_state=dst_output.dialogue_state["belief_state"],
        )
        prompt_text = self.make_ppn_dst_prompt(
            context_tuples=self._make_context_tuples(system_internal_history),
            user_da_tuples_or_none=user_da_tuples,
            updated_belief_state=updated_bs,
            max_context_turns=self.max_context_turns,
            ppn_dst_bos_token=self.ppn_dst_bos_token,
        )

        # 2. generate response
        other_ppn_token_ids = [[self.ppn_dst_bos_token_id]] + [
            [t_id] for t_id in self.ppn_nlu_token_ids + self.ppn_policy_token_ids + self.ppn_nlg_token_ids
        ]
        generation_result = self._generate_response(prompt_text=prompt_text, bad_words_ids=other_ppn_token_ids)

        # 3. recover belief_state and request_state
        ds_dict = {}
        current_key_id = None
        do_copy = False
        for token_id in generation_result["response_ids"].tolist():
            if token_id == self.policy_tokenizer.eos_token_id:
                break
            elif token_id == self.copy_token_id:
                if current_key_id is None:
                    do_copy = True
                    break
                else:
                    continue
            elif token_id in self.ppn_dst_token_ids:
                current_key_id = token_id
                ds_dict[current_key_id] = []
            elif current_key_id:
                ds_dict[current_key_id].append(token_id)
            else:
                continue

        if do_copy:
            belief_state = deepcopy(dst_output.dialogue_state["belief_state"])
        else:
            belief_state = deepcopy(prev_dialogue_state["belief_state"])
            for key_id, value_ids in ds_dict.items():
                key = self.policy_tokenizer.convert_ids_to_tokens(key_id)
                token_type, domain, constraint_type, slot = key.split(".")
                value = self.policy_tokenizer.decode(value_ids, skip_special_tokens=True).strip()
                belief_state[domain][constraint_type][slot] = value

        fixed_ds = deepcopy(dst_output.dialogue_state)
        fixed_ds["belief_state"] = belief_state
        # fixed_ds["request_state"] = request_state

        # 4. make ppn_dst_output
        ppn_dst_output = PPNDSTOutput(
            module_name=dst_output.module_name,
            dialogue_state=fixed_ds,
            is_training_example=generation_result.pop("is_training_example"),
            trajectory=generation_result
        )

        return ppn_dst_output

    def ppn_policy(self, policy_output: Union[PolicyOutput, WordPolicyOutput], system_internal_history: SystemInternalHistory) -> PPNPolicyOutput:
        # 1. make prompt text
        dialogue_state = system_internal_history.get_dialogue_state(turn_id=system_internal_history.num_turns)
        prompt_text = self.make_ppn_policy_prompt(
            context_tuples=self._make_context_tuples(system_internal_history),
            dialogue_state=dialogue_state,
            system_da_tuples_or_str=policy_output.system_action,
            max_context_turns=self.max_context_turns,
            multiwoz_db=self.multiwoz_db,
            ppn_policy_bos_token=self.ppn_policy_bos_token,
        )

        # 2. generate response
        other_ppn_token_ids = [[self.ppn_policy_bos_token_id]] + [
            [t_id] for t_id in self.ppn_nlu_token_ids + self.ppn_dst_token_ids + self.ppn_nlg_token_ids
        ]
        generation_result = self._generate_response(prompt_text=prompt_text, bad_words_ids=other_ppn_token_ids)

        # 3. lexicalize system da (tuples or str)
        if isinstance(policy_output, PolicyOutput):
            do_copy, da_tuples = self._lexicalize_system_da_tuples(
                response_ids=generation_result["response_ids"].tolist(),
                dialogue_state=dialogue_state,
            )
            if do_copy:
                system_action = deepcopy(policy_output.system_action)
            else:
                system_action = da_tuples

        elif isinstance(policy_output, WordPolicyOutput):
            do_copy, da_str = self._lexicalize_system_da_str(
                response_ids=generation_result["response_ids"].tolist(),
                dialogue_state=dialogue_state,
            )
            if do_copy:
                system_action = policy_output.system_action
            else:
                system_action = da_str

        # 4. make ppn_policy_output
        ppn_policy_output = PPNPolicyOutput(
            module_name=policy_output.module_name,
            system_action=system_action,
            is_training_example=generation_result.pop("is_training_example"),
            trajectory=generation_result
        )

        return ppn_policy_output
    
    def _lexicalize_system_da_tuples(
            self, response_ids: List[int], dialogue_state: dict
        ) -> Tuple[bool, Optional[List[Tuple[str, str, str, str]]]]:
        # Recover da tuples

        ## 0. convert token IDs to da_tuples
        da_tuples = []
        slot_counter = {}
        do_copy = False
        for token_id in response_ids:
            if token_id == self.policy_tokenizer.eos_token_id:
                break
            elif token_id == self.copy_token_id:
                if not da_tuples:
                    do_copy = True
                    break
                else:
                    continue
            elif token_id in self.ppn_policy_token_ids:
                token = self.policy_tokenizer.convert_ids_to_tokens(token_id)
                try:
                    _, intent, domain, slot, value = token.split(".")
                except ValueError as e:
                    print(response_ids)
                    print(token)
                    raise e
                if value == "*":
                    slot_counter[(intent, domain, slot)] = slot_counter.get((intent, domain, slot), 0) + 1
                    value = str(slot_counter[(intent, domain, slot)])
                da_tuples.append((intent, domain, slot, value))

        if do_copy:
            return True, None

        ## 1. estimate current domain
        ## ConvLab-2/convlab2/policy/vector/vector_multiwoz.py:L146-L151
        cur_domain = None
        for _, domain, _, _ in process_str_action(dialogue_state["user_action"]):
            if domain in self.multiwoz_db_domains:
                cur_domain = domain

        ## 2. convert da_tuples to ConvLab-2's `meta` format
        ## ConvLab-2/convlab2/policy/vector/vector_multiwoz.py:L225
        meta = {}
        for intent, domain, slot, value in da_tuples:
            domain_intent = f"{domain}-{intent}"
            if domain_intent not in meta:
                meta[domain_intent] = []
            meta[domain_intent].append([slot, value])

        ## 3. retrieve entities
        ## ConvLab-2/convlab2/policy/vector/vector_multiwoz.py:L226-L232
        entities = {}
        for domain_intent in meta:
            domain, intent = domain_intent.split('-')
            if domain not in entities and domain.lower() not in ['general', 'booking']:
                constraints = dialogue_state["belief_state"][domain.lower()]['semi'].items()
                entities[domain] = self.multiwoz_db.query(domain.lower(), constraints)
        if cur_domain and cur_domain not in entities:
            constraints = dialogue_state["belief_state"][cur_domain.lower()]['semi'].items()
            entities[cur_domain] = self.multiwoz_db.query(cur_domain.lower(), constraints)

        ## 4. lexicalize
        ## ConvLab-2/convlab2/policy/vector/vector_multiwoz.py:L233
        da_tuples = lexicalize_da_dict_to_tuples(
            meta=meta, entities=entities, state=dialogue_state["belief_state"],
            requestable=self.requestable_intent, cur_domain=cur_domain
        )

        return False, da_tuples

    def _lexicalize_system_da_str(
            self, response_ids: List[int], dialogue_state: dict
        ) -> Tuple[bool, Optional[str]]:
        # Recover da str

        ## 0. convert token IDs to slots (tuple of domain-slot pairs)
        clean_response_ids = []
        domain_slot_tuples = []
        do_copy = False
        for token_id in response_ids:
            if token_id == self.policy_tokenizer.eos_token_id:
                break
            elif token_id == self.copy_token_id:
                if not domain_slot_tuples:
                    do_copy = True
                    break
                else:
                    continue
            elif token_id in self.ppn_policy_token_ids:
                clean_response_ids.append(token_id)
                token = self.policy_tokenizer.convert_ids_to_tokens(token_id)
                try:
                    _, domain, slot = token.split(".")
                except ValueError as e:
                    print(response_ids)
                    print(token)
                    raise e
                domain_slot_tuples.append((domain, slot))
            else:
                clean_response_ids.append(token_id)

        if do_copy:
            return True, None
        
        ## 1. retrieve entities
        entities = {}
        for domain, domain_bs in  dialogue_state["belief_state"].items():
            entities[domain] = self.multiwoz_db.query(
                domain=domain, constraints=domain_bs['semi'].items()
            )
    
        ## 2. lexicalize
        domain_slot_dict = lexicalize_domain_slot_tuples(
            domain_slot_tuples=domain_slot_tuples,
            belief_state=dialogue_state["belief_state"],
            entities=entities,
        )

        ## 3. convert ids to str
        lexed_response = self.policy_tokenizer.decode(clean_response_ids, skip_special_tokens=False).strip()
        for (domain, slot), values in domain_slot_dict.items():
            delex_token = f"system_da.{domain}.{slot}"
            for value in values:
                lexed_response = lexed_response.replace(delex_token, value, 1)
            # lexed_response = lexed_response.replace(delex_token, values[0], 1)

        return False, lexed_response


    def ppn_nlg(self, nlg_output: NLGOutput, system_internal_history: SystemInternalHistory) -> PPNNLGOutput:
        # 1. make prompt text
        system_da_tuples = system_internal_history.get_system_action(turn_id=system_internal_history.num_turns)
        prompt_text = self.make_ppn_nlg_prompt(
            context_tuples=self._make_context_tuples(system_internal_history),
            system_da_tuples=system_da_tuples,
            system_response=nlg_output.system_response,
            max_context_turns=self.max_context_turns,
            ppn_nlg_bos_token=self.ppn_nlg_bos_token,
        )

        # 2. generate response
        other_ppn_token_ids = [[self.ppn_nlg_bos_token_id]] + [
            [t_id] for t_id in self.ppn_nlu_token_ids + self.ppn_dst_token_ids + self.ppn_policy_token_ids
        ]
        generation_result = self._generate_response(prompt_text=prompt_text, bad_words_ids=other_ppn_token_ids)

        # 3. recover system response
        system_response = self.policy_tokenizer.decode(
            generation_result["response_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True
        ).strip()
        if self.copy_token_id in generation_result["response_ids"] and not system_response:
            # If the copy token is the only token in the response, do copy
            system_response = nlg_output.system_response

        # 4. make ppn_nlg_output
        ppn_nlg_output = PPNNLGOutput(
            module_name=nlg_output.module_name,
            system_response=system_response,
            is_training_example=generation_result.pop("is_training_example"),
            trajectory=generation_result
        )

        return ppn_nlg_output

    def postprocess(
        self,
        module_output: Union[NLUOutput, DSTOutput, PolicyOutput, WordPolicyOutput, NLGOutput],
        system_internal_history: SystemInternalHistory,
    ) -> Union[PPNNLUOutput, PPNDSTOutput, PPNPolicyOutput, PPNNLGOutput]:
        """
        Postprocess the module output
        Args:
            module_output: The output of the module
            system_internal_history: The internal history of the system
        Returns:
            The postprocessed output
        """
        if isinstance(module_output, NLUOutput):
            ppn_fn = self.ppn_nlu
        elif isinstance(module_output, DSTOutput):
            ppn_fn = self.ppn_dst
        elif isinstance(module_output, (PolicyOutput, WordPolicyOutput)):
            ppn_fn = self.ppn_policy
        elif isinstance(module_output, NLGOutput):
            ppn_fn = self.ppn_nlg
        else:
            raise ValueError(f"Unsupported module output type: {type(module_output)}")

        ppn_output = ppn_fn(module_output, system_internal_history)

        return ppn_output

    def save(self, output_path: str) -> None:
        policy_output_path = os.path.join(output_path, "policy")
        os.makedirs(policy_output_path, exist_ok=True)
        self.policy_model.save_pretrained(policy_output_path)
        self.policy_tokenizer.save_pretrained(policy_output_path)

        value_output_path = os.path.join(output_path, "value")
        os.makedirs(value_output_path, exist_ok=True)
        self.value_model.save_pretrained(value_output_path)
        self.value_tokenizer.save_pretrained(value_output_path)
