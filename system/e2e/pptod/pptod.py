import os
import re
from copy import deepcopy
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union
from logging import getLogger

import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration

from convlab2.util.multiwoz.state import default_state
from convlab2.util.multiwoz.dbquery import Database

from system.utils import lexicalize_domain_slot_tuples
from system.dst.dst import DSTBaseforPPN
from system.module_base import PolicyBase
from system.data import (
    DSTOutput,
    WordPolicyOutput,
)
from utils import set_logger

logger = getLogger(__name__)
set_logger(logger)

cur_dir = os.path.dirname(os.path.abspath(__file__))

# You can see that the special tokens are defined in the config file:
# models/pretrained/epoch_6_best_ckpt/added_tokens.json
@dataclass
class PPTODSpecialTokens:
    context_sos: str = "<sos_context>"
    context_eos: str = "<eos_context>"
    belief_state_sos: str = "<sos_b>"
    belief_state_eos: str = "<eos_b>"
    db_result_sos: str = "<sos_db>"
    db_result_eos: str = "<eos_db>"
    user_utterance_sos: str = "<sos_u>"
    user_utterance_eos: str = "<eos_u>"
    system_response_sos: str = "<sos_r>"
    system_response_eos: str = "<eos_r>"

@dataclass
class PPTODPrefixes:
    dst: str = "translate dialogue to belief state:"
    rg: str = "translate dialogue to system response:"

def clean_utterance(text: str) -> str:
    # 1. remove return characters
    text = re.sub(r'\n+', ' ', text)

    # 2. remove white spaces before punctuations
    text = re.sub(r'\s([,.!?:;])', r'\1', text) # " ,", " .", " !"
    text = re.sub(r"(\w) '(\w)", r"\1'\2", text) # "I 'm", "I 've"
    text = re.sub(r"(\w) n't", r"\1n't", text) # "do n't"

    return text

def context_tuples2str(context: List[Tuple[str, str]], max_context_turns: int) -> str:
    """
    Convert context tuples to a single string.
    Args:
        - context: List of tuples, each tuple is (role, utterance).
            e.g., [('user', 'Hello'), ('sys', 'Hi')]
        - max_context_turns: Maximum number of context turns to keep. 
          Set to 0 to keep all turns.
    """
    context_strs = [PPTODSpecialTokens.context_sos]

    for role, text in context[-max_context_turns:]:
        if text == 'null':
            continue

        if role == 'user':
            context_strs += [
                PPTODSpecialTokens.user_utterance_sos,
                clean_utterance(text),
                PPTODSpecialTokens.user_utterance_eos
            ]
        elif role == 'sys':
            context_strs += [
                PPTODSpecialTokens.system_response_sos,
                clean_utterance(text),
                PPTODSpecialTokens.system_response_eos
            ]
        else:
            raise ValueError(f"Unknown role: {role}")

    context_strs += [PPTODSpecialTokens.context_eos]

    return ' '.join(context_strs)

def belief_state_dict2str(belief_state: dict) -> str:
    exclude_slots = ["booked"]
    non_values = ["", "not mentioned"]
    flat_bs = defaultdict(list)
    for domain, domain_bs in belief_state.items():
        for constraint_type, constraints in domain_bs.items():
            for slot, value in constraints.items():
                if slot in exclude_slots:
                    continue
                if value in non_values:
                    continue
                flat_bs[domain].append([slot, value])

    bs_strs = [PPTODSpecialTokens.belief_state_sos]
    for domain, constraints in flat_bs.items():
        bs_strs += [f"[{domain}]"]
        for slot, value in constraints:
            bs_strs += [f"{slot}={value}"]
    bs_strs += [PPTODSpecialTokens.belief_state_eos]

    return ' '.join(bs_strs)

def db_result_dict2str(db_result: Dict[str, int]) -> str:
    db_strs = [PPTODSpecialTokens.db_result_sos]
    for domain, count in db_result.items():
        db_strs += [f"[{domain.lower()}]", str(count)]
    db_strs += [PPTODSpecialTokens.db_result_eos]
    return ' '.join(db_strs)

def belief_state_str2dict(belief_state_str: str) -> dict:
    belief_state = default_state()["belief_state"]

    # Remove sos and eos tokens
    belief_state_str = re.sub(
        fr'\s*({PPTODSpecialTokens.belief_state_sos}|{PPTODSpecialTokens.belief_state_eos})\s*',
        '',
        belief_state_str
    )

    # Extract domain and slot-values pairs
    matches = re.findall(r'\[(\w+)\](.*?)(?=\[|$)', belief_state_str)

    for domain, slot_values_str in matches:
        if domain not in belief_state:
            logger.warning(f"Gotten domain is not in belief state: {domain}")
            continue

        slot_values = re.findall(r'(\w+)=([^=]+?)(?=\s+\w+=|\s*$)', slot_values_str)
        for slot, value in slot_values:
            if slot in belief_state[domain]["semi"]:
                constraint_type = "semi"
            elif slot in belief_state[domain]["book"]:
                constraint_type = "book"
            else:
                logger.warning(f"Gotten slot is not in belief state of {domain}: {slot}")
                continue
            belief_state[domain][constraint_type][slot] = value

    return belief_state

def lexicalize_response(
        delexicalized_response: str,
        belief_state: dict,
        entities: Dict[str, list],
        delexicalized_tokens: List[str]
    ) -> str:
    """
    Lexicalize delexicalized system response generated by the model
    Args:
        - delex_response: Delexicalized system response
            e.g., I found [hotel_choice] hotels in the [hotel_area]. How about we start with [hotel_name]?
        - entities: Entities found in the database
            e.g., {"Hotel": [{"name": "Molly's", "area": ...}, ...], "Restaurant": [...], ...}
        - delexicalized_tokens: List of delexicalized tokens in the response
            e.g., ["[hotel_choice]", "[hotel_area]", "[hotel_name]"]
    """
    # 1. Set up key-values mapping for each placeholder
    domain_slot_tuples = []
    for placeholder in delexicalized_tokens:
        domain, slot = placeholder[1:-1].split("_", 1)
        domain_slot_tuples.append((domain, slot))

    # 2. Lexicalize domain-slot tuples
    domain_slot_values = lexicalize_domain_slot_tuples(
        domain_slot_tuples=domain_slot_tuples,
        belief_state=belief_state, entities=entities
    )

    # 3. Replace placeholders with values
    lex_response = delexicalized_response
    for placeholder in delexicalized_tokens:
        domain, slot = placeholder[1:-1].split("_", 1)
        for value in domain_slot_values[(domain, slot)]:
            lex_response = lex_response.replace(placeholder, value, 1)

    return lex_response


class PPTOD:
    def __init__(self, device: torch.device):
        self.device = device

        # model_path = os.path.join(cur_dir, 'models/finetuned/ep20_bs16_ga1_lr5e-5/checkpoints')
        model_path = "ohashi56225/pptod-multiwoz"
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)

        # Truncate from the left side because we want to keep the most recent turns
        self.tokenizer.truncation_side = "left"

        # Find delexicalized tokens
        self.delex_token_ids = []
        for token_id, token in self.tokenizer.added_tokens_decoder.items():
            delex_token_pattern = r'\[.+_.+\]' # e.g. [hotel_name]
            if re.match(delex_token_pattern, token.content):
                self.delex_token_ids.append(token_id)

        self.max_context_turns = 20 # TODO
        self.max_input_length = 1024
        self.max_output_length = 256

        self.db = Database()

    # DST implementation
    def dst_init_session(self):
        init_dialogue_state = default_state()
        init_dialogue_state["history"].append(["sys", "null"])
        return init_dialogue_state

    def dst_update(
            self,
            user_utterance: str,
            user_action: Optional[List[Tuple[str, str, str, str]]],
            session_over: bool,
            dialogue_state: dict,
        ) -> DSTOutput:
        assert user_action is None, \
            f"D3ST does not support user action input, but got: {user_action}"
        dialogue_state = deepcopy(dialogue_state)
        dialogue_state["history"].append(["user", user_utterance])
        dialogue_state["user_action"] = user_utterance
        dialogue_state["terminated"] = session_over

        # Prepare input
        context_str = context_tuples2str(
            context=dialogue_state["history"],
            max_context_turns=self.max_context_turns
        )
        input_str = f"{PPTODPrefixes.dst} {context_str}"

        # Generate belief state
        model_input = self.tokenizer(
            input_str,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True,
        )
        decoder_start_token_id, eos_token_id = self.tokenizer.convert_tokens_to_ids(
            [PPTODSpecialTokens.belief_state_sos, PPTODSpecialTokens.belief_state_eos]
        )

        output_ids = self.model.generate(
            **model_input.to(self.device),
            max_new_tokens=self.max_output_length,
            num_beams=1,
            do_sample=False,
            decoder_start_token_id=decoder_start_token_id,
            eos_token_id=eos_token_id,
        )
        belief_state_str = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)

        # Parse belief state
        belief_state = belief_state_str2dict(belief_state_str)
        dialogue_state["belief_state"] = belief_state

        dst_output = DSTOutput(
            module_name=None,
            dialogue_state=dialogue_state,
        )

        return dst_output

    def dst_update_response(
            self,
            system_action: Union[List[Tuple[str, str, str, str]], str],
            system_response: str,
            dialogue_state: dict,
        ) -> DSTOutput:
        dialogue_state = deepcopy(dialogue_state)
        dialogue_state["system_action"] = system_action
        dialogue_state["history"].append(["sys", system_response])
        dst_output = DSTOutput(
            module_name=None,
            dialogue_state=deepcopy(dialogue_state),
        )
        return dst_output

    # Policy implementation
    def init_session_policy(self):
        pass

    def policy_predict(
            self,
            dialogue_state: dict,
        ) -> WordPolicyOutput:

        # Prepare input
        context_str = context_tuples2str(
            context=dialogue_state["history"],
            max_context_turns=self.max_context_turns
        )

        entities = {}
        for domain, domain_bs in dialogue_state["belief_state"].items():
            entities[domain] = self.db.query(
                domain=domain,
                constraints=domain_bs["semi"].items()
            )
        db_results = {d: len(e) for d, e in entities.items()}
        db_results_str = db_result_dict2str(db_results)

        input_str = f"{PPTODPrefixes.rg} {context_str} {db_results_str}"

        # Generate response
        model_input = self.tokenizer(
            input_str,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True,
        )
        decoder_start_token_id, eos_token_id = self.tokenizer.convert_tokens_to_ids(
            [PPTODSpecialTokens.system_response_sos, PPTODSpecialTokens.system_response_eos]
        )

        output_ids = self.model.generate(
            **model_input.to(self.device),
            max_new_tokens=self.max_output_length,
            num_beams=1,
            do_sample=False,
            decoder_start_token_id=decoder_start_token_id,
            eos_token_id=eos_token_id,
        )
        # skip the sos and eos tokens
        delex_response = self.tokenizer.decode(output_ids[0, 1:-1], skip_special_tokens=False).strip()

        # Find delexicalized tokens
        delex_tokens = []
        for token_id in output_ids[0, 1:-1].tolist():
            if token_id in self.delex_token_ids:
                delex_tokens.append(
                    self.tokenizer.convert_ids_to_tokens(token_id)
                )

        # Lexicalize the response
        lex_response = lexicalize_response(
            delexicalized_response=delex_response,
            belief_state=dialogue_state["belief_state"],
            entities=entities,
            delexicalized_tokens=delex_tokens
        )

        policy_output = WordPolicyOutput(
            module_name=None,
            system_action=lex_response,
            delexicalized_response=delex_response,
            active_domain=None,
        )

        return policy_output


class PPTODasDST(DSTBaseforPPN):
    module_name = "pptod"

    def __init__(self, pptod: PPTOD):
        super().__init__()
        self.pptod = pptod

    def init_session(self):
        return self.pptod.dst_init_session()
    
    def update(self, *args, **kwargs):
        dst_output = self.pptod.dst_update(*args, **kwargs)
        ds_vector = self.make_dialogue_state_vector(dst_output.dialogue_state)
        dst_output.module_name = self.module_name
        dst_output.module_state_vector = ds_vector
        return dst_output
    
    def update_response(self, *args, **kwargs):
        dst_output = self.pptod.dst_update_response(*args, **kwargs)
        dst_output.module_name = self.module_name
        return dst_output
    
class PPTODasPolicy(PolicyBase):
    module_name = "pptod"

    def __init__(self, pptod: PPTOD):
        super().__init__()
        self.pptod = pptod
    
    @property
    def dim_module_state(self) -> int:
        return 0

    @property
    def dim_module_output(self) -> int:
        return 0

    def init_session(self):
        return self.pptod.init_session_policy()
    
    def predict(self, *args, **kwargs):
        policy_output = self.pptod.policy_predict(*args, **kwargs)
        policy_output.module_name = self.module_name
        return policy_output

def build_e2e_as_modules(device: torch.device) -> Tuple[PPTODasDST, PPTODasPolicy]:
    pptod = PPTOD(device=device)
    return PPTODasDST(pptod=pptod), PPTODasPolicy(pptod=pptod)
