# This script was adapted from the Hugging Face's run_clm.py script.
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
# Original code licensed under the Apache License 2.0.
# This modified code is licensed under the CC BY-NC 4.0 license.


import os
import sys
import json
import math
import random
import logging
import warnings
from glob import glob
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Literal, Union, Callable

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import datasets
import evaluate

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.tokenization_utils_base import BatchEncoding

from sentence_transformers import SentenceTransformer

from convlab2.util.multiwoz.state import default_state
from convlab2.util.multiwoz.dbquery import Database

from dialogue_simulation.dialogue_sampler import SessionLog
from system import SYSTEM_LIST
from system.ppn.uni_ppn import UniPPN
from system.ppn.uni_ppn_utils import (
    user_da_tuples2tokens,
    extract_updated_belief_state,
    belief_state2tokens,
    request_state2tokens,
    system_da_tuples2tokens,
    context_tuples2str,
    delexicalize_system_da_tuple,
    redelexicalize_larl_response,
    redelexicalize_pptod_response,
    redelexicalize_llmicl_response,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    init_data_dir: str = field(
        metadata={
            "help": "Path to directory containing sampled dialogue data (session_logs/*.pkl) with base_system."
        },
    )
    num_turns: Optional[int] = field(
        default=None,
        metadata={"help": "Number of turns to sample from the session logs."},
    )
    num_samples_per_turn: int = field(
        default=5,
        metadata={"help": "The number of samples to be drawn from the similar contexts."},
    )
    
    max_context_turns: int = field(
        default=3,
        metadata={"help": "maximum number of context turns to be used as input."},
    )

    ppn_target_modules: List[str] = field(
        default_factory=lambda: ["nlu", "dst", "policy", "nlg"],
        metadata={"help": "The target modules to be trained."},
    )
    train_only_embeddings: bool = field(
        default=True,
        metadata={"help": "Train only the embedding matrix."},
    )

    only_successful_dialogue: bool = field(
        default=False,
        metadata={"help": "Sample only successful dialogues."},
    )
    postprocessing_ratio: float = field(
        default=0.9,
        metadata={"help": "The ratio of learning post-processing module's output instead of generating it from scratch."},
    )
    embedding_model_name: str = field(
        default="thenlper/gte-base",
        metadata={"help": "The name of the context embedding model to compute the context cossimilarity."},
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_ratio: Optional[float] = field(
        default=0.2,
        metadata={
            "help": "The ratio of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum length of the prompt text."},
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum length of the target text."},
    )

class DialogueModel:
    def __init__(self, session_logs: List[SessionLog], module_combination: Dict[str, str]):
        # Load system configuration
        self.module_combination = module_combination
        self.word_dst = module_combination["nlu"] is None
        self.word_policy = module_combination["nlg"] is None

        self.session_logs = session_logs

        # Load sampled dialogue data
        data = []
        for session_log in session_logs:
            dialogue_id = f"{session_log.iteration_id}-{session_log.process_id}-{session_log.episode_id}"
            context_tuples = []
            for system_turn in session_log.system_turns:
                turn_id = system_turn["turn_id"]
                context_tuples.append(("user", system_turn["user_utterance"]))
                data.append({
                    "turn_id": f"{dialogue_id}-{system_turn['turn_id']}",
                    "task_success": session_log.task_success,
                    "context": deepcopy(context_tuples),
                    "user_da": system_turn["nlu"]["user_action"],
                    "dialogue_state": system_turn["dst"]["dialogue_state"],
                    "system_da": system_turn["policy"]["system_action"],
                    "system_response": system_turn["nlg"]["system_response"],
                })
                context_tuples.append(("system", system_turn["system_response"]))
        self.data = pd.DataFrame(data).set_index("turn_id")

        self.predefined_user_da_tuples = []
        if not self.word_dst:
            for line in open("ConvLab-2/data/multiwoz/usr_da_voc.txt"):
                domain, intent, slot, value = line.strip().split("-")
                self.predefined_user_da_tuples.append((intent, domain, slot, value))

        # self.requestable_slots = {}
        # for line in open("ConvLab-2/data/multiwoz/usr_da_voc.txt"):
        #     domain, intent, slot, _ = line.strip().split("-")
        #     if intent == "Request":
        #         if domain.lower() not in self.requestable_slots:
        #             self.requestable_slots[domain.lower()] = []
        #         self.requestable_slots[domain.lower()].append(REF_SYS_DA[domain].get(slot, slot))

        self.predefined_system_da_tuples = []
        if not self.word_policy:
            for line in open("ConvLab-2/data/multiwoz/sys_da_voc.txt"):
                domain, intent, slot, value = line.strip().split("-")
                self.predefined_system_da_tuples.append((intent, domain, slot, value))

        self.context_cossim = None

    @classmethod
    def from_pickle(cls, session_logs_dpath: str):
        session_logs = []
        for session_log_fpath in glob(os.path.join(session_logs_dpath, "*.pkl")):
            session_logs.append(SessionLog.from_pickle(session_log_fpath))
        
        args = json.load(open(os.path.join(os.path.dirname(session_logs_dpath), "args.json")))
        system_name = args["dialogue_sampling_args"]["system_name"]
        module_combination = SYSTEM_LIST[system_name]
        return cls(session_logs=session_logs, module_combination=module_combination)

    def get_all_da_tuples(
            self,
            side: Literal["user", "system"],
            from_data: bool = True,
            from_predifined: bool = True
        ) -> List[Tuple[str, str, str, str]]:
        if (self.word_dst and side == "user") or (self.word_policy and side == "system"):
            # There is no DAs for word-level DST and Policy
            return []

        all_da_tuples = []
        if from_data:
            all_da_tuples += self.data[f"{side}_da"].sum()
        if from_predifined:
            all_da_tuples += getattr(self, f"predefined_{side}_da_tuples")
        return all_da_tuples

    def get_system_da_value_voc(self) -> Dict[Tuple[str, str, str], List[str]]:
        value_voc = {}
        for intent, domain, slot, value in self.get_all_da_tuples("system"):
            if (intent, domain, slot) not in value_voc:
                value_voc[(intent, domain, slot)] = []
            if value not in value_voc[(intent, domain, slot)]:
                value_voc[(intent, domain, slot)].append(value)
        return value_voc

    def get_all_delexicalized_system_slot_tokens(
            self,
            redelexicalize_func: Optional[Callable] = None,
        ) -> List[str]:
        assert self.word_policy, "Only word-level policy is supported."

        delex_sys_slot_tokens = []
        for session_log in self.session_logs:
            for policy_output in session_log.system_internal_history.policy_output_history:
                _, slot_tokens = redelexicalize_func(
                    delexicalized_response=policy_output.delexicalized_response,
                    active_domain=policy_output.active_domain,
                )
                delex_sys_slot_tokens += slot_tokens
        return delex_sys_slot_tokens

    def compute_context_cossim(
            self,
            embedding_model_name: str,
            device: str,
            batch_size: int,
            max_context_turns: int
        ) -> pd.DataFrame:
        model = SentenceTransformer(
            embedding_model_name,
            device=device,
            tokenizer_kwargs={"truncation_side": "left"}
        )
        context_strs = self.data.context.apply(
            context_tuples2str,
            max_context_turns=max_context_turns,
            speaker2prefix_mapping={"user": "User:", "system": "System:"}
        )
        embeddings = model.encode(
            sentences=context_strs.tolist(),
            batch_size=batch_size,
            show_progress_bar=True
        )
        cossims = embeddings @ embeddings.T

        cossims[cossims >= (1.0 - 1e-6)] = 0 # exclude self-similarity

        cossim_df = pd.DataFrame(cossims, index=self.data.index, columns=self.data.index)

        self.context_cossim = cossim_df

    def sample_similar_contexts(self, turn_id: int, top_k: int, num_samples: int) -> List[int]:
        if self.context_cossim is None:
            raise ValueError("Context cossimilarity matrix is not computed.")
        sampled_turn_ids = self.context_cossim[turn_id].sort_values(
            ascending=False
        ).head(top_k).sample(num_samples).index
        return sampled_turn_ids
 
    def sample_similar_da_tuples_or_str(
            self,
            turn_id: int,
            da_tuples_or_str: Union[List[Tuple[str, str, str, str]], str],
            side: Literal["user", "system"],
        ) -> List[Tuple[str, str, str, str]]:
        sampled_turn_ids = self.sample_similar_contexts(turn_id, top_k=100, num_samples=10)
        for id_ in sampled_turn_ids:
            sampled_da_tuples_or_str = self.data.loc[id_][f"{side}_da"]
            if sampled_da_tuples_or_str:
                break

        if isinstance(da_tuples_or_str, str):
            return sampled_da_tuples_or_str

        # Copy values from the original DA tuples
        base_da_dict = {(i,d,s): v for i,d,s,v in da_tuples_or_str}
        sampled_da_dict = {(i,d,s): v for i,d,s,v in sampled_da_tuples_or_str}
        for i,d,s in sampled_da_dict:
            if (i,d,s) in base_da_dict:
                sampled_da_dict[(i,d,s)] = base_da_dict[(i,d,s)]
        return [[i,d,s,v] for (i,d,s),v in sampled_da_dict.items()]

    def sample_similar_belief_state(
            self,
            turn_id: int,
            belief_state: Dict[str, Dict[str, str]],
            active_domain: Optional[str] = None
        ) -> Dict[str, Dict[str, str]]:
        sampled_turn_id = self.sample_similar_contexts(turn_id, top_k=100, num_samples=1)[0]
        sampled_bs = self.data.loc[sampled_turn_id].dialogue_state["belief_state"]

        modified_bs = deepcopy(belief_state)
        if active_domain:
            modified_bs[active_domain] = deepcopy(sampled_bs[active_domain])
        return modified_bs
    
    def sample_similar_request_state(
            self,
            turn_id: int,
            request_state: Dict[str, Dict[str, int]],
            active_domain: Optional[str] = None
        ) -> Dict[str, Dict[str, int]]:
        sampled_turn_id = self.sample_similar_contexts(turn_id, top_k=100, num_samples=1)[0]
        sampled_rs = self.data.loc[sampled_turn_id].dialogue_state["request_state"]

        modified_rs = deepcopy(request_state)
        if active_domain and active_domain in sampled_rs:
            modified_rs[active_domain] = deepcopy(sampled_rs[active_domain])
        return modified_rs
    
    def sample_similar_system_response(
            self,
            turn_id: int,
        ) -> str:
        sampled_turn_id = self.sample_similar_contexts(turn_id, top_k=100, num_samples=1)[0]
        sampled_response = self.data.loc[sampled_turn_id].system_response
        return sampled_response

def estimate_activate_domain(prev_ds, cur_ds):
    prev_bs = prev_ds["belief_state"]
    cur_bs = cur_ds["belief_state"]
    for domain, constraints in cur_bs.items():
        for constraint_type, slot_values in constraints.items():
            for slot, value in slot_values.items():
                if value and not prev_bs[domain][constraint_type][slot]:
                    return domain

    prev_rs = prev_ds["request_state"]
    cur_rs = cur_ds["request_state"]
    for domain in cur_rs:
        if domain not in prev_rs:
            return domain
        for slot in cur_rs[domain]:
            if slot not in prev_rs[domain]:
                return domain
    return None

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)


    # Load sampled dialogue data
    init_data_args = json.load(open(os.path.join(data_args.init_data_dir, "args.json")))
    system_name = init_data_args["dialogue_sampling_args"]["system_name"]
    session_logs = []
    for fpath in tqdm(
        glob(os.path.join(data_args.init_data_dir, "session_logs/*.pkl")),
        desc="Loading session logs",
    ):
        session_logs.append(SessionLog.from_pickle(fpath))

    # Validate target modules
    module_combination = SYSTEM_LIST[system_name]
    for module in data_args.ppn_target_modules:
        assert module_combination[module] is not None, \
            (f"{module} is not included in the system {system_name}: "
             f"{module_combination[module]}")

    # Build the dialogue model
    multiwoz_db = Database()
    dialogue_model = DialogueModel(
        session_logs=session_logs,
        module_combination=module_combination
    )
    dialogue_model.compute_context_cossim(
        embedding_model_name=data_args.embedding_model_name,
        device=training_args.device,
        batch_size=128,
        max_context_turns=data_args.max_context_turns
    )

    system_da_value_voc = dialogue_model.get_system_da_value_voc()

    if not dialogue_model.word_policy:
        redelex_func = None
    else:
        if module_combination["policy"] in ["larl", "lava"]:
            redelex_func = redelexicalize_larl_response
        elif module_combination["policy"] == "pptod":
            redelex_func = redelexicalize_pptod_response
        elif module_combination["policy"] == "gpt4om":
            redelex_func = redelexicalize_llmicl_response
        else:
            raise NotImplementedError(module_combination["policy"])

    dicts = []
    for sample_id in range(data_args.num_samples_per_turn):
        for session_log in tqdm(session_logs, desc=f"Generating prompt-target pairs (Sample ID: {sample_id})"):
            if data_args.only_successful_dialogue and not session_log.task_success:
                continue
            dialogue_id = f"{session_log.iteration_id}-{session_log.process_id}-{session_log.episode_id}"
            context_tuples = []
            prev_dialogue_state = default_state()
            for turn in session_log.system_turns:
                turn_id = turn["turn_id"]
                user_da_tuples_or_none = turn["nlu"]["user_action"]
                dialogue_state = turn["dst"]["dialogue_state"]
                system_da_tuples_or_str = turn["policy"]["system_action"]
                system_response = turn["system_response"]

                context_tuples.append(("user", turn["user_utterance"]))

                # PPN NLU
                if "nlu" in data_args.ppn_target_modules:
                    assert user_da_tuples_or_none is not None, f"User DA is None: {turn}"
                    if random.random() < data_args.postprocessing_ratio:
                        src_user_da_tuples = dialogue_model.sample_similar_da_tuples_or_str(
                            turn_id=f"{dialogue_id}-{turn_id}",
                            da_tuples_or_str=user_da_tuples_or_none,
                            side="user"
                        )
                    else:
                        src_user_da_tuples = user_da_tuples_or_none

                    ppn_nlu_prompt = UniPPN.make_ppn_nlu_prompt(
                        context_tuples=context_tuples,
                        user_da_tuples=src_user_da_tuples,
                        max_context_turns=data_args.max_context_turns,
                        ppn_nlu_bos_token=UniPPN.ppn_nlu_bos_token,
                    )

                    if src_user_da_tuples != user_da_tuples_or_none:
                        user_da_tokens = user_da_tuples2tokens(user_da_tuples_or_none)
                        ppn_nlu_output = " " + " ".join([f"{token} {value}" for token, value in user_da_tokens])
                    else:
                        ppn_nlu_output = " " + UniPPN.copy_token

                    dicts.append({"prompt_text": ppn_nlu_prompt, "target_text": ppn_nlu_output})


                # PPN DST
                if "dst" in data_args.ppn_target_modules:
                    updated_bs = extract_updated_belief_state(
                        prev_belief_state=prev_dialogue_state["belief_state"],
                        belief_state=dialogue_state["belief_state"]
                    )
                    if random.random() < data_args.postprocessing_ratio:
                        active_domain = estimate_activate_domain(prev_ds=prev_dialogue_state, cur_ds=dialogue_state)
                        sampled_bs = dialogue_model.sample_similar_belief_state(
                            turn_id=f"{dialogue_id}-{turn_id}",
                            belief_state=dialogue_state["belief_state"],
                            active_domain=active_domain
                        )
                        src_updated_bs = extract_updated_belief_state(
                            prev_belief_state=prev_dialogue_state["belief_state"],
                            belief_state=sampled_bs
                        )
                    else:
                        src_updated_bs = updated_bs

                    ppn_dst_prompt = UniPPN.make_ppn_dst_prompt(
                        context_tuples=context_tuples,
                        user_da_tuples_or_none=user_da_tuples_or_none,
                        updated_belief_state=src_updated_bs,
                        max_context_turns=data_args.max_context_turns,
                        ppn_dst_bos_token=UniPPN.ppn_dst_bos_token,
                    )

                    if src_updated_bs != updated_bs:
                        bs_tokens = belief_state2tokens(updated_bs)
                        ppn_dst_output = " " + " ".join([f"{t} {v}" for t, v in bs_tokens if v])
                    else:
                        ppn_dst_output = " " + UniPPN.copy_token


                    dicts.append({"prompt_text": ppn_dst_prompt, "target_text": ppn_dst_output})


                # PPN Policy
                if "policy" in data_args.ppn_target_modules:
                    if random.random() < data_args.postprocessing_ratio:
                        src_system_da_tuples_or_str = dialogue_model.sample_similar_da_tuples_or_str(
                            turn_id=f"{dialogue_id}-{turn_id}",
                            da_tuples_or_str=system_da_tuples_or_str,
                            side="system"
                        )
                    else:
                        src_system_da_tuples_or_str = system_da_tuples_or_str

                    ppn_policy_prompt = UniPPN.make_ppn_policy_prompt(
                        context_tuples=context_tuples,
                        dialogue_state=dialogue_state,
                        system_da_tuples_or_str=src_system_da_tuples_or_str,
                        max_context_turns=data_args.max_context_turns,
                        multiwoz_db=multiwoz_db,
                        ppn_policy_bos_token=UniPPN.ppn_policy_bos_token,
                    )

                    if src_system_da_tuples_or_str != system_da_tuples_or_str:
                        if isinstance(system_da_tuples_or_str, list):
                            delex_sys_da_tuples = delexicalize_system_da_tuple(
                                system_da_tuples=system_da_tuples_or_str,
                                system_da_value_voc=system_da_value_voc
                            )
                            sys_da_tokens = system_da_tuples2tokens(delex_sys_da_tuples)
                            ppn_policy_output = " " + " ".join(sys_da_tokens)
                        else: # Word-level policy's delexicalization
                            word_policy_output = session_log.system_internal_history.policy_output_history[turn_id]
                            delex_sys_resp, _ = redelex_func(
                                delexicalized_response=word_policy_output.delexicalized_response,
                                active_domain=word_policy_output.active_domain,
                            )
                            ppn_policy_output = " " + delex_sys_resp
                    else:
                        ppn_policy_output = " " + UniPPN.copy_token

                    dicts.append({"prompt_text": ppn_policy_prompt, "target_text": ppn_policy_output})

                # PPN NLG
                if "nlg" in data_args.ppn_target_modules:
                    assert isinstance(system_da_tuples_or_str, list), \
                        f"System DA is not a list: {system_da_tuples_or_str}"
                    if random.random() < data_args.postprocessing_ratio:
                        src_system_response = dialogue_model.sample_similar_system_response(
                            turn_id=f"{dialogue_id}-{turn_id}"
                        )
                    else:
                        src_system_response = system_response

                    ppn_nlg_prompt = UniPPN.make_ppn_nlg_prompt(
                        context_tuples=context_tuples,
                        system_da_tuples=system_da_tuples_or_str,
                        system_response=src_system_response,
                        max_context_turns=data_args.max_context_turns,
                        ppn_nlg_bos_token=UniPPN.ppn_nlg_bos_token,
                    )
                    
                    if src_system_response != system_response:
                        ppn_nlg_output = " " + system_response
                    else:
                        ppn_nlg_output = " " + UniPPN.copy_token

                    dicts.append({"prompt_text": ppn_nlg_prompt, "target_text": ppn_nlg_output})


                context_tuples.append(("system", system_response))
                prev_dialogue_state = dialogue_state

    ppn_nlu_vocab = []
    if "nlu" in data_args.ppn_target_modules:
        ppn_nlu_vocab += [UniPPN.ppn_nlu_bos_token]
        ppn_nlu_vocab += sorted(set(
            [t for t, _ in user_da_tuples2tokens(
                dialogue_model.get_all_da_tuples("user")
            )]
        ))

    ppn_dst_vocab = []
    if "dst" in data_args.ppn_target_modules:
        ppn_dst_vocab += [UniPPN.ppn_dst_bos_token]
        ppn_dst_vocab += [token for token, _ in belief_state2tokens(default_state()["belief_state"])]
        # new_token_vocab += request_state2tokens(dialogue_model.requestable_slots)

    ppn_policy_vocab = []
    if "policy" in data_args.ppn_target_modules:
        ppn_policy_vocab += [UniPPN.ppn_policy_bos_token]
        if dialogue_model.word_policy:
            ppn_policy_vocab += sorted(set(dialogue_model.get_all_delexicalized_system_slot_tokens(
                redelexicalize_func=redelex_func
            )))
        else:
            ppn_policy_vocab += sorted(set(
                system_da_tuples2tokens(
                    delexicalize_system_da_tuple(
                        system_da_tuples=dialogue_model.get_all_da_tuples("system"),
                        system_da_value_voc=system_da_value_voc
                    )
                )
            ))

    ppn_nlg_vocab = []
    if "nlg" in data_args.ppn_target_modules:
        ppn_nlg_vocab += [UniPPN.ppn_nlg_bos_token]

    new_token_vocab =[UniPPN.copy_token] + ppn_nlu_vocab + ppn_dst_vocab + ppn_policy_vocab + ppn_nlg_vocab
    json.dump(new_token_vocab, open("new_token_vocab.json", "w"), indent=4)


    # Sample a subset of the data
    if data_args.num_turns is not None:
        dicts = random.sample(dicts, data_args.num_samples_per_turn*data_args.num_turns*len(data_args.ppn_target_modules))

    # Split the data into training and validation sets
    raw_datasets = datasets.Dataset.from_list(dicts).train_test_split(
        test_size=data_args.validation_split_ratio,
        seed=training_args.seed
    )
    raw_datasets["validation"] = raw_datasets.pop("test")
    raw_datasets["validation"].to_json(f"val-ppr{data_args.postprocessing_ratio}.json")

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
    )

    # Add da tokens as special tokens
    new_token_vocab = list(filter(
        lambda v: v not in tokenizer.additional_special_tokens,
        new_token_vocab
    ))

    if new_token_vocab:
        logger.info(f"Adding new tokens to the tokenizer: {new_token_vocab}")
        # Retain the original token ids
        new_token_original_ids = tokenizer(new_token_vocab).input_ids

        # Add new tokens to the tokenizer
        tokenizer.add_special_tokens({"additional_special_tokens": new_token_vocab})

        # Resize the embeddings
        model.resize_token_embeddings(len(tokenizer))

        # Initialize new embeddings with average of the original embeddings
        # https://github.com/huggingface/transformers/issues/1413
        with torch.no_grad():
            for new_token, ori_ids in zip(new_token_vocab, new_token_original_ids):
                new_id = tokenizer.convert_tokens_to_ids(new_token)
                model.get_input_embeddings().weight[new_id] = model.get_input_embeddings().weight[ori_ids].mean(dim=0)
                model.get_output_embeddings().weight[new_id] = model.get_output_embeddings().weight[ori_ids].mean(dim=0)
    else:
        logger.info("No new tokens to add to the tokenizer.")

    # Train only embeddings
    if data_args.train_only_embeddings:
        ## Frozen the entire model
        for param in model.parameters():
            param.requires_grad = False
        # Unfrozen the embeddings
        for param in model.get_input_embeddings().parameters():
            param.requires_grad = True
        for param in model.get_output_embeddings().parameters():
            param.requires_grad = True

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            prompt_texts = examples["prompt_text"]
            target_texts = [text+tokenizer.eos_token for text in examples["target_text"]]

            tokenizer.truncation_side = "left"
            prompt_ids = tokenizer(prompt_texts, add_special_tokens=False, max_length=data_args.max_prompt_length, truncation=True).input_ids

            tokenizer.truncation_side = "right"
            target_ids = tokenizer(target_texts, add_special_tokens=False, max_length=data_args.max_target_length, truncation=True).input_ids

            tokenized_output = {
                "input_ids": [p_ids + t_ids for p_ids, t_ids in zip(prompt_ids, target_ids)],
                "labels": [[-100] * len(p_ids) + t_ids for p_ids, t_ids in zip(prompt_ids, target_ids)]
                # "labels": [[-100] * len(p_ids[:-1]) + p_ids[-1:] + t_ids for p_ids, t_ids in zip(prompt_ids, target_ids)]
            }

        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )

        return BatchEncoding(tokenized_output)

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True, # default batch size is 1000
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        copy_vocab_ids = [tokenizer.convert_tokens_to_ids(UniPPN.copy_token)]
        ppn_nlu_vocab_ids = tokenizer.convert_tokens_to_ids(ppn_nlu_vocab)
        ppn_dst_vocab_ids = tokenizer.convert_tokens_to_ids(ppn_dst_vocab)
        ppn_policy_vocab_ids = tokenizer.convert_tokens_to_ids(ppn_policy_vocab)
        # ppn_nlg_vocab_ids = tokenizer.convert_tokens_to_ids(ppn_nlg_vocab)
        def _detect_target_modules(labels: np.ndarray) -> Dict[str, np.ndarray]:
            """
            Detect the target modules from the labels.
            """
            is_copy_example = np.isin(labels, copy_vocab_ids).any(axis=1)
            is_nlu_example = np.isin(labels, ppn_nlu_vocab_ids).any(axis=1)
            is_dst_example = np.isin(labels, ppn_dst_vocab_ids).any(axis=1)
            is_policy_example = np.isin(labels, ppn_policy_vocab_ids).any(axis=1)
            is_nlg_example = ~(is_copy_example | is_nlu_example | is_dst_example | is_policy_example)
            # is_nlg_example = np.isin(labels, ppn_nlg_vocab_ids).any(axis=1)

            example_indices = {
                "copy": np.where(is_copy_example)[0],
                "nlu": np.where(is_nlu_example)[0],
                "dst": np.where(is_dst_example)[0],
                "policy": np.where(is_policy_example)[0],
                "nlg": np.where(is_nlg_example)[0],
            }
            return example_indices

        def _compute_accuracy(preds: np.ndarray, labels: np.ndarray):
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)

            # Ignore -100 and pad_token_id
            ignore_ids = [-100, tokenizer.pad_token_id]
            ignore_indices = sum(labels == i for i in ignore_ids).astype(bool)
            labels_to_eval = labels[~ignore_indices]
            preds_to_eval = preds[~ignore_indices]

            return metric.compute(predictions=preds_to_eval, references=labels_to_eval)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            module_example_indices = _detect_target_modules(labels)
            result = {}
            for module, indices in module_example_indices.items():
                if len(indices) == 0:
                    continue
                r = _compute_accuracy(preds[indices], labels[indices])
                result.update({f"{module}/{k}": v for k, v in r.items()})
            return result

    def data_collator(features: List[Dict[str, Any]]) -> BatchEncoding:
        padding = "longest"
        return_tensors = "pt"
        tokenizer.padding_side = "right"
        
        input_features = [{"input_ids": f["input_ids"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Pad input_ids
        batch = tokenizer.pad(
            input_features,
            padding=padding,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            return_tensors=return_tensors,
        )
        # Pad labels
        batch["labels"] = tokenizer.pad(
            label_features,
            padding=padding,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            return_tensors=return_tensors,
        ).input_ids

        return batch


    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=None)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
