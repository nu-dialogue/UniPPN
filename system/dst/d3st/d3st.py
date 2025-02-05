import os
import re
import random
from copy import deepcopy
from typing import List, Tuple, Dict, Optional, Union
from logging import getLogger

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from convlab2.util.multiwoz.state import default_state
from convlab2.policy.vector.vector_multiwoz import MultiWozVector

from system.dst.d3st.multiwoz_utils import load_schema, SchemaInfo

from system.dst.dst import DSTBaseforPPN
from system.data import (
    DSTOutput,
    VectorData,
)
from utils import set_logger

logger = getLogger(__name__)
set_logger(logger)

abs_dir = os.path.dirname(os.path.abspath(__file__))

def make_description_prefix(schema_info: SchemaInfo) -> Tuple[str, List[str]]:
    slot_id2name = list(schema_info.slots)
    random.shuffle(slot_id2name)
    prefix_pieces = []
    for i, slot_name in enumerate(slot_id2name):
        slot_info = schema_info.slots[slot_name]
        desc = f"{i}={slot_info.description}"
        prefix_pieces.append(desc)
    return " ".join(prefix_pieces), slot_id2name

def context_tupes2str(context_tuples: List[Tuple[str, str]]) -> str:
    speaker_prefix_mapping = {"user": "[user]", "sys": "[system]"}
    context_pieces = []
    for speaker, text in context_tuples:
        if text == "null":
            continue
        prefix = speaker_prefix_mapping[speaker]
        context_pieces.append(f"{prefix} {text}")
    return " ".join(context_pieces)

def parse_output_str(output_str: str) -> Tuple[Dict[int, str], List[int]]:
    if not ("states" in output_str and "req_slots" in output_str):
        raise ValueError(f"Invalid output string: {output_str}")
    _, states_str, _, req_slots_str = re.split(r"\[(states|req_slots)\]", output_str)[1:]

    states = {}
    slot_values = re.split(r"(\d+)=", states_str)[1:]
    for slot, value in zip(slot_values[::2], slot_values[1::2]):
        states[int(slot)] = value.strip()

    req_slots = [int(slot) for slot in req_slots_str.strip().split()]

    return states, req_slots

class D3ST(DSTBaseforPPN):
    module_name: str = "d3st"

    def __init__(self, device) -> None:
        super().__init__()

        # Prepare model
        model_path = os.path.join(abs_dir, "models/ep5-bs4-lr5e-5/checkpoints")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device
        self.max_input_length = 2048

        self.schema_info = load_schema(schema_path=os.path.join(abs_dir, "schema.json"))

    def init_session(self) -> dict:
        init_dialogue_state = default_state()
        init_dialogue_state["history"].append(["sys", "null"])
        return init_dialogue_state

    def update(self, user_utterance: str, user_action: Optional[List[Tuple[str, str, str, str]]], session_over: bool, dialogue_state: dict) -> DSTOutput:
        assert user_action is None, \
            f"D3ST does not support user action input, but got: {user_action}"
        dialogue_state = deepcopy(dialogue_state)
        dialogue_state["history"].append(["user", user_utterance])
        dialogue_state["user_action"] = user_utterance
        dialogue_state["terminated"] = session_over

        # Prepare input
        prefix_str, slot_id2name = make_description_prefix(self.schema_info)
        context_str = context_tupes2str(dialogue_state["history"])
        input_str = f"{prefix_str} {context_str}"

        # Tokenize input
        model_input = self.tokenizer(
            input_str, return_tensors="pt",
            max_length=self.max_input_length, truncation=True
        )

        # Generate output
        output_ids = self.model.generate(
            **model_input.to(self.device),
            max_new_tokens=128,
            num_beams=1,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        output_str = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Reconstruct states and req_slots
        try:
            states, req_slots = parse_output_str(output_str)
            # Reset belief state if successfully parsed
            dialogue_state["belief_state"] = default_state()["belief_state"]
        except ValueError as e:
            # Fallback to previous belief state if failed to parse
            logger.error(f"Failed to parse output string: '{output_str}' (prompt: '{input_str}')")
            states, req_slots = {}, []

        for slot_id, value in states.items():
            try:
                slot_name = slot_id2name[slot_id]
            except IndexError:
                logger.warning(f"Invalid slot_id: {slot_id}='{value}'")
                continue

            domain, slot = slot_name.split("-")
            if slot in dialogue_state["belief_state"][domain]["semi"]:
                dialogue_state["belief_state"][domain]["semi"][slot] = value
            elif slot in dialogue_state["belief_state"][domain]["book"]:
                dialogue_state["belief_state"][domain]["book"][slot] = value
            else:
                if (domain, slot) in [("taxi", "car type"), ("taxi", "phone")]:
                    # ignore taxi type and phone because they are not in the belief state
                    continue
                logger.warning(f"Invalid belief_state slot: {domain} {slot}")

        for slot_id in req_slots:
            try:
                domain, slot = slot_id2name[slot_id].split("-")
            except IndexError:
                logger.warning(f"Invalid slot_id: {slot_id}")
                continue
            if domain not in dialogue_state["request_state"]:
                dialogue_state["request_state"][domain] = {}
            dialogue_state["request_state"][domain][slot] = 0

        ds_vector = self.make_dialogue_state_vector(dialogue_state)

        dst_output = DSTOutput(
            module_name=self.module_name,
            dialogue_state=deepcopy(dialogue_state),
            module_state_vector=ds_vector,
        )
        return dst_output

    def update_response(self, system_action: Union[List[Tuple[str]], str], system_response: str, dialogue_state: Dict) -> DSTOutput:
        dialogue_state = deepcopy(dialogue_state)
        dialogue_state["system_action"] = system_action
        dialogue_state["history"].append(["sys", system_response])
        dst_output = DSTOutput(
            module_name=self.module_name,
            dialogue_state=deepcopy(dialogue_state),
        )
        return dst_output
