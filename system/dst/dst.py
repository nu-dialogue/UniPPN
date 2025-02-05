from copy import deepcopy

import torch
from convlab2.util.multiwoz.state import default_state
from convlab2.policy.vector.vector_multiwoz import MultiWozVector

from system.module_base import DSTBase
from system.data import (
    DSTOutput,
    VectorData,
)

class DSTBaseforPPN(DSTBase):
    def __init__(self):
        # Prepare vocab
        self.mwoz_vector = MultiWozVector()

        self.belief_domains = self.mwoz_vector.belief_domains
        self.belief_state_dim = self.mwoz_vector.belief_state_dim

        belief_state_vocab = []
        for domain in self.belief_domains:
            for slot in default_state()["belief_state"][domain.lower()]["semi"]:
                belief_state_vocab.append("{}-semi-{}".format(domain.lower(), slot))
        
        assert len(belief_state_vocab) == self.belief_state_dim
        self.id2slot = {}
        self.slot2id = {}
        for i, slot in enumerate(belief_state_vocab):
            self.id2slot[i] = slot
            self.slot2id[slot] = i

    @property
    def dim_module_state(self) -> int:
        return self.mwoz_vector.state_dim

    @property
    def dim_module_output(self) -> int:
        return self.belief_state_dim

    def make_dialogue_state_vector(self, dialogue_state: dict) -> torch.Tensor:
        ds_vector = self.mwoz_vector.state_vectorize(dialogue_state)
        ds_vector = torch.Tensor(ds_vector)
        assert ds_vector.shape[0] == self.dim_module_state, \
            f"Module state vector dimension mismatch: {len(ds_vector)} != {self.dim_module_state}"
        return ds_vector

    def vectorize(self, module_output: DSTOutput) -> VectorData:
        belief_state = module_output.dialogue_state["belief_state"]

        bs_vec = torch.zeros(len(self.slot2id), dtype=torch.float32)
        for domain, domain_bs in belief_state.items():
            for slot, value in domain_bs["semi"].items():
                if value:
                    slot_key = "{}-semi-{}".format(domain.lower(), slot)
                    bs_vec[self.slot2id[slot_key]] = 1
        vec_data = VectorData(
            module_name=self.module_name,
            vector=bs_vec,
            data_to_restore={"dialogue_state": module_output.dialogue_state},
        )
        return vec_data
    
    def devectorize(self, vector_data: VectorData) -> DSTOutput:
        bs_vec = vector_data.vector
        dialogue_state = deepcopy(vector_data.data_to_restore["dialogue_state"])

        processed_bs, deleted_bs = [], []
        for i, is_active in enumerate(bs_vec):
            slot_key = self.id2slot[i]
            if is_active:
                processed_bs.append(slot_key)
            else:
                domain, semi, slot = slot_key.split("-")
                try:
                    if dialogue_state["belief_state"][domain][semi][slot]:
                        deleted_bs.append(slot_key)
                    dialogue_state["belief_state"][domain][semi][slot] = ""
                except:
                    pass
        dst_output = DSTOutput(
            module_name=self.module_name,
            dialogue_state=dialogue_state,
        )
        return dst_output
