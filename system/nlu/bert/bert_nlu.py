from typing import List
from logging import getLogger
import re
import torch

from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from system.data import ModuleOutputBase, VectorData

from system.module_base import NLUBase, NLUOutput
from utils import get_default_device, set_logger

logger = getLogger(__name__)
set_logger(logger)

class MyBERTNLU(NLUBase):
    module_name = "bert_nlu"

    def __init__(self, device=get_default_device()) -> None:
        self.nlu_core = BERTNLU(
            mode="usr",
            config_file="multiwoz_usr_context.json",
            model_file="https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/bert_multiwoz_usr_context.zip",
            device=device
        )

        # Prepare vocab
        intent_vocab = self.nlu_core.dataloader.intent_vocab
        tag_vocab = []
        for bio_intent in self.nlu_core.dataloader.tag_vocab:
            if bio_intent.startswith("B-"): # Use only B- intents
                tag_vocab.append("{}*value".format(bio_intent.replace("B-", "")))
        vocab = intent_vocab + tag_vocab
        self.id2da = {}
        self.da2id = {}
        for i, da in enumerate(vocab):
            self.id2da[i] = da
            self.da2id[da] = i

    @property
    def dim_module_state(self) -> int:
        return len(self.nlu_core.dataloader.intent_vocab) + len(self.nlu_core.dataloader.tag_vocab)

    @property
    def dim_module_output(self) -> int:
        return len(self.id2da)

    def init_session(self) -> None:
        pass

    def predict(self, user_utterance: str, context_tuples: List[str]) -> NLUOutput:
        user_action, intent_probs, tag_probs = self.nlu_core.predict(
            utterance=user_utterance, context=context_tuples, return_probs=True
        )
        module_state_vector = torch.cat([intent_probs, tag_probs])
        assert module_state_vector.shape[0] == self.dim_module_state, \
            f"Module state vector dimension mismatch: {module_state_vector.shape[0]} != {self.dim_module_state}"

        nlu_output = NLUOutput(
            module_name=self.module_name,
            user_action=user_action,
            module_state_vector=module_state_vector,
        )

        return nlu_output

    def vectorize(self, module_output: NLUOutput) -> VectorData:
        user_action = module_output.user_action
        da_vector = torch.zeros(len(self.da2id), dtype=torch.float32)
        domain_memory = set() # Domains in user action
        value_memory = {} # values of slots in user action
        for intent, domain, slot, value in user_action:
            domain_memory.add(domain)
            da_key = "{}-{}+{}".format(domain, intent, slot)
            intent_da = "{}*{}".format(da_key, value)
            tag_da = "{}*value".format(da_key)
            if intent_da in self.da2id:
                da_vector[self.da2id[intent_da]] = 1
            elif tag_da in self.da2id:
                da_vector[self.da2id[tag_da]] = 1
                value_memory[da_key] = value
            else:
                logger.info(f"Unknown DA: {intent_da}")
        
        vec_data = VectorData(
            module_name=self.module_name,
            vector=da_vector,
            data_to_restore={
                "domain_memory": domain_memory,
                "value_memory": value_memory,
            }
        )
            
        return vec_data

    def devectorize(self, vector_data: VectorData) -> NLUOutput:
        da_vector = vector_data.vector
        domain_memory = vector_data.data_to_restore["domain_memory"]
        value_memory = vector_data.data_to_restore["value_memory"]

        da = []
        for index in range(len(da_vector)):
            if not da_vector[index]:
                continue
            domain_intent_slot, value = self.id2da[index].split('*')
            domain, intent, slot = re.split('[+-]', domain_intent_slot)
            # if self.ignore_new_domain and domain not in domain_memory:
            #     continue
            if value == "value":
                # Restore value from memory
                if domain_intent_slot in value_memory:
                    value = value_memory[domain_intent_slot]
                else:
                    continue
            da.append([intent, domain, slot, value])
        nlu_output = NLUOutput(
            module_name=self.module_name,
            user_action=da,
        )
        return nlu_output
