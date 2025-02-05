from logging import getLogger
from typing import List, Tuple

import torch
from convlab2.nlg.sclstm.multiwoz import SCLSTM

from system.data import NLGOutput
from system.module_base import NLGBase
from system.nlg.utils import remove_ws_before_punctuation

from utils import set_logger
logger = getLogger(__name__)
set_logger(logger)

def clean_system_response(response):
    tokens = []
    for token in response.split():
        if token not in ["UNK_token"]:
            tokens.append(token)
    return " ".join(tokens)

class MySCLSTM(NLGBase):
    module_name: str = "sclstm_nlg"

    def __init__(self, device: torch.device) -> None:
        torch.cuda.set_device(device)
        self.nlg_core = SCLSTM(
            model_file="https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/nlg_sclstm_multiwoz.zip",
            use_cuda=True,
        )

    @property
    def dim_module_state(self) -> int:
        return 0
    
    @property
    def dim_module_output(self) -> int:
        return 0
    
    def init_session(self) -> None:
        pass

    def generate(self, system_action: List[Tuple[str, str, str, str]]) -> NLGOutput:
        self.nlg_core.args["beam_size"] = 1 # greedy decoding
        with torch.no_grad():
            system_response = self.nlg_core.generate(system_action)
        system_response = clean_system_response(system_response)
        system_response = remove_ws_before_punctuation(system_response)
        nlg_output = NLGOutput(
            module_name=self.module_name,
            system_response=system_response,
        )
        return nlg_output
