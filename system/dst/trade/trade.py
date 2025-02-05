import os
from copy import deepcopy
from typing import List, Tuple, Dict, Optional, Union

import torch

from convlab2.dst.trade.multiwoz import TRADE

from system.dst.dst import DSTBaseforPPN
from system.module_base import DSTOutput

class MyTRADEDST(DSTBaseforPPN):
    module_name: str = "trade"

    def __init__(self, device: torch.device) -> None:
        super().__init__()

        assert device.index == torch.cuda.current_device(), \
            (f"Specified device {device.index} is not the default "
             f"device {torch.cuda.current_device()}")
        self.dst_core = TRADE()

    def init_session(self) -> None:
        self.dst_core.init_session()
        dialogue_state = self.dst_core.to_cache()
        dialogue_state["history"].append(["sys", "null"])
        return dialogue_state

    def update(self, user_utterance: str, user_action: Optional[List[Tuple[str, str, str, str]]],
               session_over: bool, dialogue_state: dict) -> DSTOutput:
        assert user_action is None, \
            f"DST does not support user action input, but got: {user_action}"
        dialogue_state = deepcopy(dialogue_state)
        dialogue_state["history"].append(["user", user_utterance])
        dialogue_state["user_action"] = user_utterance
        dialogue_state["terminated"] = session_over

        self.dst_core.from_cache(dialogue_state)
        dialogue_state = self.dst_core.update(user_act=user_utterance)

        ds_vector = self.make_dialogue_state_vector(dialogue_state)

        dst_output = DSTOutput(
            module_name=self.module_name,
            dialogue_state=deepcopy(dialogue_state),
            module_state_vector=ds_vector,
        )
        return dst_output

    def update_response(self, system_action: Union[List[Tuple[str, str, str, str]], str],
                        system_response: str, dialogue_state: dict) -> DSTOutput:
        dialogue_state = deepcopy(dialogue_state)
        dialogue_state["system_action"] = system_action
        dialogue_state["history"].append(["sys", system_response])
        dst_output = DSTOutput(
            module_name=self.module_name,
            dialogue_state=dialogue_state,
        )
        return dst_output
