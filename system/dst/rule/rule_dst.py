from copy import deepcopy
from typing import List, Optional, Tuple, Union
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.util.multiwoz.state import default_state

from system.dst.dst import DSTBaseforPPN
from system.data import DSTOutput

class MyRuleDST(DSTBaseforPPN):
    module_name: str = "rule_dst"

    def __init__(self) -> None:
        super().__init__()
        self.dst_core = RuleDST()

    def init_session(self) -> dict:
        init_dialogue_state = default_state()
        init_dialogue_state["history"].append(["sys", "null"])
        return init_dialogue_state

    def update(self, user_utterance: str, user_action: Union[List[Tuple[str, str, str, str]], str],
               session_over:bool, dialogue_state: dict) -> DSTOutput:
        
        dialogue_state = deepcopy(dialogue_state)
        dialogue_state["history"].append(["user", user_utterance])
        dialogue_state["user_action"] = user_action
        dialogue_state["terminated"] = session_over

        self.dst_core.from_cache(dialogue_state)
        dialogue_state = self.dst_core.update(user_act=user_action)
        
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
            dialogue_state=deepcopy(dialogue_state),
        )
        return dst_output
