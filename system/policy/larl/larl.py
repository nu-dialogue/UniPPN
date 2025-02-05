import torch
from convlab2.policy.larl.multiwoz import LaRL

from system.data import WordPolicyOutput
from system.module_base import PolicyBase

class MyLaRLPolicy(PolicyBase):
    module_name: str = "larl"

    def __init__(self, device: torch.device) -> None:
        assert device.index == torch.cuda.current_device(), \
            (f"Specified device {device.index} is not the default "
             f"device {torch.cuda.current_device()}")
        self.policy_core = LaRL()

    @property
    def dim_module_state(self) -> int:
        return 0

    @property
    def dim_module_output(self) -> int:
        return 0

    def init_session(self) -> None:
        return self.policy_core.init_session()

    def predict(self, dialogue_state: dict) -> WordPolicyOutput:
        response, delex_response, active_domain = self.policy_core.predict(state=dialogue_state)

        policy_output = WordPolicyOutput(
            module_name=self.module_name,
            system_action=response,
            delexicalized_response=delex_response,
            active_domain=active_domain,
        )
        return policy_output
