import os
import torch
from convlab2.policy.mle.multiwoz import MLEPolicy

from system.data import PolicyOutput, VectorData
from system.module_base import PolicyBase
from utils.path import ROOT_DPATH

MLE_DIRECTORY = os.path.join(ROOT_DPATH, "ConvLab-2/convlab2/policy/mle")

class MyMLEPolicy(PolicyBase):
    module_name: str = "mle_policy"

    def __init__(self, device) -> None:
        self.policy_core = MLEPolicy()
        self.policy_core.policy.eval()
        self.device = device

    @property
    def dim_module_state(self) -> int:
        return self.policy_core.vector.da_dim

    @property
    def dim_module_output(self) -> int:
        return self.policy_core.vector.da_dim
    
    def init_session(self) -> None:
        self.policy_core.init_session()
    
    def predict(self, dialogue_state: dict) -> PolicyOutput:
        s_vec = torch.Tensor(self.policy_core.vector.state_vectorize(dialogue_state))

        with torch.no_grad():
            a = self.policy_core.policy.select_action(s_vec.to(device=self.device), False)
            a_logits = self.policy_core.policy.forward(s_vec.to(device=self.device))
        
        system_action = self.policy_core.vector.action_devectorize(a)
        a_probs = torch.sigmoid(a_logits).cpu()

        policy_output = PolicyOutput(
            module_name=self.module_name,
            system_action=system_action,
            module_state_vector=a_probs,
        )
        return policy_output

    def vectorize(self, module_output: PolicyOutput) -> VectorData:
        system_action = module_output.system_action
        da_vec = self.policy_core.vector.action_vectorize(system_action)
        vec_data = VectorData(
            module_name=self.module_name,
            vector=torch.Tensor(da_vec),
            data_to_restore=None
        )
        return vec_data
    
    def devectorize(self, vector_data: VectorData) -> PolicyOutput:
        da_vec = vector_data.vector
        system_action = self.policy_core.vector.action_devectorize(da_vec)
        policy_output = PolicyOutput(
            module_name=self.module_name,
            system_action=system_action,
        )
        return policy_output
