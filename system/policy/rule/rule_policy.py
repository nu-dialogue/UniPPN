import torch
from convlab2.policy.vector.vector_multiwoz import MultiWozVector
from convlab2.policy.rule.multiwoz.rule_based_multiwoz_bot import RuleBasedMultiwozBot

from system.data import PolicyOutput, VectorData
from system.module_base import PolicyBase

class MyRulePolicy(PolicyBase):
    module_name: str = "rule_policy"

    def __init__(self) -> None:
        self.policy_core = RuleBasedMultiwozBot()
        self.mwoz_vector = MultiWozVector()

    @property
    def dim_module_state(self) -> int:
        return self.mwoz_vector.da_dim

    @property
    def dim_module_output(self) -> int:
        return self.mwoz_vector.da_dim
    
    def init_session(self) -> None:
        return self.policy_core.init_session()
    
    def predict(self, dialogue_state: dict) -> PolicyOutput:
        _ = self.mwoz_vector.state_vectorize(dialogue_state) # to update current domain
        
        system_action = self.policy_core.predict(
            state=dialogue_state
        )
        da_vec = self.mwoz_vector.action_vectorize(system_action)
        da_vec = torch.Tensor(da_vec)
        assert da_vec.shape[0] == self.dim_module_state, \
            f"Module state vector dimension mismatch: {len(da_vec)} != {self.dim_module_state}"
        
        policy_output = PolicyOutput(
            module_name=self.module_name,
            system_action=system_action,
            module_state_vector=da_vec,
        )
        return policy_output

    def vectorize(self, module_output: PolicyOutput) -> VectorData:
        system_action = module_output.system_action
        value_memory, domain_memory = dict(), set()
        for i,d,s,v in system_action:
            value_memory[f'{i}-{d}-{s}'] = v
            domain_memory.add(d)

        da_vec = self.mwoz_vector.action_vectorize(system_action)
        vec_data = VectorData(
            module_name=self.module_name,
            vector=torch.Tensor(da_vec),
            data_to_restore={
                "domain_memory": domain_memory,
                "value_memory": value_memory,
            }
        )
        return vec_data
    
    def devectorize(self, vector_data: VectorData) -> PolicyOutput:
        da_vec = vector_data.vector
        domain_memory = vector_data.data_to_restore["domain_memory"]
        value_memory = vector_data.data_to_restore["value_memory"]

        system_action = []
        dialog_acts = self.mwoz_vector.action_devectorize(da_vec)
        for i,d,s,v in dialog_acts:
            # if self.ignore_new_domain and d in domain_memory:
            #     continue
            da_key = f"{i}-{d}-{s}"
            if da_key in value_memory:
                v = value_memory[da_key]
            system_action.append([i,d,s,v])

        policy_output = PolicyOutput(
            module_name=self.module_name,
            system_action=system_action,
        )
        return policy_output