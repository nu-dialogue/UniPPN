from copy import deepcopy
from typing import Optional, List, Union, Tuple, Any

import torch

from system.data import (
    UserInput,
    SystemOutput,

    NLUOutput,
    DSTOutput,
    PolicyOutput,
    NLGOutput,

    PPNNLUOutput,
    PPNDSTOutput,
    PPNPolicyOutput,
    PPNNLGOutput,

    SystemInternalHistory,
)
from system.module_base import (
    ModuleBase,
    NLUBase,
    DSTBase,
    PolicyBase,
    NLGBase,
)
from system.ppn.uni_ppn import UniPPN

SYSTEM_LIST = {
    "sys_rul": {
        "nlu": "bert_nlu",
        "dst": "rule_dst",
        "policy": "rule_policy",
        "nlg": "template_nlg",
    },
    "sys_trade": {
        "nlu": None,
        "dst": "trade",
        "policy": "rule_policy",
        "nlg": "template_nlg",
    },
    "sys_d3st": {
        "nlu": None,
        "dst": "d3st",
        "policy": "rule_policy",
        "nlg": "template_nlg",
    },
    "sys_mle": {
        "nlu": "bert_nlu",
        "dst": "rule_dst",
        "policy": "mle_policy",
        "nlg": "template_nlg",
    },
    "sys_ppo": {
        "nlu": "bert_nlu",
        "dst": "rule_dst",
        "policy": "ppo_policy",
        "nlg": "template_nlg",
    },
    "sys_larl": {
        "nlu": "bert_nlu",
        "dst": "rule_dst",
        "policy": "larl",
        "nlg": None,
    },
    "sys_lava": {
        "nlu": "bert_nlu",
        "dst": "rule_dst",
        "policy": "lava",
        "nlg": None,
    },
    "sys_scl": {
        "nlu": "bert_nlu",
        "dst": "rule_dst",
        "policy": "rule_policy",
        "nlg": "sclstm_nlg",
    },
    "sys_scg": {
        "nlu": "bert_nlu",
        "dst": "rule_dst",
        "policy": "rule_policy",
        "nlg": "scgpt_nlg",
    },
    "sys_gpt4om": {
        "nlu": None,
        "dst": "gpt4om",
        "policy": "gpt4om",
        "nlg": None,
    },
    "sys_pptod": {
        "nlu": None,
        "dst": "pptod",
        "policy": "pptod",
        "nlg": None,
    }
}

def modules_factory(
        system_name: str, device: torch.device
    ) -> Tuple[Optional[NLUBase], Optional[DSTBase], Optional[PolicyBase], Optional[NLGBase]]:
    nlu, dst, policy, nlg = None, None, None, None

    module_combination = SYSTEM_LIST[system_name]
    # E2E
    if all([
        module_combination["nlu"] is None,
        module_combination["dst"] is not None,
        module_combination["policy"] is not None,
        module_combination["nlg"] is None,
    ]):
        if module_combination["policy"] == "gpt4om":
            from system.e2e.llmicl.llmicl import build_e2e_as_modules
            dst, policy = build_e2e_as_modules(module_name="gpt4om", model_name="gpt-4o-mini", device=device)
        elif module_combination["policy"] == "pptod":
            from system.e2e.pptod.pptod import build_e2e_as_modules
            dst, policy = build_e2e_as_modules(device=device)
        else:
            raise ValueError(f"Invalid policy module: {module_combination['policy']}")
    
    else:
        # NLU
        if module_combination["nlu"] == "bert_nlu":
            from system.nlu.bert.bert_nlu import MyBERTNLU
            nlu = MyBERTNLU(device=device)
        elif module_combination["nlu"] is not None:
            raise ValueError(f"Invalid NLU module: {module_combination['nlu']}")

        # DST
        if module_combination["dst"] == "rule_dst":
            from system.dst.rule.rule_dst import MyRuleDST
            dst = MyRuleDST()
        elif module_combination["dst"] == "trade":
            from system.dst.trade.trade import MyTRADEDST
            dst = MyTRADEDST(device=device)
        elif module_combination["dst"] == "d3st":
            from system.dst.d3st.d3st import D3ST
            dst = D3ST(device=device)
        elif module_combination["dst"] is not None:
            raise ValueError(f"Invalid DST module: {module_combination['dst']}")

        # Policy
        if module_combination["policy"] == "rule_policy":
            from system.policy.rule.rule_policy import MyRulePolicy
            policy = MyRulePolicy()
        elif module_combination["policy"] == "mle_policy":
            from system.policy.mle.mle_policy import MyMLEPolicy
            policy = MyMLEPolicy(device=device)
        elif module_combination["policy"] == "ppo_policy":
            from system.policy.ppo.ppo_policy import MyPPOPolicy
            policy = MyPPOPolicy(device=device)
        elif module_combination["policy"] == "larl":
            from system.policy.larl.larl import MyLaRLPolicy
            policy = MyLaRLPolicy(device=device)
        elif module_combination["policy"] == "lava":
            from system.policy.lava.lava import LAVA
            policy = LAVA(device=device)
        else:
            raise ValueError(f"Invalid policy module: {module_combination['policy']}")

        # NLG
        if module_combination["nlg"] == "template_nlg":
            from system.nlg.template.template_nlg import MyTemplateNLG
            nlg = MyTemplateNLG()
        elif module_combination["nlg"] == "sclstm_nlg":
            from system.nlg.sclstm.sclstm_nlg import MySCLSTM
            nlg = MySCLSTM(device=device)
        elif module_combination["nlg"] == "scgpt_nlg":
            from system.nlg.scgpt.scgpt import SCGPT
            nlg = SCGPT(device=device)
        elif module_combination["nlg"] is not None:
            raise ValueError(f"Invalid NLG module: {module_combination['nlg']}")

    return nlu, dst, policy, nlg

class SystemAgent:
    def __init__(
        self,
        system_name: str,
        device: torch.device,
    ) -> None:
        # Load modules
        (
            self.nlu,
            self.dst,
            self.policy,
            self.nlg,
        ) = modules_factory(system_name=system_name, device=device)

        # PPN placeholders
        self.ppn_nlu: Union[UniPPN, None] = None
        self.ppn_dst: Union[UniPPN, None] = None
        self.ppn_policy: Union[UniPPN, None] = None
        self.ppn_nlg: Union[UniPPN,  None] = None

        self.device = device

    def attach_unippn(
        self,
        target_modules: List[str],
        model_dtype: str,
        local_rank: int,
        max_context_turns: int,
        max_prompt_tokens: int,
        max_response_tokens: int,
        do_sample: bool,
        top_p: float,
        policy_model_name: str,
        value_model_name: str,
        ref_policy_model_name: Optional[str] = None,
    ) -> UniPPN:
        unippn = UniPPN(
            policy_model_name=policy_model_name,
            value_model_name=value_model_name,
            ref_policy_model_name=ref_policy_model_name,
            model_dtype=model_dtype,
            local_rank=local_rank,
            max_context_turns=max_context_turns,
            max_prompt_tokens=max_prompt_tokens,
            max_response_tokens=max_response_tokens,
            do_sample=do_sample,
            top_p=top_p,
        )
        for target_module in target_modules:
            setattr(self, f"ppn_{target_module}", unippn)
        return unippn

    def init_session(self) -> None:
        if self.nlu is not None:
            self.nlu.init_session()

        if self.dst is not None:
            init_dialogue_state = self.dst.init_session()
        else:
            init_dialogue_state = None
        
        self.policy.init_session()
        
        if self.nlg is not None:
            self.nlg.init_session()

        self.turn_id = 0
        self.last_dialogue_state = init_dialogue_state
        self.internal_history = SystemInternalHistory()

    def response(self, user_utterance: str, session_over: bool) -> Tuple[str, bool]:
        # 0 Set user input
        self.internal_history.update(UserInput(user_utterance=user_utterance))

        # 1 NLU
        user_action = None
        ## 1.1 Original NLU
        if self.nlu is not None:
            nlu_output = self.nlu.predict(
                user_utterance=user_utterance,
                context_tuples=self.internal_history.to_context_tuples(),
            )
            user_action = nlu_output.user_action
        else:
            nlu_output = NLUOutput()
        self.internal_history.update(nlu_output)
        
        ## 1.2 PPN_NLU
        if user_action is not None and self.ppn_nlu is not None:
            ppn_nlu_output = self.ppn_nlu.postprocess(
                module_output=nlu_output,
                system_internal_history=self.internal_history,
            )
            user_action = ppn_nlu_output.user_action
        else:
            ppn_nlu_output = PPNNLUOutput()
        self.internal_history.update(ppn_nlu_output)


        # 2 DST (on user input)
        dialogue_state = None
        ## 2.1 Original DST
        if self.dst is not None:
            dst_output = self.dst.update(
                user_utterance=user_utterance,
                user_action=user_action,
                session_over=session_over,
                dialogue_state=self.last_dialogue_state,
            )
            dialogue_state = dst_output.dialogue_state
        else:
            dst_output = DSTOutput()
        self.internal_history.update(dst_output)

        ## 2.2 PPN_DST
        if dialogue_state is not None and self.ppn_dst is not None:
            ppn_dst_output = self.ppn_dst.postprocess(
                module_output=dst_output,
                system_internal_history=self.internal_history,
            )
            dialogue_state = ppn_dst_output.dialogue_state
        else:
            ppn_dst_output = PPNDSTOutput()
        self.internal_history.update(ppn_dst_output)


        # 3 Policy
        ## 3.1 Original Policy
        policy_output = self.policy.predict(
            dialogue_state=dialogue_state,
        )
        system_action = policy_output.system_action
        self.internal_history.update(policy_output)

        ## 3.2 PPN_Policy
        if self.ppn_policy is not None:
            ppn_policy_output = self.ppn_policy.postprocess(
                module_output=policy_output,
                system_internal_history=self.internal_history,
            )
            system_action = ppn_policy_output.system_action
        else:
            ppn_policy_output = PPNPolicyOutput()
        self.internal_history.update(ppn_policy_output)


        # 4 NLG
        system_response = None
        ## 4.1 Original NLG
        if self.nlg is not None:
            nlg_output = self.nlg.generate(
                system_action=system_action,
            )
            system_response = nlg_output.system_response
        else:
            nlg_output = NLGOutput()
        self.internal_history.update(nlg_output)
        
        ## 4.2 PPN_NLG
        if system_response is not None and self.ppn_nlg is not None:
            ppn_nlg_output = self.ppn_nlg.postprocess(
                module_output=nlg_output,
                system_internal_history=self.internal_history,
            )
            system_response = ppn_nlg_output.system_response
        else:
            ppn_nlg_output = PPNNLGOutput()
        self.internal_history.update(ppn_nlg_output)

        ## 4.3 Finalize system response
        if system_response is None:
            system_response = system_action


        # 5 DST (on system response)
        if self.dst is not None:
            self.last_dialogue_state = self.dst.update_response(
                system_action=system_action,
                system_response=system_response,
                dialogue_state=dialogue_state,
            ).dialogue_state

        # 6 Finalize
        self.internal_history.update(SystemOutput(system_response=system_response))
        is_training_example = self.internal_history.finalize_turn()
        self.turn_id += 1

        return system_response, is_training_example

    def get_dialogue_state(self) -> dict:
        return deepcopy(self.last_dialogue_state)

    def get_turn_dicts(self) -> dict:
        return self.internal_history.to_turn_dicts()

    def get_internal_history(self) -> SystemInternalHistory:
        return self.internal_history
