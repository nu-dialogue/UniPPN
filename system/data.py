from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import pickle

import torch

@dataclass
class UserInput:
    user_utterance: str

@dataclass
class SystemOutput:
    system_response: Optional[str] = None

@dataclass
class ModuleOutputBase:
    module_name: Optional[str] = None
    module_state_vector: Optional[torch.Tensor] = None

    def to_dict(self, *args, **kwargs) -> Dict[str, Any]:
        d = deepcopy(self.__dict__)
        del d["module_state_vector"]
        return d
    
    def is_empty(self) -> bool:
        return all([v is None for v in self.__dict__.values()])

@dataclass
class NLUOutput(ModuleOutputBase):
    user_action: Optional[List[Tuple[str, str, str, str]]] = None

@dataclass
class DSTOutput(ModuleOutputBase):
    dialogue_state: Optional[dict] = None

    def to_dict(self, exclude_history: bool = True) -> Dict[str, Any]:
        d = super().to_dict()
        if exclude_history:
            try:
                del d["dialogue_state"]["history"]
            except KeyError:
                pass
        return d

@dataclass
class PolicyOutput(ModuleOutputBase):
    system_action: Optional[List[Tuple[str, str, str, str]]] = None

@dataclass
class WordPolicyOutput(ModuleOutputBase):
    system_action: Optional[str] = None
    delexicalized_response: Optional[str] = None
    active_domain: Optional[str] = None

@dataclass
class NLGOutput(ModuleOutputBase):
    system_response: Optional[str] = None

@dataclass
class PPNOutputBase:
    module_name: Optional[str] = None
    trajectory: Optional[Any] = None
    is_training_example: bool = True
    
    def to_dict(self, *args, **kwargs) -> Dict[str, Any]:
        # Exclude ppn_input_data and ppn_output_data
        d = deepcopy(self.__dict__)
        del d["trajectory"]
        return d
    
    @classmethod
    def from_module_output(cls, module_output: ModuleOutputBase, trajectory: Any) -> "PPNOutputBase":
        raise NotImplementedError

@dataclass
class PPNNLUOutput(PPNOutputBase):
    user_action: Optional[List[Tuple[str, str, str, str]]] = None

    @classmethod
    def from_module_output(cls, module_output: NLUOutput, trajectory: Any, is_training_example: bool) -> "PPNNLUOutput":
        return cls(
            module_name=module_output.module_name,
            user_action=module_output.user_action,
            trajectory=trajectory,
            is_training_example=is_training_example,
        )

@dataclass
class PPNDSTOutput(PPNOutputBase):
    dialogue_state: Optional[dict] = None

    def to_dict(self, exclude_history: bool = True) -> Dict[str, Any]:
        d = super().to_dict()
        dialogue_state = d["dialogue_state"]
        if exclude_history and dialogue_state is not None:
            if "history" in dialogue_state:
                del dialogue_state["history"]
        return d
    
    @classmethod
    def from_module_output(cls, module_output: DSTOutput, trajectory: Any, is_training_example: bool) -> "PPNDSTOutput":
        return cls(
            module_name=module_output.module_name,
            dialogue_state=module_output.dialogue_state,
            trajectory=trajectory,
            is_training_example=is_training_example,
        )

@dataclass
class PPNPolicyOutput(PPNOutputBase):
    system_action: Union[List[Tuple[str, str, str, str]], str, None] = None

    @classmethod
    def from_module_output(cls, module_output: Union[PolicyOutput, WordPolicyOutput], trajectory: Any, is_training_example: bool) -> "PPNPolicyOutput":
        return cls(
            module_name=module_output.module_name,
            system_action=module_output.system_action,
            trajectory=trajectory,
            is_training_example=is_training_example,
        )

@dataclass
class PPNNLGOutput(PPNOutputBase):
    system_response: Optional[str] = None

    @classmethod
    def from_module_output(cls, module_output: NLGOutput, trajectory: Any, is_training_example: bool) -> "PPNNLGOutput":
        return cls(
            module_name=module_output.module_name,
            system_response=module_output.system_response,
            trajectory=trajectory,
            is_training_example=is_training_example,
        )

@dataclass
class SystemInternalData:
    user_input: UserInput
    nlu_output: NLUOutput
    ppn_nlu_output: PPNNLUOutput
    dst_output: DSTOutput
    ppn_dst_output: PPNDSTOutput
    policy_output: Union[PolicyOutput, WordPolicyOutput]
    ppn_policy_output: PPNPolicyOutput
    nlg_output: NLGOutput
    ppn_nlg_output: PPNNLGOutput
    system_output: SystemOutput

@dataclass
class SystemInternalHistory:
    num_turns: int = 0

    user_input_history: List[UserInput] = field(default_factory=list)

    nlu_output_history: List[NLUOutput] = field(default_factory=list)
    ppn_nlu_output_history: List[PPNNLUOutput] = field(default_factory=list)

    dst_output_history: List[DSTOutput] = field(default_factory=list)
    ppn_dst_output_history: List[PPNDSTOutput] = field(default_factory=list)

    policy_output_history: List[Union[PolicyOutput, WordPolicyOutput]] = field(default_factory=list)
    ppn_policy_output_history: List[PPNPolicyOutput] = field(default_factory=list)

    nlg_output_history: List[NLGOutput] = field(default_factory=list)
    ppn_nlg_output_history: List[PPNNLGOutput] = field(default_factory=list)

    system_output_history: List[SystemOutput] = field(default_factory=list)

    def update(self, output: Union[UserInput, ModuleOutputBase, PPNOutputBase, SystemOutput]) -> None:
        output = deepcopy(output)
        if isinstance(output, UserInput):
            self.user_input_history.append(output)
        elif isinstance(output, NLUOutput):
            self.nlu_output_history.append(output)
        elif isinstance(output, PPNNLUOutput):
            self.ppn_nlu_output_history.append(output)
        elif isinstance(output, DSTOutput):
            self.dst_output_history.append(output)
        elif isinstance(output, PPNDSTOutput):
            self.ppn_dst_output_history.append(output)
        elif isinstance(output, (PolicyOutput, WordPolicyOutput)):
            self.policy_output_history.append(output)
        elif isinstance(output, PPNPolicyOutput):
            self.ppn_policy_output_history.append(output)
        elif isinstance(output, NLGOutput):
            self.nlg_output_history.append(output)
        elif isinstance(output, PPNNLGOutput):
            self.ppn_nlg_output_history.append(output)
        elif isinstance(output, SystemOutput):
            self.system_output_history.append(output)
        else:
            raise ValueError(f"Invalid output type: {type(output)}")

    def finalize_turn(self) -> bool:
        # Increment the number of turns
        self.num_turns += 1

        # Check if all histories have the same length
        assert len(self.user_input_history) == self.num_turns
        assert len(self.nlu_output_history) == self.num_turns
        assert len(self.ppn_nlu_output_history) == self.num_turns
        assert len(self.dst_output_history) == self.num_turns
        assert len(self.ppn_dst_output_history) == self.num_turns
        assert len(self.policy_output_history) == self.num_turns
        assert len(self.ppn_policy_output_history) == self.num_turns
        assert len(self.nlg_output_history) == self.num_turns
        assert len(self.ppn_nlg_output_history) == self.num_turns
        assert len(self.system_output_history) == self.num_turns

        # Check whether the last turn is training example
        is_training_example = all([
            self.ppn_nlu_output_history[-1].is_training_example,
            self.ppn_dst_output_history[-1].is_training_example,
            self.ppn_policy_output_history[-1].is_training_example,
            self.ppn_nlg_output_history[-1].is_training_example
        ])
        return is_training_example

    def get_last_turn(self) -> SystemInternalData:
        if self.user_input_history:
            user_input = self.user_input_history[-1]
        else:
            raise ValueError("User input history is empty, this means the session did not start yet")

        last_turn = SystemInternalData(
            user_input = user_input,
            
            nlu_output = self.nlu_output_history[-1] if self.nlu_output_history else NLUOutput(),
            ppn_nlu_output = self.ppn_nlu_output_history[-1] if self.ppn_nlu_output_history else PPNNLUOutput(),

            dst_output = self.dst_output_history[-1] if self.dst_output_history else DSTOutput(),
            ppn_dst_output = self.ppn_dst_output_history[-1] if self.ppn_dst_output_history else PPNDSTOutput(),

            policy_output = self.policy_output_history[-1] if self.policy_output_history else PolicyOutput(),
            ppn_policy_output = self.ppn_policy_output_history[-1] if self.ppn_policy_output_history else PPNPolicyOutput(),

            nlg_output = self.nlg_output_history[-1] if self.nlg_output_history else NLGOutput(),
            ppn_nlg_output = self.ppn_nlg_output_history[-1] if self.ppn_nlg_output_history else PPNNLGOutput(),

            system_output = self.system_output_history[-1] if self.system_output_history else SystemOutput()
        )

        return last_turn

    def get_user_action(self, turn_id: int) -> List[Tuple[str, str, str, str]]:
        if self.ppn_nlu_output_history[turn_id].user_action is not None:
            return self.ppn_nlu_output_history[turn_id].user_action
        else:
            return self.nlu_output_history[turn_id].user_action
    
    def get_dialogue_state(self, turn_id: int) -> dict:
        if self.ppn_dst_output_history[turn_id].dialogue_state is not None:
            return self.ppn_dst_output_history[turn_id].dialogue_state
        else:
            return self.dst_output_history[turn_id].dialogue_state
    
    def get_system_action(self, turn_id: int) -> List[Tuple[str, str, str, str]]:
        if self.ppn_policy_output_history[turn_id].system_action is not None:
            return self.ppn_policy_output_history[turn_id].system_action
        else:
            return self.policy_output_history[turn_id].system_action
    
    def get_system_response(self, turn_id: int) -> str:
        if self.ppn_nlg_output_history[turn_id].system_response is not None:
            return self.ppn_nlg_output_history[turn_id].system_response
        else:
            return self.nlg_output_history[turn_id].system_response

    def to_context_tuples(self) -> List[Tuple[str, str]]:
        context = []
        for turn_id in range(self.num_turns):
            context.append(["user", self.user_input_history[turn_id].user_utterance])
            context.append(["sys", self.system_output_history[turn_id].system_response])
        return context

    def to_turn_dicts(self) -> dict:
        turn_dicts = []
        for turn_id in range(self.num_turns):
            turn_dicts.append({
                "turn_id": turn_id,
                "user_utterance": self.user_input_history[turn_id].user_utterance,
                "nlu": self.nlu_output_history[turn_id].to_dict(),
                "ppn_nlu": self.ppn_nlu_output_history[turn_id].to_dict(),
                "dst": self.dst_output_history[turn_id].to_dict(),
                "ppn_dst": self.ppn_dst_output_history[turn_id].to_dict(),
                "policy": self.policy_output_history[turn_id].to_dict(),
                "ppn_policy": self.ppn_policy_output_history[turn_id].to_dict(),
                "nlg": self.nlg_output_history[turn_id].to_dict(),
                "ppn_nlg": self.ppn_nlg_output_history[turn_id].to_dict(),
                "system_response": self.system_output_history[turn_id].system_response,
            })
        return turn_dicts

@dataclass
class VectorData:
    module_name: str
    vector: torch.Tensor
    data_to_restore: Optional[Any]
