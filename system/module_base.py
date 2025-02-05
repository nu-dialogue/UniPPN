from typing import Optional, List, Tuple, Union

from system.data import (
    ModuleOutputBase,
    NLUOutput,
    DSTOutput,
    PolicyOutput,
    WordPolicyOutput,
    NLGOutput,
    VectorData,
)

class ModuleBase:
    module_name: str

    @property
    def dim_module_state(self) -> int:
        raise NotImplementedError
    
    @property
    def dim_module_output(self) -> int:
        raise NotImplementedError

    def init_session(self) -> None:
        raise NotImplementedError
    
    def vectorize(self, module_output: ModuleOutputBase) -> VectorData:
        raise NotImplementedError

    def devectorize(self, vector_data: VectorData) -> ModuleOutputBase:
        raise NotImplementedError

class NLUBase(ModuleBase):
    def predict(
            self,
            user_utterance: str,
            context_tuples: List[Tuple[str, str]]
        ) -> NLUOutput:
        raise NotImplementedError

class DSTBase(ModuleBase):
    def init_session(self) -> dict:
        raise NotImplementedError
    
    def update(
            self,
            user_utterance: str,
            user_action: Optional[List[Tuple[str, str, str, str]]],
            session_over: bool,
            dialogue_state: dict,
        ) -> DSTOutput:
        raise NotImplementedError
    
    def update_response(
            self,
            system_action: Union[List[Tuple[str, str, str, str]], str],
            system_response: str,
            dialogue_state: dict,
        ) -> DSTOutput:
        raise NotImplementedError

class PolicyBase(ModuleBase):
    def predict(
            self,
            dialogue_state: Optional[dict],
        ) -> Union[PolicyOutput, WordPolicyOutput]:
        raise NotImplementedError

class NLGBase(ModuleBase):
    def generate(self, system_action: List[Tuple[str, str, str, str]]) -> NLGOutput:
        raise NotImplementedError
