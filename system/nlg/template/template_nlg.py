from typing import List, Tuple
from convlab2.nlg.template.multiwoz import TemplateNLG

from system.data import NLGOutput, VectorData
from system.module_base import NLGBase

class MyTemplateNLG(NLGBase):
    module_name: str = "template_nlg"

    def __init__(self, mode: str = "manual") -> None:
        self.nlg_core = TemplateNLG(
            is_user=False,
            mode=mode,
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
        system_response = self.nlg_core.generate(system_action)

        nlg_output = NLGOutput(
            module_name=self.module_name,
            system_response=system_response,
        )

        return nlg_output

    def vectorize(self, module_output: NLGOutput) -> VectorData:
        raise NotImplementedError
    
    def devectorize(self, vector_data: VectorData) -> NLGOutput:
        raise NotImplementedError
    