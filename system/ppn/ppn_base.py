from system.data import (
    ModuleOutputBase,
    PPNOutputBase,
    SystemInternalHistory,
)

class PPNBase:
    def save(self, output_path: str) -> None:
        """
        Save model to output_path
        Args:
            output_path: path to directory to save config and model
        """
        raise NotImplementedError
    
    def postprocess(self, module_output: ModuleOutputBase, system_internal_history: SystemInternalHistory) -> PPNOutputBase:
        """
        Postprocess module output
        Args:
            module_output: output from the module
            system_internal_history: system internal history
        Returns:
            postprocessed module output
        """
        raise NotImplementedError
    
class PPOTrainerBase:
    ...