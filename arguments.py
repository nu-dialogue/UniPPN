import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import torch

from system import SYSTEM_LIST

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # if not isinstance(obj, (int, float, str, list, dict, tuple, bool, type(None))):
        #     breakpoint()
        if isinstance(obj, torch.device):
            return {"type": "torch.device", "value": str(obj)}
        return super().default(obj)

def save_args(args_dict: Dict[str, Any], fpath) -> None:
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "w") as f:
        json.dump(
            {key: args.__dict__ for key, args in args_dict.items()},
            f, indent=4, cls=CustomJSONEncoder
        )

@dataclass
class GeneralArguments:
    run_dpath: str = field(
        metadata={
            "help": "Run directory path. Save all the outputs in this directory"
        }
    )
    random_seed: int = field(
        metadata={
            "help": "random seed"
        }
    )

    def __post_init__(self):
        self.run_id = os.path.basename(self.run_dpath)


@dataclass
class DialogueSamplingArguments:
    system_name: str = field(
        metadata={
            "choices": list(SYSTEM_LIST),
            "help": "name of the system agent"
        }
    )

    user_max_initiative: int = field(
        default=4,
        metadata={
            "help": "maximum number of slots user can mention in a turn"
        }
    )

    max_turns_per_dialogue: int = field(
        default=20,
        metadata={
            "help": "maximum number of timesteps per episode"
        }
    )

    save_session_logs: bool = field(
        default=True,
        metadata={
            "help": "whether to save session logs"
        }
    )



@dataclass
class CommonRLTrainingArguments:
    ddp_type: str = field(
        default="default",
        metadata={
            "choices": ["default", "mpi"],
            "help": ("distributed environment type. Use 'default' for torch.distributed.launch"
                     "and accelerate.launch. Use 'mpi' for OpenMPI")
        }
    )
    ddp_port: Optional[str] = field(
        default=None,
        metadata={
            "help": "port for distributed training"
        }
    )

    wandb_project_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "wandb project name"
        }
    )
    wandb_group_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "wandb group name"
        }
    )

    total_iterations: int = field(
        default=200,
        metadata={
            "help": "total number of training iterations"
        }
    )
    batch_size_per_device: int = field(
        default=512,
        metadata={
            "help": "number of turns to be sampled per iteration on each process"
        }
    )
    save_iterations: Optional[int] = field(
        default=None,
        metadata={
            "help": "save model every n iterations"
        }
    )
