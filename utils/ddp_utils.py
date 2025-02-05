import os
from typing import Tuple, Optional, List
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

def get_default_device() -> torch.device:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        local_rank = os.environ.get('LOCAL_RANK', 0)
        device = torch.device('cuda', int(local_rank))
    return device

def set_ddp_env(ddp_type: str, ddp_port: Optional[int]) -> Tuple[int, int, int]:
    if ddp_type == "default":
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        world_rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # if world_size > 1:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    
    elif ddp_type == "mpi":
        world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
        world_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))

        master_addr = os.environ.get("HOSTNAME", None)
        if master_addr is None:
            raise ValueError("HOSTNAME environment variable is not set")
        
        if ddp_port is not None:
            master_port = str(ddp_port)
        else:
            master_port = "29500"

        # os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(world_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

    else:
        raise ValueError(f"Unknown distributed environment type: {ddp_type}")

    torch.cuda.set_device(local_rank) # set default device

    return world_size, world_rank, local_rank

def all_reduce_dict(stats: dict, world_size: int) -> dict:
    for key, obj in stats.items():
        if obj is None:
            obj = 0
            num = 0
        else:
            num = 1
        if isinstance(obj, dict):
            stats[key] = all_reduce_dict(obj, world_size)
        elif not isinstance(obj, (torch.Tensor, int, float)):
            continue
        else:
            num_tensor = torch.tensor(num).cuda()
            obj_tensor = obj.cuda() if isinstance(obj, torch.Tensor) else torch.tensor(obj, dtype=torch.float).cuda()
            dist.all_reduce(num_tensor, op=ReduceOp.SUM)
            dist.all_reduce(obj_tensor, op=ReduceOp.SUM)
            if torch.is_nonzero(num_tensor):
                stats[key] = (obj_tensor / num_tensor).item()
            else:
                stats[key] = None
    return stats

def flatten_dict(d: dict, sep: str = "/") -> dict:
    flat_dict = {}
    for key, obj in d.items():
        if isinstance(obj, dict):
            for subkey, subobj in flatten_dict(obj, sep=sep).items():
                flat_dict[f"{key}{sep}{subkey}"] = subobj
        else:
            flat_dict[key] = obj
    return flat_dict
