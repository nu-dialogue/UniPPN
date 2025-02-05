import os
import json
import numpy as np
import pandas as pd
from logging import getLogger
from typing import Optional, List
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from transformers import (
    HfArgumentParser,
    set_seed
)

from arguments import (
    GeneralArguments,
    DialogueSamplingArguments,
    save_args
)
from system import SystemAgent
from dialogue_simulation  import UserAgent, sample_dialogues
from utils import (
    set_logger,
    set_ddp_env,
    all_reduce_dict,
    TEST_GOAL_SEEDS
)

logger = getLogger(__name__)
set_logger(logger)

@dataclass
class RunningArguments:
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
            "help": "port number for distributed training"
        }
    )
    
    num_sample_turns: Optional[int] = field(
        default=None,
        metadata={
            "help": "number of turns to be sampled in total"
        }
    )
    do_test: bool = field(
        default=False,
        metadata={
            "help": (
                "whether to run as test. If True, goal seeds are set to: np.arange(1024)"
            )
        }
    )

    def __post_init__(self):
        if not self.do_test:
            assert self.num_sample_turns is not None, (
                "turns_per_process must be specified when do_test=False"
            )
        elif self.num_sample_turns is not None:
            logger.warning("num_sample_turns is ignored when do_test=True")

@dataclass
class UniPPNArguments:
    unippn_target_modules: List[str] = field(
        default_factory=list,
        metadata={
            "help": "list of target modules to attach UniPPN"
        }
    )
    unippn_model_dtype: str = field(
        default="float32",
        metadata={
            "choices": ["float32", "float16"],
            "help": "dtype of the UniPPN model"
        }
    )
    unippn_max_context_turns: int = field(
        default=3,
        metadata={
            "help": "maximum number of context turns to be used for UniPPN"
        }
    )
    unippn_max_prompt_tokens: int = field(
        default=512,
        metadata={
            "help": "maximum number of tokens to be used for the prompt"
        }
    )
    unippn_max_response_tokens: int = field(
        default=128,
        metadata={
            "help": "maximum number of tokens to be generated for the response"
        }
    )

    unippn_do_sample: bool = field(
        default=True,
        metadata={
            "help": "whether to sample from UniPPN"
        }
    )
    unippn_top_p: float = field(
        default=1.0,
        metadata={
            "help": "top-p value for sampling"
        }
    )
    unippn_policy_model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "model name for the policy model"
        }
    )
    unippn_value_model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "model name for the value model"
        }
    )


def main():
    parser = HfArgumentParser((
        GeneralArguments, RunningArguments, DialogueSamplingArguments,
        UniPPNArguments
    ))

    (
        general_args, run_args, ds_args,
        unippn_args
    ) = parser.parse_args_into_dataclasses()

    # Setup distributed parallel setting
    world_size, world_rank, local_rank = set_ddp_env(ddp_type=run_args.ddp_type, ddp_port=run_args.ddp_port)
    run_args.ddp = world_size != 1
    if run_args.ddp:
        logger.info(f"Using DistributedDataParallel: world_size: {world_size}, world_rank: {world_rank}, local_rank: {local_rank}")
        run_args.world_size = world_size
    else:
        logger.info("Not using DistributedDataParallel")

    # Initialize distributed training
    if run_args.ddp:
        dist.init_process_group(backend="nccl",
                                world_size=run_args.world_size,
                                rank=world_rank)
    
    # Make run directory
    os.makedirs(general_args.run_dpath, exist_ok=True)
    if world_rank == 0:
        # Save args
        args = {
            "general_args": general_args, "dialogue_sampling_args": ds_args, "running_args": run_args,
            "unippn_args": unippn_args
        }
        save_args(args, os.path.join(general_args.run_dpath, "args.json"))

    # Create directory to save sampled data
    session_logs_dpath = os.path.join(general_args.run_dpath, "session_logs")
    if world_rank == 0:
        if not os.path.exists(session_logs_dpath):
            os.makedirs(session_logs_dpath)
    
    if run_args.ddp:
        dist.barrier()

    # Build system and user agents
    sys_agent = SystemAgent(
        system_name=ds_args.system_name,
        device=torch.device("cuda", local_rank)
    )

    # Attach UniPPN
    if unippn_args.unippn_target_modules:
        logger.info(f"Attaching UniPPN to {unippn_args.unippn_target_modules}")
        sys_agent.attach_unippn(
            target_modules=unippn_args.unippn_target_modules,
            model_dtype=unippn_args.unippn_model_dtype,
            local_rank=local_rank,
            max_context_turns=unippn_args.unippn_max_context_turns,
            max_prompt_tokens=unippn_args.unippn_max_prompt_tokens,
            max_response_tokens=unippn_args.unippn_max_response_tokens,
            do_sample=unippn_args.unippn_do_sample,
            top_p=unippn_args.unippn_top_p,
            policy_model_name=unippn_args.unippn_policy_model_name,
            value_model_name=unippn_args.unippn_value_model_name,
        )
    
    user_agent = UserAgent(
        max_turn=ds_args.max_turns_per_dialogue,
        max_initiative=ds_args.user_max_initiative,
        device=torch.device("cuda", local_rank)
    )
    
    # 1. Sample dialogues
    # Set different seed on each process for different dialogue sampling
    set_seed(general_args.random_seed+world_rank)
    
    # Make goal seeds for each process
    if run_args.do_test:
        turns_per_process = None
        goal_seeds = np.reshape(TEST_GOAL_SEEDS, [world_size, -1])[world_rank].tolist()
        logger.info(f"(world_rank={world_rank}) Sampling dialogues with goal seeds: {goal_seeds}")
    else:
        turns_per_process = run_args.num_sample_turns // world_size
        goal_seeds = None
        logger.info(f"(world_rank={world_rank}) Sampling dialogues with {turns_per_process} turns per process")
    sample_result = sample_dialogues(
        iteration_id=0,
        process_id=world_rank,
        sys_agent=sys_agent,
        user_agent=user_agent,
        max_turns_per_dialogue=ds_args.max_turns_per_dialogue,
        turns_per_process=turns_per_process,
        goal_seeds=goal_seeds,
    )
    dialogue_task_stats = sample_result.dialogue_task_stats
    num_dialogues = sample_result.num_sampled_dialogues
    num_turns = sample_result.num_sampled_turns
    num_training_turns = sample_result.num_sampled_training_turns

    # 2. Aggregate sampled dialogues
    if run_args.ddp:
        dialogue_task_stats = all_reduce_dict(dialogue_task_stats, world_size)
        num_sampled = torch.Tensor([num_dialogues, num_turns, num_training_turns]).cuda()
        dist.all_reduce(num_sampled, op=dist.ReduceOp.SUM)
        num_dialogues, num_turns, num_training_turns = num_sampled.tolist()

    if world_rank == 0:
        logger.info("Aggregating sampled dialogues")
        eval_summary = pd.Series({
            **dialogue_task_stats,
            "num_sampled_dialogues": num_dialogues,
            "num_sampled_turns": num_turns,
            "num_training_turns": num_training_turns
        })
        logger.info(f"Evaluation Summary:\n{eval_summary}")
        json.dump(eval_summary.to_dict(), open(os.path.join(general_args.run_dpath, "eval_summary.json"), "w"), indent=4)

    logger.info("Saving sampled dialogues")
    for session_log in sample_result.session_logs:
        session_log_fpath = os.path.join(
            session_logs_dpath,
            f"{session_log.iteration_id}-{session_log.process_id}-{session_log.episode_id}.pkl"
        )
        session_log.to_pickle(session_log_fpath)


if __name__ == "__main__":
    # Run the main function
    main()
