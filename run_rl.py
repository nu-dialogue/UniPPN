import os
import json
import random
import warnings
import numpy as np
import pandas as pd
from logging import getLogger
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field, asdict

import wandb
def maybe_compress_history(obj: Any) -> Tuple[Any, bool]:
    """
    Automatically cast to histogram if size is 32 or more (i.e., `>=32`)
    while more than 32 (i.e., `>32`) is cast to histogram in the original implementation.
    """
    # 32 or smaller np obj has been converted to list by `wandb.util.json_friendly()`,
    # so we need to convert it back to np array
    if isinstance(obj, list):
        obj = np.array(obj)

    if np and isinstance(obj, np.ndarray) and obj.size >= 32:
        return wandb.Histogram(obj, num_bins=32).to_json(), True
    else:
        return obj, False
wandb.util.maybe_compress_history = maybe_compress_history

from tqdm import tqdm
import torch
import torch.distributed as dist
from transformers import (
    HfArgumentParser,
    set_seed,
)

from arguments import (
    GeneralArguments,
    DialogueSamplingArguments,
    CommonRLTrainingArguments,
    save_args
)
from system import (
    SystemAgent,
    PPOTrainerForUniPPN,
)
from dialogue_simulation  import (
    UserAgent,
    SessionLog,
    SamplingResult,
    sample_dialogues,
    compute_dialogue_task_stats,
)
from utils import (
    set_logger,
    set_ddp_env,
    flatten_dict,
)

logger = getLogger(__name__)
set_logger(logger)

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

    unippo_target_modules_to_train: List[str] = field(
        default_factory=list,
        metadata={
            "help": "list of target modules to train with UniPPN"
        }
    )
    unippo_mdp_level: str = field(
        default="module",
        metadata={
            "choices": ["module", "turn"],
            "help": "MDP level for UniPPN"
        }
    )
    unippo_num_epochs: int = field(
        default=4,
        metadata={
            "help": "number of epochs for PPO training"
        }
    )
    unippo_minibatch_size_per_device: int = field(
        default=8,
        metadata={
            "help": "mini-batch size per device for PPO training"
        }
    )
    unippo_policy_train_only_embeddings: bool = field(
        default=False,
        metadata={
            "help": "train only embeddings of the policy model"
        }
    )
    unippo_gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "gradient accumulation steps for PPO training"
        }
    )
    unippo_policy_learning_rate: float = field(
        default=5e-6,
        metadata={
            "help": "learning rate for the policy model"
        }
    )
    unippo_policy_scheduler: str = field(
        default="constant",
        metadata={
            "choices": ["constant", "linear"],
            "help": "scheduler for the policy learning rate"
        }
    )
    unippo_value_learning_rate: float = field(
        default=5e-5,
        metadata={
            "help": "learning rate for the value model"
        }
    )
    unippo_gamma: float = field(
        default=0.99,
        metadata={
            "help": "discount factor for PPO"
        }
    )
    unippo_lam: float = field(
        default=0.95,
        metadata={
            "help": "lambda for GAE"
        }
    )
    unippo_policy_warmup_steps: int = field(
        default=5,
        metadata={
            "help": "warmup steps for the policy model"
        }
    )
    unippo_policy_max_grad_norm: float = field(
        default=1.0,
        metadata={
            "help": "max grad norm for the policy model"
        }
    )
    unippo_policy_clip_range: float = field(
        default=0.2,
        metadata={
            "help": "clip range for PPO training"
        }
    )
    unippo_policy_kl_penalty_coef: float = field(
        default=0.1,
        metadata={
            "help": "coefficient for KL penalty"
        }
    )
    unippo_policy_use_full_kl_penalty: bool = field(
        default=True,
        metadata={
            "help": "use full KL penalty"
        }
    )

def main():
    parser = HfArgumentParser((
        GeneralArguments, DialogueSamplingArguments, CommonRLTrainingArguments, UniPPNArguments
    ))
    general_args, ds_args, train_args, unippn_args = parser.parse_args_into_dataclasses()

    # Setup distributed parallel setting
    world_size, world_rank, local_rank = set_ddp_env(ddp_type=train_args.ddp_type, ddp_port=train_args.ddp_port)
    train_args.ddp = world_size != 1
    train_args.world_size = world_size
    if train_args.ddp:
        logger.info(f"Using DistributedDataParallel: world_size: {world_size}, world_rank: {world_rank}, local_rank: {local_rank}")
    else:
        logger.info("Not using DistributedDataParallel")
    
    # Initialize distributed training
    if train_args.ddp:
        dist.init_process_group(backend="nccl",
                                world_size=train_args.world_size,
                                rank=world_rank)
    
    # Make run directory
    os.makedirs(general_args.run_dpath, exist_ok=True)
    if world_rank == 0:
        # Save args
        all_args = {
            "general_args": general_args, "dialogue_sampling_args": ds_args, "train_args": train_args,
            "unippn_args": unippn_args
        }
        save_args(all_args, os.path.join(general_args.run_dpath, "args.json"))

    # Create directory to save sampled data
    session_logs_dpath = os.path.join(general_args.run_dpath, "session_logs")
    if world_rank == 0 and ds_args.save_session_logs:
        if not os.path.exists(session_logs_dpath):
            os.makedirs(session_logs_dpath)
    
    if train_args.ddp:
        dist.barrier()
    
    # Build system and user agents
    sys_agent = SystemAgent(
        system_name=ds_args.system_name,
        device=torch.device("cuda", local_rank)
    )

    # Attach UniPPN and build trainer
    if not unippn_args.unippn_target_modules:
        logger.info("No target modules specified. Skipping UniPPN training.")
        return

    logger.info(f"Attaching UniPPN to {unippn_args.unippn_target_modules}")
    unippn = sys_agent.attach_unippn(
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
    ppo_trainer = PPOTrainerForUniPPN(
        target_modules=unippn_args.unippo_target_modules_to_train,
        ppn=unippn,
        mdp_level=unippn_args.unippo_mdp_level,
        total_iterations=train_args.total_iterations,
        num_epochs=unippn_args.unippo_num_epochs,
        batch_size_per_device=train_args.batch_size_per_device,
        mini_batch_size_per_device=unippn_args.unippo_minibatch_size_per_device,
        poliyc_train_only_embeddings=unippn_args.unippo_policy_train_only_embeddings,
        policy_learning_rate=unippn_args.unippo_policy_learning_rate,
        policy_scheduler=unippn_args.unippo_policy_scheduler,
        value_learning_rate=unippn_args.unippo_value_learning_rate,
        gamma=unippn_args.unippo_gamma,
        lam=unippn_args.unippo_lam,
        gradient_accumulation_steps=unippn_args.unippo_gradient_accumulation_steps,
        policy_warmup_steps=unippn_args.unippo_policy_warmup_steps,
        policy_max_grad_norm=unippn_args.unippo_policy_max_grad_norm,
        policy_clip_range=unippn_args.unippo_policy_clip_range,
        policy_kl_penalty_coef=unippn_args.unippo_policy_kl_penalty_coef,
        policy_use_full_kl_penalty=unippn_args.unippo_policy_use_full_kl_penalty,
    )

    user_agent = UserAgent(
        max_turn=ds_args.max_turns_per_dialogue,
        max_initiative=ds_args.user_max_initiative,
        device=torch.device("cuda", local_rank)
    )

    # PPO training

    # Set different seed on each process for different dialogue sampling
    set_seed(general_args.random_seed+world_rank)

    # Log hyperparameters
    if world_rank == 0:
        run = wandb.init(
            dir=general_args.run_dpath,
            project=train_args.wandb_project_name,
            group=train_args.wandb_group_name,
            name=general_args.run_id,
            config={key: asdict(args) for key, args in all_args.items()},
        )

        hyperparams = pd.Series({
            "Target Modules to PP": unippn_args.unippn_target_modules,
            "Target Modules to Train": unippn_args.unippo_target_modules_to_train,
            "# Total Iterations": train_args.total_iterations,
            "# Processes": train_args.world_size,
            "Batch Size (per process)": train_args.batch_size_per_device,
            "Batch Size (in total)": train_args.world_size*train_args.batch_size_per_device,
            "Mini-batch Size (per process)": unippn_args.unippo_minibatch_size_per_device,
            "Gradient Accumulation Steps": unippn_args.unippo_gradient_accumulation_steps,
            "Mini-batch Size (in total)": train_args.world_size*unippn_args.unippo_minibatch_size_per_device*unippn_args.unippo_gradient_accumulation_steps,
        })
        logger.info(f"\n*** PPO hyperparameters ***\n{hyperparams.to_string()}\n")
    
    if train_args.ddp:
        dist.barrier()

    # Start PPO training
    logger.info("*** Start PPO training ***")
    iterator = tqdm(
        range(train_args.total_iterations), desc="PPO Training", ncols=100, disable=world_rank!=0
    )
    for iteration_id in iterator:
        # 1. Sample dialogues
        sample_result = sample_dialogues(
            iteration_id=iteration_id,
            process_id=world_rank,
            sys_agent=sys_agent,
            user_agent=user_agent,
            max_turns_per_dialogue=ds_args.max_turns_per_dialogue,
            turns_per_process=train_args.batch_size_per_device,
        )
        # 2. Save session logs if necessary
        if ds_args.save_session_logs:
            for session_log in sample_result.session_logs:
                session_log_fpath = os.path.join(
                    session_logs_dpath,
                    f"{session_log.iteration_id}-{session_log.process_id}-{session_log.episode_id}.pkl"
                )
                session_log.to_pickle(session_log_fpath)
        
        # 3. Compute and gather dialogue task stats
        dialogue_task_stats = compute_dialogue_task_stats(sample_result.session_logs)
        if train_args.ddp:
            dt_stats_list = [None] * world_size
            dist.all_gather_object(dt_stats_list, dialogue_task_stats)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                dialogue_task_stats = {
                    key: np.nanmean(np.array(
                        [dt_stats[key] for dt_stats in dt_stats_list],
                        dtype=float
                    )) for key in dialogue_task_stats
                }

        # 4. PPO update
        train_stats = ppo_trainer.step(sample_result)

        # 5. Log stats
        if world_rank == 0:
            stats = {
                "dialogue_task": dialogue_task_stats,
                "ppn_training": train_stats,
            }
            stats = flatten_dict(stats)
            run.log(stats)

        # 6. Save model
        if train_args.save_iterations is not None and iteration_id > 0 and iteration_id % train_args.save_iterations == 0:
            if world_rank == 0:
                logger.info(f"Saving model at iteration {iteration_id}")
                checkpoint_path = os.path.join(general_args.run_dpath, f"ppn/checkpoint-{iteration_id:03d}")
                unippn.save(checkpoint_path)
                ppo_trainer.save_state(checkpoint_path)

    if world_rank == 0:
        logger.info("PPO training finished")
        run.finish()
        checkpoint_path = os.path.join(general_args.run_dpath, f"ppn/checkpoint-{iteration_id:03d}")
        unippn.save(checkpoint_path)
        ppo_trainer.save_state(checkpoint_path)


if __name__ == "__main__":
    main()
