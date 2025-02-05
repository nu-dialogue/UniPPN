import os
import json
import random
from copy import deepcopy
from logging import getLogger
from dataclasses import dataclass
from typing import Optional, List, Tuple, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.distributed as dist

import wandb

from transformers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from accelerate import Accelerator


from utils import (
    get_default_device,
    set_logger,
)
from dialogue_simulation import SamplingResult
from system.ppn.uni_ppn import UniPPN

logger = getLogger(__name__)
set_logger(logger)

@dataclass
class BatchedRollout:
    prompt_tensors: List[torch.Tensor]
    response_tensors: List[torch.Tensor]
    logprobs_tensors: List[torch.Tensor]
    return_tensors: List[torch.Tensor]
    advantage_tensors: List[torch.Tensor]
    state_tensors: List[torch.Tensor]

class RolloutStorage:
    def __init__(self) -> None:
        self.module_names: List[str] = []
        self.prompt_texts: List[str] = []
        self.response_texts: List[str] = []
        self.prompt_lengths: List[int] = []
        self.response_lengths: List[int] = []
        self.prompt_tensors: List[torch.Tensor] = []
        self.response_tensors: List[torch.Tensor] = []
        self.logprobs_tensors: List[torch.Tensor] = []
        self.ref_logprobs_tensors: List[torch.Tensor] = []
        self.full_kl_tensors: List[torch.Tensor] = []
        self.kl_tensors: List[torch.Tensor] = []
        self.reward_tensors: List[torch.Tensor] = []
        self.kl_reward_tensors: List[torch.Tensor] = []
        self.state_tensors: List[torch.Tensor] = []
        self.value_tensors: List[torch.Tensor] = []
        self.return_tensors: List[torch.Tensor] = []
        self.advantage_tensors: List[torch.Tensor] = []

    def __getitem__(self, indices) -> BatchedRollout:
        return BatchedRollout(
            prompt_tensors=[self.prompt_tensors[i] for i in indices],
            response_tensors=[self.response_tensors[i] for i in indices],
            logprobs_tensors=[self.logprobs_tensors[i] for i in indices],
            return_tensors=[self.return_tensors[i] for i in indices],
            advantage_tensors=[self.advantage_tensors[i] for i in indices],
            state_tensors=[self.state_tensors[i] for i in indices],
        )

    def __add__(self, other: "RolloutStorage") -> "RolloutStorage":
        new_dict = {}
        for key in self.__dict__:
            new_dict[key] = getattr(self, key) + getattr(other, key)
        new_storage = RolloutStorage()
        new_storage.append(**new_dict)
        return new_storage

    def append(self, **kwargs) -> None:
        assert len(set([len(v) for v in kwargs.values()])) <= 1, \
            "Lengths of input lists must be the same"

        assert self.__dict__.keys() - kwargs.keys() == set(), \
            f"Some keys of the input dict are missing: {self.__dict__.keys() - kwargs.keys()}"
        assert kwargs.keys() - self.__dict__.keys() == set(), \
            f"Some keys of the input dict are redundant: {kwargs.keys() - self.__dict__.keys()}"

        for key, values in kwargs.items():
            self.__dict__[key].extend(values)

    def cutoff(self, length: int) -> None:
        for key, tensors in self.__dict__.items():
            self.__dict__[key] = tensors[:length]

    def to(self, device) -> None:
        for key, values in self.__dict__.items():
            if isinstance(values[0], torch.Tensor):
                self.__dict__[key] = [t.to(device) for t in values]

    def normalize_advantages(self) -> None:
        # Gather advantages across all processes
        tensors = torch.cat(self.advantage_tensors)
        device = get_default_device()
        assert tensors.device == device, \
            f"Tensors must be on the device {device}, but got {tensors.device}"
        gathered_tensors = [torch.zeros_like(tensors) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_tensors, tensors)
        all_advantages = torch.cat(gathered_tensors)

        # Normalize advantages
        mean = all_advantages.mean()
        std = all_advantages.std()
        for i in range(len(self.advantage_tensors)):
            self.advantage_tensors[i] = (self.advantage_tensors[i] - mean) / (std + 1e-8)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[bool] = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()

def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor, gather: bool = True) -> torch.Tensor:
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)

    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy

def compute_return_and_advantage(
        rewards: List[torch.Tensor],
        values: List[torch.Tensor],
        gamma: float,
        lam: float,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Compute return and advantage from rewards and values
    Args:
        rewards: rewards of each turn, list of Size([1]) tensors
        values: values of each turn, list of Size([1]) tensors
        gamma: discount factor
        lam: GAE parameter
    Returns:
        returns: list of Size([1]) tensors
        advantages: list of Size([1]) tensors
    """
    rewards = torch.cat(rewards).to("cpu")
    values = torch.cat(values).to("cpu")
    num_turns = rewards.shape[0]

    advantages = torch.zeros(num_turns)
    gae = 0
    for t in reversed(range(num_turns)):
        nextvalue = values[t+1] if t+1 < num_turns else 0
        delta = rewards[t] + gamma*nextvalue - values[t]
        gae = delta + gamma*lam*gae
        advantages[t] = gae
    returns = advantages + values
    return list(returns.view(-1,1)), list(advantages.view(-1,1))

class PPOTrainerForUniPPN:
    def __init__(
        self,
        target_modules: List[str],
        ppn: UniPPN,
        mdp_level: Literal["module", "turn"],
        total_iterations: int,
        num_epochs: int,
        batch_size_per_device: int,
        mini_batch_size_per_device: int,
        poliyc_train_only_embeddings: bool,
        policy_learning_rate: float,
        policy_scheduler: str,
        value_learning_rate: float,
        gamma: float,
        lam: float,
        gradient_accumulation_steps: int,
        policy_warmup_steps: int,
        policy_max_grad_norm: float,
        policy_clip_range: float,
        policy_kl_penalty_coef: float,
        policy_use_full_kl_penalty: bool,
        trainer_state_path: Optional[str] = None,
    ):
        self.target_modules = target_modules
        self.ppn = ppn
        self.mdp_level = mdp_level

        # Update gradient_accumulation_steps
        if len(self.target_modules) > 1:
            logger.info((
                f"We are training multiple modules, so gradient_accumulation_steps ({gradient_accumulation_steps}) "
                f"will be multiplied by the number of target modules ({len(self.target_modules)}) to keep the total number of "
                 "gradient updates consistent across modules."
            ))
            gradient_accumulation_steps *= len(self.target_modules)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        self.policy_model = ppn.policy_model
        self.policy_tokenizer = ppn.policy_tokenizer
        self.value_model = ppn.value_model
        self.value_tokenizer = ppn.value_tokenizer

        if poliyc_train_only_embeddings:
            ## Frozen the entire model
            for param in self.policy_model.parameters():
                param.requires_grad = False
            # Unfrozen the embeddings
            for param in self.policy_model.get_input_embeddings().parameters():
                param.requires_grad = True
            for param in self.policy_model.get_output_embeddings().parameters():
                param.requires_grad = True
    
        # Count trainable parameters
        total_params, trainable_params = 0, 0
        for param in self.policy_model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        logger.info(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")


        self.policy_optimizer = Adam(
            filter(lambda p: p.requires_grad, self.policy_model.parameters()),
            lr=policy_learning_rate,
        )
        if policy_scheduler == "constant":
            self.policy_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.policy_optimizer,
                num_warmup_steps=policy_warmup_steps*self.accelerator.num_processes,
            )
        elif policy_scheduler == "linear":
            self.policy_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.policy_optimizer,
                num_warmup_steps=policy_warmup_steps*self.accelerator.num_processes,
                num_training_steps=total_iterations*self.accelerator.num_processes,
            )
        else:
            raise ValueError(f"Unsupported policy_scheduler: {policy_scheduler}")

        self.value_optimizer = Adam(
            filter(lambda p: p.requires_grad, self.value_model.parameters()),
            lr=value_learning_rate,
        )
        self.value_scheduler = None

        if trainer_state_path:
            logger.info(f"Loading trainer state from {trainer_state_path}")
            self.load_state(trainer_state_path)

        (
            self.policy_model,
            self.policy_optimizer,
            self.policy_scheduler,
            self.value_model,
            self.value_optimizer,
            self.value_scheduler,
        ) = self.accelerator.prepare(
            self.policy_model,
            self.policy_optimizer,
            self.policy_scheduler,
            self.value_model,
            self.value_optimizer,
            self.value_scheduler,
        )

        self.num_epochs = num_epochs
        self.batch_size_per_device = batch_size_per_device
        self.mini_batch_size_per_device = mini_batch_size_per_device
        self.gamma = gamma
        self.lam = lam

        self.policy_max_grad_norm = policy_max_grad_norm
        self.policy_clip_range = policy_clip_range

        self.kl_penalty_coef = policy_kl_penalty_coef
        self.use_full_kl_penalty = policy_use_full_kl_penalty

        self.current_step = 0

    def save_state(self, output_path: str) -> None:
        # Save optimizer and scheduler states
        self.accelerator.save(
            self.policy_optimizer.state_dict(),
            os.path.join(output_path, "policy", "optimizer.pt")
        )
        self.accelerator.save(
            self.policy_scheduler.state_dict(),
            os.path.join(output_path, "policy", "scheduler.pt")
        )

        self.accelerator.save(
            self.value_optimizer.state_dict(),
            os.path.join(output_path, "value", "optimizer.pt")
        )

        # Save rng state
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if self.accelerator.num_processes > 1:
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
        else:
            rng_state["cuda"] = torch.cuda.get_rng_state()

        if self.accelerator.num_processes > 1:
            self.accelerator.save(
                rng_state, os.path.join(output_path, f"rng_state_{self.accelerator.process_index}.pth")
            )
        else:
            self.accelerator.save(rng_state, os.path.join(output_path, "rng_state.pth"))

    def load_state(self, checkpoint_path: str) -> None:
        # Load optimizer and scheduler states
        self.policy_optimizer.load_state_dict(
            torch.load(
                os.path.join(checkpoint_path, "policy", "optimizer.pt"),
                map_location=self.accelerator.device,
            )
        )
        self.policy_scheduler.load_state_dict(
            torch.load(os.path.join(checkpoint_path, "policy", "scheduler.pt"))
        )

        self.value_optimizer.load_state_dict(
            torch.load(
                os.path.join(checkpoint_path, "value", "optimizer.pt"),
                map_location=self.accelerator.device,)
        )

        # Load rng states
        if self.accelerator.num_processes > 1:
            rng_state = self.accelerator.load(
                os.path.join(checkpoint_path, f"rng_state_{self.accelerator.process_index}.pth")
            )
        else:
            rng_state = self.accelerator.load(os.path.join(checkpoint_path, "rng_state.pth"))
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.random.set_rng_state(rng_state["cpu"])
        if self.accelerator.num_processes > 1:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        else:
            torch.cuda.set_rng_state(rng_state["cuda"])

    def _make_module_level_mdp_rollout(self, sample_result: SamplingResult):
        reward_unit = 0.1
        reward_mapping = {
            -1: {tm: -reward_unit for tm in self.target_modules},
             5: {tm: -reward_unit for tm in self.target_modules},
            40: {tm: -reward_unit for tm in self.target_modules},
        }
        # full reward for the last module at the last turn
        reward_mapping[40][self.target_modules[-1]] = reward_unit * 20 * len(self.target_modules)

        rollout = RolloutStorage()

        # 1 Preprocess sample result
        for session_log in sample_result.session_logs:
            episode_trajectory = {
                "module_names": [],
                "prompt_texts": [],
                "response_texts": [],
                "prompt_lengths": [],
                "response_lengths": [],
                "prompt_tensors": [],
                "response_tensors": [],
                "logprobs_tensors": [],
                "ref_logprobs_tensors": [],
                "full_kl_tensors": [],
                "kl_tensors": [],
                "reward_tensors": [],
                "kl_reward_tensors": [],
                "state_tensors": [],
                "value_tensors": [],
            }

            for turn_id, is_training_turn in enumerate(session_log.training_turn_masks):
                # Skip non-training turns
                if not is_training_turn:
                    continue
                
                for target_module in self.target_modules:
                    ppn_output = getattr(
                        session_log.system_internal_history,
                        f"ppn_{target_module}_output_history",
                    )[turn_id]
                    assert ppn_output.is_training_example, \
                        "Non-training example is included in training data"

                    episode_trajectory["module_names"].append(target_module)
                    episode_trajectory["prompt_texts"].append(ppn_output.trajectory["prompt_text"])
                    episode_trajectory["response_texts"].append(ppn_output.trajectory["response_text"])
                    episode_trajectory["prompt_lengths"].append(ppn_output.trajectory["prompt_ids"].shape[0])
                    episode_trajectory["response_lengths"].append(ppn_output.trajectory["response_ids"].shape[0])

                    episode_trajectory["prompt_tensors"].append(ppn_output.trajectory["prompt_ids"])
                    episode_trajectory["response_tensors"].append(ppn_output.trajectory["response_ids"])
                    episode_trajectory["logprobs_tensors"].append(ppn_output.trajectory["logprobs"])
                    episode_trajectory["ref_logprobs_tensors"].append(ppn_output.trajectory["ref_logprobs"])

                    resp_len = ppn_output.trajectory["response_ids"].shape[0]
                    full_logprobs_resp = ppn_output.trajectory["full_logprobs"][-resp_len:]
                    ref_full_logprobs_resp = ppn_output.trajectory["ref_full_logprobs"][-resp_len:]
                    logprobs_resp = ppn_output.trajectory["logprobs"][-resp_len:]
                    ref_logprobs_resp = ppn_output.trajectory["ref_logprobs"][-resp_len:]

                    # Compute KL divergence
                    ## 1. Full KL divergence over all vocabulary
                    full_kl = F.kl_div(
                        input=ref_full_logprobs_resp, target=full_logprobs_resp, log_target=True, reduction="none"
                    ).sum(-1).mean().unsqueeze(0)
                    ## 2. KL divergence over the response
                    kl = (logprobs_resp - ref_logprobs_resp).mean().unsqueeze(0)

                    episode_trajectory["full_kl_tensors"].append(full_kl)
                    episode_trajectory["kl_tensors"].append(kl)

                    # Get reward
                    reward = torch.Tensor([
                        reward_mapping[session_log.global_rewards[turn_id]][target_module]
                    ])
                    episode_trajectory["reward_tensors"].append(reward)

                    kl_penalty = full_kl if self.use_full_kl_penalty else kl
                    episode_trajectory["kl_reward_tensors"].append(
                        reward - self.kl_penalty_coef * kl_penalty
                    )

                    episode_trajectory["state_tensors"].append(ppn_output.trajectory["state_ids"])
                    episode_trajectory["value_tensors"].append(ppn_output.trajectory["value"])

            # Compute advantages and returns
            return_tensors, advantage_tensors = compute_return_and_advantage(
                rewards=episode_trajectory["kl_reward_tensors"],
                values=episode_trajectory["value_tensors"],
                gamma=self.gamma,
                lam=self.lam,
            )
            episode_trajectory.update({
                "return_tensors": return_tensors,
                "advantage_tensors": advantage_tensors,
            })

            rollout.append(**episode_trajectory)

        rollout.cutoff(length=self.batch_size_per_device*len(self.target_modules))
        return rollout

    def _make_turn_level_mdp_rollout(self, sample_result: SamplingResult):
        reward_unit = 0.1
        reward_mapping = {-1: -reward_unit, 5: -reward_unit, 40: reward_unit * 20}

        module_rollouts = []

        # 1 Preprocess sample result
        for target_module in self.target_modules:
            rollout = RolloutStorage()
            for session_log in sample_result.session_logs:
                episode_trajectory = {
                    "module_names": [],
                    "prompt_texts": [],
                    "response_texts": [],
                    "prompt_lengths": [],
                    "response_lengths": [],
                    "prompt_tensors": [],
                    "response_tensors": [],
                    "logprobs_tensors": [],
                    "ref_logprobs_tensors": [],
                    "full_kl_tensors": [],
                    "kl_tensors": [],
                    "reward_tensors": [],
                    "kl_reward_tensors": [],
                    "state_tensors": [],
                    "value_tensors": [],
                }

                for turn_id, is_training_turn in enumerate(session_log.training_turn_masks):
                    # Skip non-training turns
                    if not is_training_turn:
                        continue

                    ppn_output = getattr(
                        session_log.system_internal_history,
                        f"ppn_{target_module}_output_history",
                    )[turn_id]
                    assert ppn_output.is_training_example, \
                        "Non-training example is included in training data"

                    episode_trajectory["module_names"].append(target_module)
                    episode_trajectory["prompt_texts"].append(ppn_output.trajectory["prompt_text"])
                    episode_trajectory["response_texts"].append(ppn_output.trajectory["response_text"])
                    episode_trajectory["prompt_lengths"].append(ppn_output.trajectory["prompt_ids"].shape[0])
                    episode_trajectory["response_lengths"].append(ppn_output.trajectory["response_ids"].shape[0])

                    episode_trajectory["prompt_tensors"].append(ppn_output.trajectory["prompt_ids"])
                    episode_trajectory["response_tensors"].append(ppn_output.trajectory["response_ids"])
                    episode_trajectory["logprobs_tensors"].append(ppn_output.trajectory["logprobs"])
                    episode_trajectory["ref_logprobs_tensors"].append(ppn_output.trajectory["ref_logprobs"])

                    resp_len = ppn_output.trajectory["response_ids"].shape[0]
                    full_logprobs_resp = ppn_output.trajectory["full_logprobs"][-resp_len:]
                    ref_full_logprobs_resp = ppn_output.trajectory["ref_full_logprobs"][-resp_len:]
                    logprobs_resp = ppn_output.trajectory["logprobs"][-resp_len:]
                    ref_logprobs_resp = ppn_output.trajectory["ref_logprobs"][-resp_len:]

                    # Compute KL divergence
                    ## 1. Full KL divergence over all vocabulary
                    full_kl = F.kl_div(
                        input=ref_full_logprobs_resp, target=full_logprobs_resp, log_target=True, reduction="none"
                    ).sum(-1).mean().unsqueeze(0)
                    ## 2. KL divergence over the response
                    kl = (logprobs_resp - ref_logprobs_resp).mean().unsqueeze(0)

                    episode_trajectory["full_kl_tensors"].append(full_kl)
                    episode_trajectory["kl_tensors"].append(kl)

                    # Get reward
                    reward = torch.Tensor([
                        reward_mapping[session_log.global_rewards[turn_id]]
                    ])
                    episode_trajectory["reward_tensors"].append(reward)

                    kl_penalty = full_kl if self.use_full_kl_penalty else kl
                    episode_trajectory["kl_reward_tensors"].append(
                        reward - self.kl_penalty_coef * kl_penalty
                    )

                    # episode_trajectory["state_tensors"].append(ppn_output.trajectory["state_ids"])
                    # episode_trajectory["value_tensors"].append(ppn_output.trajectory["value"])
                    # We use the same state and value for all target modules
                    episode_trajectory["state_tensors"].append(
                        getattr(
                            session_log.system_internal_history,
                            f"ppn_{self.target_modules[0]}_output_history",
                        )[turn_id].trajectory["state_ids"]
                    )
                    episode_trajectory["value_tensors"].append(
                        getattr(
                            session_log.system_internal_history,
                            f"ppn_{self.target_modules[0]}_output_history",
                        )[turn_id].trajectory["value"]
                    )

                # Compute advantages and returns
                return_tensors, advantage_tensors = compute_return_and_advantage(
                    rewards=episode_trajectory["kl_reward_tensors"],
                    values=episode_trajectory["value_tensors"],
                    gamma=self.gamma,
                    lam=self.lam,
                )
                episode_trajectory.update({
                    "return_tensors": return_tensors,
                    "advantage_tensors": advantage_tensors,
                })

                rollout.append(**episode_trajectory)

            rollout.cutoff(length=self.batch_size_per_device)
            module_rollouts.append(rollout)

        rollout = sum(module_rollouts, RolloutStorage())

        return rollout

    def step(self, sample_result: SamplingResult):
        # 1 Preprocess sample result
        if self.mdp_level == "module":
            make_rollout_fn = self._make_module_level_mdp_rollout
        elif self.mdp_level == "turn":
            make_rollout_fn = self._make_turn_level_mdp_rollout
        else:
            raise ValueError(f"Unsupported MDP level: {self.mdp_level}")

        rollout = make_rollout_fn(sample_result)
        rollout.to(self.accelerator.device)
        rollout.normalize_advantages()

        # 2 Training loop
        loss_stats = {
            "value_model": [],
            "policy_model": [],
        }
        for epoch_id in range(self.num_epochs):

            # 2.1 Train value model
            minibatch_indices = np.random.permutation(
                self.batch_size_per_device * len(self.target_modules)
            ).reshape(-1, self.mini_batch_size_per_device)
            # minibatch_indices: Size([num_minibatches, mini_batch_size_per_device])
            for indices in minibatch_indices:
                minibatch = rollout[indices]
                with self.accelerator.accumulate(self.value_model):
                    value_loss = self.compute_value_loss(minibatch)
                    self.accelerator.backward(value_loss)
                    self.value_optimizer.step()
                    self.value_optimizer.zero_grad()
                    loss_stats["value_model"].append(value_loss.item())

            # 2.2 Train policy model
            minibatch_indices = np.random.permutation(
                self.batch_size_per_device * len(self.target_modules)
            ).reshape(-1, self.mini_batch_size_per_device)
            # minibatch_indices: Size([num_minibatches, mini_batch_size_per_device])
            for indices in minibatch_indices:
                minibatch = rollout[indices]
                with self.accelerator.accumulate(self.policy_model):
                    policy_loss = self.compute_policy_loss(minibatch)
                    self.accelerator.backward(policy_loss)
                    if self.policy_max_grad_norm and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.policy_model.parameters(), self.policy_max_grad_norm)
                    self.policy_optimizer.step()
                    self.policy_optimizer.zero_grad()
                    loss_stats["policy_model"].append(policy_loss.item())

        # 3 Log stats
        rollout.to("cpu")
        ## 3.1 Get raw stats
        stats_df = pd.DataFrame({
            "module_name": rollout.module_names,
            "prompt_length": [torch.tensor([l], dtype=torch.float) for l in rollout.prompt_lengths],
            "response_length": [torch.tensor([l], dtype=torch.float) for l in rollout.response_lengths],
            "kl": rollout.kl_tensors,
            "full_kl": rollout.full_kl_tensors,
            "reward": rollout.reward_tensors,
            "kl_reward": rollout.kl_reward_tensors,
            "value": rollout.value_tensors,
            "return": rollout.return_tensors,
            "advantage": rollout.advantage_tensors,
        })
        stats_dict = stats_df.groupby("module_name").agg(lambda x: x.tolist()).T.unstack().to_dict()
        raw_stats = {
            f"{mod_name}/{stat_key}": torch.cat(tensors).to(self.accelerator.device)
            for (mod_name, stat_key), tensors in stats_dict.items()
        }
        raw_stats.update({
            f"loss/{k}": torch.tensor(v, device=self.accelerator.device) for k, v in loss_stats.items()
        })

        # 3.2 Gather raw stats
        if self.accelerator.num_processes > 1:
            raw_stats = {k: self.accelerator.gather(v) for k, v in raw_stats.items()}

        # 3.3 Aggregate stats
        stats_ = {
            "loss/value_model": raw_stats.pop("loss/value_model").mean().item(),
            "loss/policy_model": raw_stats.pop("loss/policy_model").mean().item(),
            "learning_rate/value_model": self.value_optimizer.param_groups[0]["lr"],
            "learning_rate/policy_model": self.policy_optimizer.param_groups[0]["lr"],
        }
        for key, tensor in raw_stats.items():
            stats_[key] = tensor.detach().cpu()
            stats_[f"{key}_mean"] = tensor.mean().item()

        # 3.4 Make tables
        table_df = pd.DataFrame({
            "module_name": rollout.module_names,
            "Prompt": rollout.prompt_texts,
            "Response": rollout.response_texts,
            "Reward": [round(r.item(), 4) for r in rollout.reward_tensors],
            "KL": [round(k.item(), 4) for k in rollout.kl_tensors],
            "Full KL": [round(k.item(), 4) for k in rollout.full_kl_tensors],
            "Value": [round(v.item(), 4) for v in rollout.value_tensors],
            "Return": [round(r.item(), 4) for r in rollout.return_tensors],
            "Advantage": [round(a.item(), 4) for a in rollout.advantage_tensors],
        })
        for target_module, df_ in table_df.groupby("module_name"):
            df_ = df_.drop(columns="module_name")
            stats_[f"generations/{target_module}"] = wandb.Table(dataframe=df_)

        self.policy_scheduler.step()

        self.current_step += 1
        return stats_

    def compute_value_loss(self, minibatch: BatchedRollout):
        self.value_tokenizer.padding_side = "right"

        input_tensors = [{"input_ids": s} for s in minibatch.state_tensors]
        model_inputs = self.value_tokenizer.pad(
            input_tensors,
            padding="longest",
            return_tensors="pt",
        ).to(self.accelerator.device)

        model_inputs = {
            "input_ids": self.accelerator.pad_across_processes(
                tensor=model_inputs.input_ids, dim=1,
                pad_index=self.value_tokenizer.pad_token_id, pad_first=False,
            ),
            "attention_mask": self.accelerator.pad_across_processes(
                tensor=model_inputs.attention_mask, dim=1,
                pad_index=0, pad_first=False,
            ),
        }
        values = self.value_model(**model_inputs).view(-1)
        returns = torch.cat(minibatch.return_tensors)
        value_loss = F.mse_loss(values, returns)

        return value_loss

    def compute_policy_loss(self, minibatch: BatchedRollout):
        self.policy_tokenizer.padding_side = "right"

        input_tensors = [
            {"input_ids": torch.cat([p, r])} for p, r in zip(
                minibatch.prompt_tensors, minibatch.response_tensors
            )
        ]
        model_inputs = self.policy_tokenizer.pad(
            input_tensors,
            padding="longest",
            return_tensors="pt",
        ).to(self.accelerator.device)

        model_inputs = {
            "input_ids": self.accelerator.pad_across_processes(
                tensor=model_inputs.input_ids, dim=1,
                pad_index=self.policy_tokenizer.pad_token_id, pad_first=False,
            ),
            "attention_mask": self.accelerator.pad_across_processes(
                tensor=model_inputs.attention_mask, dim=1,
                pad_index=0, pad_first=False,
            ),
        }

        # Get logprobs for entire input sequence
        logits = self.policy_model(**model_inputs).logits
        logprobs = logprobs_from_logits(logits[:, :-1, :], model_inputs["input_ids"][:, 1:])

        # Pad old logprobs
        old_logprobs = torch.zeros_like(logprobs, device=self.accelerator.device)
        for i, old_lps in enumerate(minibatch.logprobs_tensors):
            old_logprobs[i, :old_lps.shape[0]] = old_lps

        ratio = torch.exp(logprobs - old_logprobs)

        # Advantages
        advantages = torch.cat(minibatch.advantage_tensors).unsqueeze(-1)

        # Compute policy loss
        losses_1 = -advantages * ratio
        losses_2 = -advantages * torch.clamp(ratio, 1.0-self.policy_clip_range, 1.0+self.policy_clip_range)

        # Compute masks
        masks_attnmask = model_inputs["attention_mask"][:, 1:]
        masks_prompt = torch.ones_like(masks_attnmask)
        for i, prompt_tensor in enumerate(minibatch.prompt_tensors):
            masks_prompt[i, :prompt_tensor[1:].shape[0]] = 0
        masks = masks_attnmask * masks_prompt

        loss = masked_mean(torch.max(losses_1, losses_2), masks)

        return loss
