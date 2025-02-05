from __future__ import annotations
import pickle
from copy import deepcopy
from logging import getLogger
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, TYPE_CHECKING

import pandas as pd
from tqdm import tqdm
from transformers import set_seed

from dialogue_simulation.evaluator import Evaluator
from dialogue_simulation.user_agent import UserAgent
from utils.log import set_logger
if TYPE_CHECKING:
    from system.data import SystemInternalHistory
    from system.system_agent import SystemAgent

logger = getLogger(__name__)
set_logger(logger)

@dataclass
class SessionLog:
    """
    Dataclass for storing the session logs
    """
    goal_seed: Optional[int]
    iteration_id: int
    process_id: int
    episode_id: int
    initial_goal: dict
    final_goal: dict
    task_complete: bool
    task_success: bool
    book_rate: float
    inform_f1: float
    inform_precision: float
    inform_recall: float
    goal_match_rate: float
    domain_eval: Dict[str, Dict[str, float]]
    user_da_f1: float
    system_da_f1: float
    num_turns: int
    num_training_turns: int
    training_turn_masks: List[bool]
    global_rewards: List[float]
    turn_evals_from_user: List[dict]
    turn_evals_from_system: List[dict]
    turn_evals_of_da_accuracy: List[dict]
    # turn_evals_from_reqt_goal: List[dict]
    user_turns: List[dict]
    system_turns: List[dict]
    # is_training_example: List[bool]
    system_internal_history: SystemInternalHistory

    def __post_init__(self):
        assert self.num_turns == len(self.turn_evals_from_user), \
            f"Mismatch: {self.num_turns} != {len(self.turn_evals_from_user)}"
        assert self.num_turns == len(self.turn_evals_from_system), \
            f"Mismatch: {self.num_turns} != {len(self.turn_evals_from_system)}"
        assert self.num_turns == len(self.turn_evals_of_da_accuracy), \
            f"Mismatch: {self.num_turns} != {len(self.turn_evals_of_da_accuracy)}"
        assert self.num_turns == len(self.user_turns), \
            f"Mismatch: {self.num_turns} != {len(self.user_turns)}"
        assert self.num_turns == len(self.system_turns), \
            f"Mismatch: {self.num_turns} != {len(self.system_turns)}"
        assert self.num_turns == self.system_internal_history.num_turns, \
            f"Mismatch: {self.num_turns} != {self.system_internal_history.num_turns}"
        
        assert self.num_turns -1 == len(self.global_rewards), \
            f"Mismatch: {self.num_turns -1} != {len(self.global_rewards)}"
    
    def to_dict(self) -> dict:
        d = deepcopy(self.__dict__)
        del d["system_internal_history"]
        return d
    
    def to_pickle(self, fpath) -> None:
        with open(fpath, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def from_pickle(fpath) -> "SessionLog":
        with open(fpath, "rb") as f:
            return pickle.load(f)

class Session:
    """
    Manage the interaction between the system and the simulator
    Reference to convlab2.dialog_agent.BiSession
    """
    def __init__(self, goal_seed: Optional[int], sys_agent: SystemAgent, user_agent: UserAgent,
                 evaluator: Evaluator, iteration_id: int, process_id: int, episode_id: int):
        self.goal_seed = goal_seed
        self.sys_agent = sys_agent
        self.user_agent = user_agent
        self.evaluator = evaluator
        self.iteration_id = iteration_id
        self.process_id = process_id
        self.episode_id = episode_id

    def init_session(self):
        self.sys_agent.init_session()
        self.user_agent.init_session()
        goal = self.user_agent.policy.get_goal()
        self.evaluator.add_goal(goal)

        self.final_goal = goal
        self.initial_goal = deepcopy(goal)

        self.training_turn_masks = []
        self.global_rewards = []

        self.turn_evals_from_user = []
        self.turn_evals_from_system = []

    def step(self, last_system_response) -> Tuple[str, bool, str, bool]:
        """
        Perform a single step of the dialogue
        """
        # 1. Response from user
        user_utterance = self.user_agent.response(last_system_response)

        # 2. Evaluate dialogue progress based on user's state
        self.evaluator.add_sys_da(self.user_agent.get_in_da())
        self.evaluator.add_usr_da(self.user_agent.get_out_da())
        turn_eval_from_user = self.evaluator.evaluate_from_user(self.user_agent)
        self.turn_evals_from_user.append(turn_eval_from_user)

        # 3. Get global reward
        if len(self.training_turn_masks) > 0: # Skip the first turn since there is no system response
            global_reward = self.evaluator.get_reward_v2(self.user_agent)
            self.global_rewards.append(global_reward)
        
        # 4. Response from system
        system_response, is_training_example = self.sys_agent.response(
            user_utterance=user_utterance,
            session_over=turn_eval_from_user["session_over"]
            # Final turn is processed by the system side for episode termination
        )
        
        turn_eval_from_system = self.evaluator.evaluate_from_system(self.sys_agent)
        self.turn_evals_from_system.append(turn_eval_from_system)

        is_training_turn = not turn_eval_from_user["session_over"] and is_training_example
        self.training_turn_masks.append(is_training_turn)
        
        return user_utterance, turn_eval_from_user["session_over"], system_response, is_training_turn

    def terminate(self) -> SessionLog:
        task_complete = self.user_agent.task_complete()
        prec, rec, f1 = self.evaluator.inform_F1()
        task_success = self.evaluator.task_success()
        book_rate = self.evaluator.book_rate()
        goal_match_rate = self.evaluator.final_goal_analyze()
        # is_training_example = self.sys_agent.is_training_example
        domain_eval = self.evaluator.domain_eval()

        turn_evals_of_da_accuracy = self.evaluator.evaluate_da_accuracy(user_agent=self.user_agent, sys_agent=self.sys_agent)
        user_da_f1, system_da_f1 = pd.json_normalize(
            turn_evals_of_da_accuracy
        ).fillna(value=float("nan")).mean(numeric_only=True)[["user_da.f1", "system_da.f1"]]

        # turn_evals_from_reqt_goal = self.evaluator.evaluate_from_reqt_goal(self.sys_agent)

        system_internal_history = self.sys_agent.get_internal_history()

        session_log = SessionLog(
            goal_seed=self.goal_seed,
            iteration_id=self.iteration_id,
            process_id=self.process_id,
            episode_id=self.episode_id,
            initial_goal=self.initial_goal,
            final_goal=self.final_goal,
            task_complete=task_complete,
            task_success=task_success,
            book_rate=book_rate,
            inform_f1=f1,
            inform_precision=prec,
            inform_recall=rec,
            goal_match_rate=goal_match_rate,
            domain_eval=domain_eval,
            user_da_f1=user_da_f1 if not pd.isna(user_da_f1) else None,
            system_da_f1=system_da_f1 if not pd.isna(system_da_f1) else None,
            num_turns=len(self.training_turn_masks),
            num_training_turns=sum(self.training_turn_masks),
            training_turn_masks=self.training_turn_masks,
            global_rewards=self.global_rewards,
            turn_evals_from_user=self.turn_evals_from_user,
            turn_evals_from_system=self.turn_evals_from_system,
            turn_evals_of_da_accuracy=turn_evals_of_da_accuracy,
            # turn_evals_from_reqt_goal=turn_evals_from_reqt_goal,
            user_turns=self.user_agent.get_turn_dicts(),
            system_turns=self.sys_agent.get_turn_dicts(),
            # is_training_example=is_training_example,
            system_internal_history=system_internal_history
        )

        return session_log

def compute_dialogue_task_stats(session_logs: List[SessionLog]):
    """
    Compute the task success rate, book rate, and inform F1
    """
    stats = {
        "task_success": [log_.task_success for log_ in session_logs if log_.task_success is not None],
        "book_rate": [log_.book_rate for log_ in session_logs if log_.book_rate is not None],
        "inform_f1": [log_.inform_f1 for log_ in session_logs if log_.inform_f1 is not None],
        "inform_precision": [log_.inform_precision for log_ in session_logs if log_.inform_precision is not None],
        "inform_recall": [log_.inform_recall for log_ in session_logs if log_.inform_recall is not None],
        "goal_match_rate": [log_.goal_match_rate for log_ in session_logs if log_.goal_match_rate is not None],
        **{
            f"{domain}/{key}": [
                log_.domain_eval[domain][key] for log_ in session_logs if log_.domain_eval[domain][key] is not None
            ] for domain in session_logs[0].domain_eval for key in session_logs[0].domain_eval[domain]
        },
        "user_da_f1": [log_.user_da_f1 for log_ in session_logs if log_.user_da_f1 is not None],
        "system_da_f1": [log_.system_da_f1 for log_ in session_logs if log_.system_da_f1 is not None],
        "num_turns": [log_.num_turns for log_ in session_logs],
    }
    mean_stats = {}
    for k, v in stats.items():
        if len(v) > 0:
            mean_stats[k] = sum(v) / len(v)
        else:
            mean_stats[k] = None
    return mean_stats

@dataclass
class SamplingResult:
    iteration_id: int
    process_id: int
    session_logs: List[SessionLog]
    num_sampled_dialogues: int
    num_sampled_turns: int
    num_sampled_training_turns: int
    dialogue_task_stats: Dict[str, float]

def sample_dialogues(
        iteration_id: int,
        process_id: int,
        sys_agent: SystemAgent,
        user_agent: UserAgent,
        max_turns_per_dialogue: int,
        turns_per_process: Optional[int] = None,
        goal_seeds: Optional[List[int]] = None
    ) -> SamplingResult:
    sampling_type = "dialogues" if goal_seeds is not None else "training turns"

    if sampling_type == "dialogues":
        assert turns_per_process is None, \
            "If goal_seeds is not None, turns_per_process must be None"
        pbar = tqdm(
            total=len(goal_seeds),
            desc=f"Iteration {iteration_id}, Process {process_id}, Sampled Dialogues",
            dynamic_ncols=True
        )
    else:
        assert turns_per_process is not None, \
            "If goal_seeds is None, turns_per_process must be specified"
        pbar = tqdm(
            total=turns_per_process,
            desc=f"Iteration {iteration_id}, Process {process_id}, Sampled Training Turns",
            dynamic_ncols=True
        )
        
    num_sampled_dialogues = 0 # Number of sampled dialogues
    num_sampled_turns = 0 # Number of sampled turns in total
    num_sampled_training_turns = 0 # Number of sampled turns to be used for training
    session_logs = []

    def is_sampling_finished() -> bool:
        if sampling_type == "dialogues":
            return num_sampled_dialogues >= len(goal_seeds)
        else:
            return num_sampled_training_turns >= turns_per_process

    while not is_sampling_finished():

        if goal_seeds is not None:
            goal_seed = goal_seeds[num_sampled_dialogues]
            set_seed(goal_seed)
        else:
            goal_seed = None

        evaluator = Evaluator(max_turn=max_turns_per_dialogue)
        session = Session(goal_seed=goal_seed,
                          sys_agent=sys_agent,
                          user_agent=user_agent,
                          evaluator=evaluator,
                          iteration_id=iteration_id,
                          process_id=process_id,
                          episode_id=num_sampled_dialogues)
        session.init_session()

        system_response = ""
        for t in range(max_turns_per_dialogue+1):
            _, session_over, system_response, is_training_turn = session.step(system_response)
            if session_over:
                break
            if sampling_type == "training turns" and is_training_turn:
                pbar.update(1)
        session_log = session.terminate()
        # Increment sampled data
        num_sampled_dialogues += 1
        num_sampled_turns += session_log.num_turns
        num_sampled_training_turns += session_log.num_training_turns
        # Record dialogue history and log
        session_logs.append(session_log)

        if sampling_type == "dialogues":
            pbar.update(1)
    pbar.close()

    result = SamplingResult(
        iteration_id=iteration_id,
        process_id=process_id,
        session_logs=session_logs,
        num_sampled_dialogues=num_sampled_dialogues,
        num_sampled_turns=num_sampled_turns,
        num_sampled_training_turns=num_sampled_training_turns,
        dialogue_task_stats=compute_dialogue_task_stats(session_logs)
    )

    return result
