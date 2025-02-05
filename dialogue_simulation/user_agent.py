import torch
from typing import Union
from logging import getLogger
from typing import List
import requests
import json
import os

from convlab2.dialog_agent import PipelineAgent
from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.policy.rule.multiwoz.policy_agenda_multiwoz import UserPolicyAgendaMultiWoz
from convlab2.nlg.template.multiwoz import TemplateNLG

from utils.ddp_utils import get_default_device
from utils.log import set_logger

logger = getLogger(__name__)
set_logger(logger)

class UserAgent(PipelineAgent):
    def __init__(self, max_turn, max_initiative, device: Union[torch.device, str] = get_default_device()):
        self.name = 'user'
        self.opponent_name = 'sys'

        self.nlu = BERTNLU(
            mode="sys",
            config_file="multiwoz_sys_context.json",
            model_file="https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/bert_multiwoz_sys_context.zip",
            device=device
        )
        self.dst = None
        self.policy = UserPolicyAgendaMultiWoz(max_turn=max_turn, max_initiative=max_initiative)
        self.nlg = TemplateNLG(is_user=True, mode="manual")

        self.turn_count = 0
        self.history = []
        self.log = []

    def init_session(self, ini_goal=None):
        self.turn_count = 0
        self.history = []
        self.log = []

        self.nlu.init_session()
        self.policy.init_session(ini_goal=ini_goal)
        self.nlg.init_session()

    def response(self, system_response):

        user_utterance = super().response(system_response)
        system_action = self.input_action.copy()
        user_action = self.output_action.copy()

        self.log.append({
            "turn_id": self.turn_count,
            "system_response": system_response,
            "nlu": {"system_action": system_action},
            "policy": {"user_action": user_action},
            "nlg": {"user_utterance": user_utterance},
            "user_utterance": user_utterance,
        })
        self.turn_count += 1
        return user_utterance
    
    def task_complete(self):
        return self.policy.goal.task_complete()

    def get_turn_dicts(self):
        return self.log
