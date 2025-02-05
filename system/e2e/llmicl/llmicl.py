import os
import re
from copy import deepcopy
from logging import getLogger
from typing import List, Tuple, Dict, Optional, Union, List

import torch

from openai import (
    OpenAI,
    RateLimitError,
    NotGiven,
)
from openai.types.chat.completion_create_params import ResponseFormat

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential, # for exponential backoff
    # retry_if_not_exception_type,
    retry_if_exception_type,
)
import httpx

from convlab2.util.multiwoz.state import default_state
from convlab2.util.multiwoz.dbquery import Database

from system.dst.dst import DSTBaseforPPN
from system.module_base import PolicyBase
from system.data import (
    DSTOutput,
    WordPolicyOutput,
    VectorData,
)
from utils import set_logger
from system.e2e.llmicl.prompts import (
    process_active_domain_str,
    domain_belief_state_diff_jsonstr2dict,
    lexicalize_response,
    PromptFormatter,
)
from system.e2e.llmicl.fewshot_exampler import FewshotExampler

logger = getLogger(__name__)
set_logger(logger)

openai_client = OpenAI(timeout=httpx.Timeout(timeout=60), max_retries=1)
@retry(
    retry=retry_if_exception_type(RateLimitError), # for rate limit
    wait=wait_random_exponential(min=1, max=20),
    stop=stop_after_attempt(3),
    after=lambda x: print(f"Retrying: {x}"),
)
def chat_completion_with_retry(**kwargs):
    return openai_client.chat.completions.create(**kwargs)

def call_openai_api(
        model_name: str, messages: List[Dict[str, str]],
        response_format: Union[ResponseFormat, NotGiven], max_tokens: int
    ) -> str:
    try:
        response = chat_completion_with_retry(
            model=model_name,
            messages=messages,
            response_format=response_format,
            max_tokens=max_tokens,
            temperature=0,
            # request_timeout=60, # this may don't work
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"{e}")
        return f"Error: {e}"

abs_dir = os.path.dirname(os.path.abspath(__file__))

def remove_return_char(text: str) -> str:
    return re.sub(r"\n+", " ", text)

class LLMICL:
    """
    Large Language Model with In-Context Learning
    """

    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.fewshot_num_examples = 3
        self.fewshot_sampling_top_k = 5
        self.max_context_turns = 3

        self.prompt_formatter = PromptFormatter()
        self.example_retriever = FewshotExampler.load(
            saved_path=os.path.join(abs_dir, "vector_db/mwoz23-gte_base-ctx3-dpd20.pt"),
            device=device
        )

        self.db = Database()

    # DST implementation
    def dst_init_session(self):
        init_dialogue_state = default_state()
        init_dialogue_state["history"].append(["sys", "null"])
        self.cache = {
            "active_domain": None,
            "fewshot_examples": None,
        }
        return init_dialogue_state

    def dst_update(
            self,
            user_utterance: str,
            user_action: Optional[List[Tuple[str, str, str, str]]],
            session_over: bool,
            dialogue_state: dict
        ) -> DSTOutput:
        assert user_action is None, \
            "LLMICL does not support user action"

        dialogue_state = deepcopy(dialogue_state)
        dialogue_state["history"].append(["user", user_utterance])
        dialogue_state["user_action"] = user_utterance
        dialogue_state["terminated"] = session_over

        # 0. Retrieve few-shot examples
        examples = self.example_retriever.retrieve(
            context=dialogue_state["history"],
            num_examples=self.fewshot_num_examples,
            sampling_top_k=self.fewshot_sampling_top_k
        )

        # 1. Detect the current active domain
        ad_prompt_messages, ad_response_format = self.prompt_formatter.active_domain_tracking(
            context=dialogue_state["history"],
            max_context_turns=self.max_context_turns
        )
        active_domain_str = call_openai_api(
            model_name=self.model_name,
            messages=ad_prompt_messages,
            response_format=ad_response_format,
            max_tokens=128,
        )
        active_domain = process_active_domain_str(active_domain_str)

        # 2. Predict belief state
        if active_domain not in [None, "police"]:
            bs_prompt_messages, bs_response_format = self.prompt_formatter.belief_state_tracking(
                context=dialogue_state["history"],
                active_domain=active_domain,
                fewshot_examples=examples,
                max_context_turns=self.max_context_turns
            )
            active_bs_diff_str = call_openai_api(
                model_name=self.model_name,
                messages=bs_prompt_messages,
                response_format=bs_response_format,
                max_tokens=128,
            )
            active_bs_diff = domain_belief_state_diff_jsonstr2dict(
                domain_belief_state_diff_str=active_bs_diff_str,
                domain=active_domain
            )
            for constraint_type, constraints in active_bs_diff.items():
                for slot, value in constraints.items():
                    dialogue_state["belief_state"][active_domain][constraint_type][slot] = value

        self.cache = {
            "active_domain": active_domain,
            "fewshot_examples": examples,
        }

        return DSTOutput(
            module_name=None,
            dialogue_state=deepcopy(dialogue_state),
        )

    def dst_update_response(
            self,
            system_action: Union[List[Tuple[str, str, str, str]], str],
            system_response: str,
            dialogue_state: dict
        ) -> DSTOutput:
        dialogue_state = deepcopy(dialogue_state)
        dialogue_state["history"].append(["sys", system_response])
        dialogue_state["system_action"] = system_action

        return DSTOutput(
            module_name=None,
            dialogue_state=deepcopy(dialogue_state),
        )


    # Policy implementation
    def init_session_policy(self):
        pass

    def policy_predict(
            self,
            dialogue_state: dict,
        ) -> WordPolicyOutput:

        # Ensure that cache is saved in the dialogue state tracking
        active_domain = self.cache["active_domain"]
        fewshot_examples = self.cache["fewshot_examples"]

        if active_domain is None:
            response = "Is there anything else I can help you with today?"
            return WordPolicyOutput(
                module_name=None,
                system_action=response,
                delexicalized_response=response,
                active_domain=None
            )

        context = dialogue_state["history"]
        belief_state = dialogue_state["belief_state"]
        entities = {}
        for domain, domain_bs in belief_state.items():
            entities[domain.capitalize()] = self.db.query(
                domain=domain,
                constraints= domain_bs["semi"].items()
            )
        
        # Generate system response
        messages, response_format = self.prompt_formatter.response_generation(
            context=context,
            belief_state=belief_state,
            db_results={d: len(e) for d, e in entities.items()},
            active_domain=active_domain,
            fewshot_examples=fewshot_examples,
            max_context_turns=self.max_context_turns
        )
        delex_sys_response = call_openai_api(
            model_name=self.model_name,
            messages=messages,
            response_format=response_format,
            max_tokens=128,
        )
        delex_sys_response = remove_return_char(delex_sys_response)
        sys_response = lexicalize_response(
            delexicalized_response=delex_sys_response,
            belief_state=belief_state,
            entities=entities,
            active_domain=active_domain
        )

        return WordPolicyOutput(
            module_name=None,
            system_action=sys_response,
            delexicalized_response=delex_sys_response,
            active_domain=active_domain
        )

class LLMICLasDST(DSTBaseforPPN):
    def __init__(self, module_name: str, llmicl: LLMICL):
        super().__init__()
        self.module_name = module_name
        self.llmicl = llmicl

    def init_session(self):
        return self.llmicl.dst_init_session()

    def update(self, *args, **kwargs):
        dst_output = self.llmicl.dst_update(*args, **kwargs)
        ds_vector = self.make_dialogue_state_vector(dst_output.dialogue_state)

        dst_output.module_name = self.module_name
        dst_output.module_state_vector = ds_vector
        return dst_output

    def update_response(self, *args, **kwargs):
        dst_output = self.llmicl.dst_update_response(*args, **kwargs)
        dst_output.module_name = self.module_name
        return dst_output

class LLMICLasPolicy(PolicyBase):
    def __init__(self, module_name: str, llmicl: LLMICL):
        super().__init__()
        self.module_name = module_name
        self.llmicl = llmicl

    @property
    def dim_module_state(self) -> int:
        return 0
    
    @property
    def dim_module_output(self) -> int:
        return 0

    def init_session(self):
        return self.llmicl.init_session_policy()

    def predict(self, *args, **kwargs):
        policy_output = self.llmicl.policy_predict(*args, **kwargs)
        policy_output.module_name = self.module_name
        return policy_output

def build_e2e_as_modules(
        module_name: str, model_name: str, device: torch.device
    ) -> Tuple[LLMICLasDST, LLMICLasPolicy]:
    llmicl = LLMICL(model_name=model_name, device=device)
    dst = LLMICLasDST(module_name=module_name, llmicl=llmicl)
    policy = LLMICLasPolicy(module_name=module_name, llmicl=llmicl)
    return dst, policy
