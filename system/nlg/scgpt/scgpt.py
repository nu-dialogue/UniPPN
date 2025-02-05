import os
import random
import string
from typing import List, Tuple

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from convlab2.nlg.scgpt.utils import tuple2seq

from system.module_base import NLGBase
from system.data import NLGOutput
from system.nlg.utils import remove_ws_before_punctuation

abs_dir = os.path.dirname(os.path.abspath(__file__))

class SCGPT(NLGBase):
    module_name = "scgpt_nlg"
    def __init__(self, device: torch.device) -> None:
        model_name_or_path = 'ohashi56225/scgpt-multiwoz-sys'
        self.device = device
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def dim_module_state(self) -> int:
        return 0

    @property
    def dim_module_output(self) -> int:
        return 0

    def init_session(self) -> None:
        pass

    def generate(self, system_action: List[Tuple[str, str, str, str]]) -> NLGOutput:
        # SCGPT cannot generate convlab2-style ref nums, so we replace them with 
        # ref nums in the original multiwoz format
        # true_ref_num = None
        # fake_ref_num = None
        # for idsv in system_action:
        #     if idsv[2] == "Ref":
        #         true_ref_num = idsv[3]
        #         fake_ref_num = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
        #         idsv[3] = fake_ref_num
        #         break

        raw_text = tuple2seq(system_action)
        raw_text += " &"
        input_ids = self.tokenizer.encode(raw_text, return_tensors="pt",
                                          add_special_tokens=False)
        outputs = self.model.generate(
            input_ids=input_ids.to(self.device),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=128,
            num_beams=1,
            do_sample=True,
            top_k=0, top_p=1.0,
            temperature=0.8 # We found that sampling with temp=0.8 is much better than greedy search
        )
        gen_ids = outputs[0, input_ids.shape[-1]:]
        text = self.tokenizer.decode(gen_ids, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        text = text.split('& ')[-1]
        text = remove_ws_before_punctuation(text)

        # Restore the ref num
        # if true_ref_num is not None:
        #     text = text.replace(fake_ref_num, true_ref_num)
        #     text = text.replace(fake_ref_num.lower(), true_ref_num)

        nlg_output = NLGOutput(
            module_name=self.module_name,
            system_response=text,
        )

        return nlg_output
