import os
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Generator

from tqdm import tqdm
from convlab2.util.multiwoz.dbquery import Database
from convlab2.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA

abs_dir = os.path.dirname(os.path.abspath(__file__))

def detect_active_domain_from_sys_state(prev_active_domains: List[str], metadata: dict) -> List[str]:
    active_domains = []
    for domain, domain_state in metadata.items():
        if any(domain_state["semi"].values()) or any(domain_state["book"].values()):
            if domain not in prev_active_domains:
                active_domains.append(domain.capitalize())
    return active_domains
    
def detect_active_domains_from_da(dialog_act) -> List[str]:
    active_domains = []
    for domain_intent in dialog_act:
        domain, intent = domain_intent.split('-')
        if domain in ["general", "Booking"]:
            continue
        if domain not in active_domains:
            active_domains.append(domain)
    return active_domains

def estimate_booking_domain(active_domains_history: List[List[str]]) -> Optional[str]:
    def trial(target_domains: List[str]):
        for active_domains in active_domains_history[::-1]:
            for domain in active_domains[::-1]:
                if domain in target_domains:
                    return domain
        return None
    # 1. Check main booking domains
    booking_domain = trial(["Hotel", "Restaurant", "Train"])
    if booking_domain:
        return booking_domain
    
    # 2. Check other booking domains
    booking_domain = trial(["Taxi", "Attraction"])
    if booking_domain:
        return booking_domain

    booking_domain = trial(["Hospital"])
    if booking_domain:
        return booking_domain
    
    raise ValueError("Failed to estimate booking domain")

def delexicalize_system_response(
        system_response: str,
        span_info: List[Tuple[str, str, str, int, int]],
        booking_domain: Optional[str]
    ) -> Tuple[str, List[str]]:
    tokens = system_response.split()
    bio_tags = ['O'] * len(tokens)
    delex_tokens = []
    for domain_intent, slot, value, start_idx, end_idx in span_info:
        domain, intent = domain_intent.split('-')
        try:
            sys_slot = REF_SYS_DA[domain][slot]
        except KeyError:
            try:
                sys_slot = REF_SYS_DA['Booking'][slot]
            except KeyError:
                print(f"Failed to delexicalize: {domain_intent}, {slot}, {value}")
                delex_token = "O"

        if domain == 'Booking':
            assert booking_domain is not None
            domain = booking_domain

        delex_token = f"[{domain.lower()}_{sys_slot}]"
        bio_tags[start_idx] = delex_token
        for i in range(start_idx+1, end_idx+1):
            bio_tags[i] = "I"
        delex_tokens.append(delex_token)
    
    delex_response = []
    for token, bio_tag in zip(tokens, bio_tags):
        if bio_tag == 'O':
            delex_response.append(token)
        elif bio_tag == 'I':
            continue
        else:
            delex_response.append(bio_tag)
    return " ".join(delex_response), delex_tokens

@dataclass
class MultiWOZ23Turn:
    dialogue_id: str
    turn_id: int
    context: List[Tuple[str, str]]
    belief_state: dict
    booking_domain: Optional[str]
    db_results: Dict[str, int]
    system_response: str
    delexicalized_system_response: str

class MultiWOZ23Dataset:
    split_keys = ['train', 'val', 'test']
    available_domains = ['Attraction', 'Hotel', 'Restaurant', 'Taxi', 'Train', 'Hospital', 'Police']

    def __init__(self):
        self.split_list = {}
        for split_key in self.split_keys:
            self.split_list[split_key] = []
            for line in open(os.path.join(abs_dir, 'MultiWOZ2_3', f'{split_key}ListFile.txt')):
                self.split_list[split_key].append(line.strip())

        self.data = json.load(open(os.path.join(abs_dir, 'MultiWOZ2_3/data.json')))
        delex_vocab = []
        for dialog_id, dialog in self.data.items():
            initial_domain = list(dialog["new_goal"])[0].capitalize()
            active_domains_history = [[initial_domain]]
            for turn in dialog["log"]:
                active_domains = detect_active_domains_from_da(dialog_act=turn["dialog_act"])
                if turn["turn_id"] % 2 != 0 and not active_domains:
                    active_domains = detect_active_domain_from_sys_state(
                        prev_active_domains=active_domains_history[-1],
                        metadata=turn["metadata"]
                    )
                active_domains_history.append(active_domains)

                if turn["turn_id"] % 2 != 0:
                    # Estimate booking domain
                    if "Booking" in [da.split('-')[0] for da in turn["dialog_act"]]:
                        booking_domain = estimate_booking_domain(active_domains_history)
                    else:
                        booking_domain = None
                    turn["booking_domain"] = booking_domain

                    # Delexicalize system response
                    delex_text, delex_tokens = delexicalize_system_response(
                        system_response=turn["text"],
                        span_info=turn["span_info"],
                        booking_domain=booking_domain,
                    )
                    turn["delexicalized_text"] = delex_text
                    delex_vocab.extend(delex_tokens)
        
        self.delex_vocab = sorted(set(delex_vocab))

        self.db = Database()

    def get_goal(self, dialog_id: str) -> dict:
        return self.data[dialog_id]["new_goal"]

    def iter_dialogue(
            self, split: str, progress_bar: bool = False
        ) -> Generator[str, None, None]:
        dialogue_ids = self.split_list[split]
        if progress_bar:
            dialogue_ids = tqdm(
                dialogue_ids,
                desc=f"Processing {split} split",
                dynamic_ncols=True
            )
        for dial_id in dialogue_ids:
            yield dial_id
    
    def iter_system_turn(self, dialogue_id: str) -> Generator[MultiWOZ23Turn, None, None]:
        dialog = self.data[dialogue_id]
        context = []
        for turn in dialog["log"]:
            if turn["turn_id"] % 2 == 0:
                context.append(["user", turn["text"]])
            else:
                db_results = {}
                for domain in self.available_domains:
                    entities = self.db.query(
                        domain=domain.lower(),
                        constraints=turn["metadata"][domain.lower()]["semi"].items()
                    )
                    db_results[domain] = len(entities)

                yield MultiWOZ23Turn(
                    dialogue_id=dialogue_id,
                    turn_id=turn["turn_id"],
                    context=deepcopy(context),
                    belief_state=turn["metadata"],
                    booking_domain=turn["booking_domain"],
                    db_results=db_results,
                    system_response=turn["text"],
                    delexicalized_system_response=turn["delexicalized_text"]
                )

                context.append(["sys", turn["text"]])

if __name__ == "__main__":
    multiwoz = MultiWOZ23Dataset()
    breakpoint()
