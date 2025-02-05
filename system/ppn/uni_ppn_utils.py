import re
import random
from logging import getLogger
from collections import defaultdict
from typing import Optional, List, Tuple, Dict, Any

from convlab2.util.multiwoz.multiwoz_slot_trans import REF_USR_DA
from convlab2.util.multiwoz.dbquery import Database

from utils import set_logger

logger = getLogger(__name__)
set_logger(logger)

def context_tuples2str(context_tuples: List[Tuple[str, str]], max_context_turns: int,
                       speaker2prefix_mapping: Dict[str, str]) -> str:
    """
    Convert a list of context tuples to a string.
    Args:
        - context: List of context tuples, where each tuple is (speaker_id, utterance).
        - max_context_turns: Maximum number of context turns to keep. Set to 0 to keep all turns.
        - speaker2prefix_mapping: Mapping from speaker_id to prefix (e.g., {"user": "User:", "system": "System:"})
    Returns:
        - context_str: String representation of the context.
    """
    context_str = " ".join([f"{speaker2prefix_mapping[speaker]} {utterance}"
                            for speaker, utterance in context_tuples[-max_context_turns:]])
    return context_str

def da_tuples2structured_str(da_tuples: List[Tuple[str, str, str, str]]) -> str:
    """
    Convert a list of dialogue act tuples to a structured string.
    Args:
        - da_tuples: List of dialogue act tuples, where each tuple is (intent, domain, slot, value).
    Returns:
        - da_str: Structured string representation of the dialogue acts.
            e.g., "Inform-Attraction(area='centre', name='The Eagle') Inform-Restaurant(food='Chinese')"
    """
    da_dict = {}
    for intent, domain, slot, value in da_tuples:
        intent_domain = f"{intent}-{domain}"
        if intent_domain not in da_dict:
            da_dict[intent_domain] = {}
        if slot not in da_dict[intent_domain]:
            da_dict[intent_domain][slot] = []
        da_dict[intent_domain][slot].append(value)
    
    da_strs = []
    for domain_intent, slots in da_dict.items():
        da_strs.append(
            f"{domain_intent}(" + ", ".join([f"{slot}='{', '.join(values)}'" for slot, values in slots.items()]) + ")"
        )
    da_str = " ".join(da_strs)
    return da_str

def extract_updated_belief_state(
        prev_belief_state: Dict[str, Dict[str, Dict[str, str]]],
        belief_state: Dict[str, Dict[str, Dict[str, str]]]
    ) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Extract the slot-values that have been updated from the previous belief state to the current belief state.
    """
    updated_bs = {}
    for domain, domain_bs in belief_state.items():
        updated_bs[domain] = {}
        for constraint_type, constraints in domain_bs.items():
            updated_bs[domain][constraint_type] = {}
            for slot, value in constraints.items():
                if prev_belief_state[domain][constraint_type][slot] != value:
                    updated_bs[domain][constraint_type][slot] = value
    return updated_bs

def belief_state2structured_str(belief_state: Dict[str, Dict[str, Dict[str, str]]], exclude_empty_slots: bool = True) -> str:
    """
    Convert a belief state to a structured string.
    Args:
        - belief_state: Belief state, where each domain has a dictionary of constraints.
            e.g., {
                "Attraction": {
                    "semi": {"area": "centre", "name": "The Eagle"},
                    "book": {}
                },
                "Restaurant": {
                    "semi": {"food": "Chinese"},
                    "book": {"people": "2", "day": "sunday", "time": "18:45"}
                }
            }
    Returns:
        - belief_state_str: Structured string representation of the belief state.
            e.g., "Attraction semi (area='centre', name='The Eagle') Restaurant semi (food='Chinese') book (people='2', day='sunday', time='18:45')"
    """
    # 1 Exclude empty slot-values
    if exclude_empty_slots:
        bs_wo_empv = {}
        for domain, domain_bs in belief_state.items():
            bs_wo_empv[domain] = {}
            for constraint_type, constraints in domain_bs.items():
                bs_wo_empv[domain][constraint_type] = {}
                for slot, value in constraints.items():
                    if value:
                        bs_wo_empv[domain][constraint_type][slot] = value
    else:
        bs_wo_empv = belief_state

    # 2 Convert to string
    belief_state_str = ""
    for domain, consts in bs_wo_empv.items():
        semi_const_str = ""
        if consts["semi"]:
            semi_const_str = "semi (" + ", ".join([f"{s}='{v}'" for s,v in consts["semi"].items()]) + ")"
        book_const_str = ""
        if consts["book"]:
            book_const_str = "book (" + ", ".join([f"{s}='{v}'" for s,v in consts["book"].items()]) + ")"
            
        if book_const_str or semi_const_str:
            belief_state_str += f" {domain}"
            if book_const_str:
                belief_state_str += f" {book_const_str}"
            if semi_const_str:
                belief_state_str += f" {semi_const_str}"
    
    return belief_state_str.strip()

def request_state2structured_str(request_state: Dict[str, List[str]]) -> str:
    """
    Convert a request state to a structured string.
    Args:
        - request_state: Request state, where each domain has a list of slots.
            e.g., {
                "Attraction": {"area": 0, "name": 0}
            }
    Returns:
        - request_state_str: Structured string representation of the request state.
            e.g., "Attraction (area, name)"
    """
    request_state_str = ""
    for domain, slots in request_state.items():
        request_state_str += f" {domain} (" + ", ".join(slots) + ")"
    return request_state_str.strip()  

def dialogue_state2structured_strs(
        dialogue_state: Dict[str, Any], multiwoz_db: Database
    ) -> Tuple[str, str, str, str]:

    # 1 Belief state
    belief_state_str = belief_state2structured_str(dialogue_state["belief_state"])

    # 2 Request state
    request_state_str = request_state2structured_str(dialogue_state["request_state"])

    # 3 Num entities
    ## 3.1 Estimate active domains
    active_domains = []
    for domain, domain_bs in dialogue_state["belief_state"].items():
        if any(domain_bs["semi"].values()) or any(domain_bs["book"].values()):
            active_domains.append(domain)

    ## 3.2 Count entities
    num_entities_str = ""
    for domain in active_domains:
        entities = multiwoz_db.query(
            domain=domain, constraints=dialogue_state["belief_state"][domain]["semi"].items()
        )
        num_entities_str += f" {domain}={len(entities)}"
    num_entities_str = num_entities_str.strip()

    # 4 Is terminated
    terminated_str = str(dialogue_state["terminated"])

    return (belief_state_str, request_state_str, num_entities_str, terminated_str)

def user_da_tuples2tokens(da_tuples: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str]]:
    """
    Convert a list of dialogue act tuples to a list of tokens.
    Args:
        - da_tuples: List of dialogue act tuples, where each tuple is (intent, domain, slot, value).
    Returns:
        - da_tokens: List of (token, value) pairs, where each token is "user_da.intent.domain.slot".
    """
    da_tokens = []
    for intent, domain, slot, value in da_tuples:
        da_tokens.append(
            [f"user_da.{intent}.{domain}.{slot}", value]
        )
    return da_tokens

def belief_state2tokens(belief_state: Dict[str, Dict[str, Dict[str, str]]]) -> List[Tuple[str, str]]:
    bs_tokens = []
    for domain, constraints in belief_state.items():
        for constraint_type, slots in constraints.items():
            for slot, value in slots.items():
                if constraint_type == "book" and slot == "booked":
                    continue
                bs_tokens.append((f"belief_state.{domain}.{constraint_type}.{slot}", value))
    return bs_tokens

def request_state2tokens(request_state: Dict[str, List[str]]) -> List[str]:
    rs_tokens = []
    for domain, slots in request_state.items():
        for slot in slots:
            rs_tokens.append(f"request_state.{domain}.{slot}")
    return rs_tokens

def system_da_tuples2tokens(da_tuples: List[Tuple[str, str, str, str]]) -> List[str]:
    da_tokens = []
    for intent, domain, slot, value in da_tuples:
        da_tokens.append(f"system_da.{intent}.{domain}.{slot}.{value}")
    return da_tokens

def delexicalize_system_da_tuple(
        system_da_tuples: Tuple[str, str, str, str],
        system_da_value_voc: Dict[Tuple[str, str, str], List[str]]
    ) -> Tuple[str, str, str, str]:
    delex_da_tuples = []
    for intent, domain, slot, value in system_da_tuples:
        if all([
            system_da_value_voc[(intent, domain, slot)] != ["none"],
            system_da_value_voc[(intent, domain, slot)] != ["?"]
        ]):
            value = "*"
        delex_da_tuples.append((intent, domain, slot, value))
    return delex_da_tuples

def redelexicalize_larl_response(delexicalized_response: str, active_domain: Optional[str]) -> List[str]:
    """
    Re-delexicalize LaRL-based deleixalized response and tokenize it.
    Args:
        - delexicalized_response: System response string delexicalized by LaRL's rule
            e.g., How about the [restaurant_name] in the [value_area] part of town?
    Returns:
        - delex_system_da_str: Delexicalized system response string.
            e.g., How about the restaurant.name in the restaurant.area part of town?

    Rules:
    [
        '[attraction_name]', '[attraction_address]', '[attraction_phone]', '[attraction_postcode]',
        '[hospital_address]', '[hospital_postcode]', '[hospital_phone]', '[hospital_department]',
        '[hotel_name]', '[hotel_address]', '[hotel_postcode]', '[hotel_phone]', 
        '[police_name]', '[police_phone]', '[police_address]', '[police_postcode]',
        '[restaurant_name]', '[restaurant_postcode]', '[restaurant_phone]', '[restaurant_address]',
        '[taxi_type]', '[taxi_phone]',
    ] -> as is
    '[train_id]' -> {'train': 'trainID'}
    '[*_reference]' -> {'*': 'Ref'}
    '[hospital_name]' -> {'hospital': name?}

    '[value_count]' -> {
        'attraction': ['entrance fee', 'choice'],
        'hotel': ['people', 'stars', 'choice'],
        'restaurant': ['people', 'choice'],
        'train': ['people', 'duration', 'choice'],
    }
    '[value_place]' -> {'train': ['departure', 'destination']}
    '[value_time]' -> {'train': ['leaveAt', 'arriveBy'], 'restaurant': 'time'}
    '[value_price]' -> {'train': 'price', 'attraction': 'entrance fee'}
    '[value_day]' -> {'hotel': 'day', 'restaurant': 'day', 'train': 'day'}
    '[value_pricerange]' -> {'hotel': 'pricerange', 'restaurant': 'pricerange'}
    '[value_area]' -> {'attraction': 'area', 'hotel': 'area', 'restaurant': 'area'}
    '[value_food]' -> {'restaurant': 'food'}
    """

    def convert_specific_domain_slot(domain: str, slot: str):
        if domain == 'train' and slot == 'id':
            return f'{domain}.trainID'
        elif slot == 'reference':
            return f'{domain}.Ref'
        return f'{domain}.{slot}'

    def convert_vague_domain_slot(slot: str, slot_index: int, tokens: List[str]):
        tokens = [token.lower() for token in tokens]
        tokens = tokens + ["", "", ""] # padding

        if slot == 'count':
            if any([
                "there are" in " ".join(tokens[slot_index-2:slot_index]),
                "i have" in " ".join(tokens[slot_index-2:slot_index]),
                "we have" in " ".join(tokens[slot_index-2:slot_index]),
                "find" in " ".join(tokens[slot_index-2:slot_index]),
            ]):
                # e.g., there are [value_count] attractions
                return f'{active_domain}.choice'
            elif "entrance fee" in " ".join(tokens[slot_index-3:slot_index]):
                # e.g., entrance fee is [value_count]
                return 'attraction.entrance fee'
            elif "star" in tokens[slot_index+1]:
                # e.g., [value_count] stars hotel
                return 'hotel.stars'
            elif "minute" in tokens[slot_index+1]:
                # e.g., [value_count] minutes
                if active_domain in ['train', 'restaurant']:
                    return f'{active_domain}.duration'
                else:
                    logger.warning(f"Unexpected active domain '{active_domain}' and slot '{slot}' in response {' '.join(tokens)}")
                    return f'{random.choice(["train", "restaurant"])}.duration'
            elif any([
                "people" == tokens[slot_index+1], # e.g., [value_count] people
                "table" in tokens[slot_index-2:slot_index], # e.g., table for [value_count]
                "ticket" in " ".join(tokens[slot_index-2:slot_index+2]) # e.g., ticket for [value_count]
            ]):
                if active_domain in ['hotel', 'restaurant', 'train']:
                    return f'{active_domain}.people'
                else:
                    logger.warning(f"Unexpected active domain '{active_domain}' and slot '{slot}' in response {' '.join(tokens)}")
                    return f"{random.choice(['hotel', 'restaurant', 'train'])}.people"
            elif "night" in tokens[slot_index+1] or "day" in tokens[slot_index+1]: 
                return 'hotel.stay'
            else:
                logger.warning(f"Unexpected active domain '{active_domain}' and slot '{slot}' in response {' '.join(tokens)}")
                return f'{random.choice(["attraction", "hotel", "restaurant", "train"])}.choice'

        elif slot == 'place':
            if any([
                "arriv" in " ".join(tokens[slot_index-2:slot_index]), # e.g., arrive at [value_place]
                "to" in " ".join(tokens[slot_index-2:slot_index]) # e.g., to [value_place]
            ]):
                if active_domain in ['train', 'taxi']:
                    return f'{active_domain}.destination'        
                else:
                    logger.warning(f"Unexpected active domain '{active_domain}' and slot '{slot}' in response {' '.join(tokens)}")
                    return f'{random.choice(["train", "taxi"])}.destination'    
            elif any([
                'leav' in " ".join(tokens[slot_index-2:slot_index]), # e.g., leave from [value_place]
                "from" in tokens[slot_index-2:slot_index], # e.g., from [value_place]
                "depart" in " ".join(tokens[slot_index-2:slot_index]) # e.g., depart from [value_place]
            ]):
                if active_domain in ['train', 'taxi']:
                    return f'{active_domain}.departure'
                else:
                    logger.warning(f"Unexpected active domain '{active_domain}' and slot '{slot}' in response {' '.join(tokens)}")
                    return f'{random.choice(["train", "taxi"])}.departure'
            else:
                logger.warning(f"Unexpected active domain '{active_domain}' and slot '{slot}' in response {' '.join(tokens)}")
                return f'{random.choice(["train", "taxi"])}.{random.choice(["departure", "destination"])}'

        elif slot == 'time':
            if any([
                "arriv" in " ".join(tokens[slot_index-3:slot_index]), # e.g., arrives by [value_time]
                all([p in " ".join(tokens[slot_index-4:slot_index]) for p in ['arriv', '[value_place]']])
                # e.g., arrives in [value_place] by [value_time]
            ]):
                if active_domain != 'train':
                    logger.warning(f"Unexpected active domain '{active_domain}' and slot '{slot}' in response {' '.join(tokens)}")
                return 'train.arriveBy'
            elif any([
                "leav" in " ".join(tokens[slot_index-3:slot_index]), # e.g., leaves at [value_time]
                "depart" in " ".join(tokens[slot_index-3:slot_index]), # e.g., departs at [value_time]
                all([p in " ".join(tokens[slot_index-4:slot_index]) for p in ['leav', '[value_place]']])
                # e.g., leaves from [value_place] at [value_time]
            ]):
                if active_domain != 'train':
                    logger.warning(f"Unexpected active domain '{active_domain}' and slot '{slot}' in response {' '.join(tokens)}")
                return 'train.leaveAt'
            elif active_domain == 'restaurant':
                return 'restaurant.time'
            else:
                logger.warning(f"Unexpected active domain '{active_domain}' and slot '{slot}' in response {' '.join(tokens)}")
                return random.choice(["train.arriveBy", "train.leaveAt", "restaurant.time"])
        
        elif slot == 'price':
            if active_domain == 'attraction':
                return 'attraction.entrance fee'
            elif active_domain == 'train':
                return 'train.price'
            else:
                logger.warning(f"Unexpected active domain '{active_domain}' and slot '{slot}' in response {' '.join(tokens)}")
                return random.choice(["attraction.entrance fee", "train.price"])
        
        elif slot == 'day':
            if active_domain in ['hotel', 'restaurant', 'train']:
                return f'{active_domain}.day'
            else:
                logger.warning(f"Unexpected active domain '{active_domain}' and slot '{slot}' in response {' '.join(tokens)}")
                return f'{random.choice(["hotel", "restaurant", "train"])}.day'

        elif slot == 'pricerange':
            if active_domain in ['hotel', 'restaurant']:
                return f'{active_domain}.pricerange'
            else:
                logger.warning(f"Unexpected active domain '{active_domain}' and slot '{slot}' in response {' '.join(tokens)}")
                return f'{random.choice(["hotel", "restaurant"])}.pricerange'

        elif slot == 'area':
            if active_domain in ['attraction', 'hotel', 'restaurant']:
                return f'{active_domain}.area'
            else:
                logger.warning(f"Unexpected active domain '{active_domain}' and slot '{slot}' in response {' '.join(tokens)}")
                return f'{random.choice(["attraction", "hotel", "restaurant"])}.area'

        elif slot == 'food':
            return 'restaurant.food'

        else:
            raise ValueError(f"Unexpected active domain '{active_domain}' and slot '{slot}' in response {' '.join(tokens)}")

    domains = ['attraction', 'hotel', 'restaurant', 'taxi', 'train', 'police', 'hospital']
    tokens = delexicalized_response.split()

    converted_tokens = []
    converted_slots = []
    for index, token in enumerate(tokens):
        # Special case for "[value_count],[value_count]"
        if token == "[value_count],[value_count]":
            token = "[value_count]"

        if token.startswith('[') and any([
            token.endswith(']'),
            token.endswith('].'), token.endswith('],'),
            token.endswith(']!'), token.endswith(']?')
        ]):
            punct = '' if token.endswith(']') else token[-1]
            token = token if token.endswith(']') else token[:-1]
            domain, slot = token[1:-1].split('_')
            if domain in domains:
                converted_slot = convert_specific_domain_slot(domain=domain, slot=slot)
            else:
                converted_slot = convert_vague_domain_slot(slot=slot, slot_index=index, tokens=tokens)
            converted_slot = "system_da." + converted_slot
            token = converted_slot + punct
            converted_slots.append(converted_slot)
        converted_tokens.append(token)
    return " ".join(converted_tokens), converted_slots

def redelexicalize_pptod_response(
        delexicalized_response: str, active_domain: Optional[None] = None
    ) -> Tuple[str, List[str]]:
    """
    Convert the delexicalized response to the format system_da.domain.slot
    e.g., I found [hotel_choice] hotels -> I found system_da.hotel.choice hotels
    """
    # 1. Find all placeholders in the delexicalized response
    placeholders = re.findall(r"\[([^\[\]]+)\]", delexicalized_response)

    # 2. Replace the placeholders with the format system_da.domain.slot
    domains = ["hotel", "restaurant", "attraction", "train", "taxi", "police", "hospital"]
    old2new_mapping = {}
    delex_vocab = []
    for placeholder in placeholders:
        try:
            domain, slot = placeholder.split("_", 1)
        except ValueError:
            continue
        if domain in domains:
            new_delex_token = f"system_da.{domain}.{slot}"
            old2new_mapping[f"[{placeholder}]"] = new_delex_token
            delex_vocab.append(new_delex_token)
    
    for old, new in old2new_mapping.items():
        delexicalized_response = delexicalized_response.replace(old, new)
    
    return delexicalized_response, delex_vocab

def redelexicalize_llmicl_response(
        delexicalized_response: str, active_domain: Optional[str]
    ) -> Tuple[str, List[str]]:
    """
    Convert the delexicalized response to the format system_da.domain.slot
    e.g., I found [choice] hotels -> I found system_da.hotel.choice hotels
    """
    if active_domain is None:
        return delexicalized_response, []
    
    # 1. Find all placeholders in the delexicalized response
    placeholders = re.findall(r"\[([^\[\]]+)\]", delexicalized_response)

    # 2. Replace the placeholders with the format system_da.domain.slot
    old2new_mapping = {}
    delex_vocab = []
    for placeholder in placeholders:
        # Special case for taxi domain
        if (active_domain, placeholder) == ("taxi", "phone"):
            slot = f"taxi_phone"
        elif (active_domain, placeholder) == ("taxi", "type"):
            slot = f"taxi_types"
        else:
            slot = placeholder
            # Check if the domain-slot is valid
            if all([
                slot not in REF_USR_DA[active_domain.capitalize()],
                slot not in ["choice", "Ref"]
            ]):
                logger.warning(f"Invalid domain-slot pair: {active_domain}/{slot}")
                continue

        new_delex_token = f"system_da.{active_domain}.{slot}"
        old2new_mapping[f"[{placeholder}]"] = new_delex_token
        delex_vocab.append(new_delex_token)

    for old, new in old2new_mapping.items():
        delexicalized_response = delexicalized_response.replace(old, new)

    return delexicalized_response, delex_vocab
