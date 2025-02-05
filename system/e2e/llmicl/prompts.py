import re
import json
from copy import deepcopy
from logging import getLogger
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Union
from collections import defaultdict

from openai import (
    NotGiven,
    NOT_GIVEN,
)
from openai.types.chat.completion_create_params import ResponseFormat

from convlab2.util.multiwoz.state import default_state

from system.utils import lexicalize_domain_slot_tuples
from utils import set_logger
logger = getLogger(__name__)
set_logger(logger)

@dataclass
class Example:
    dialogue_id: str
    turn_id: int
    context: List[Tuple[str, str]]
    belief_state_diff: dict
    belief_state: dict
    db_results: Dict[str, int]
    delexicalized_system_response: str

DOMAIN_TRACKING_SYSTEM_PROMPT = """
Determine which domain is considered in the following dialogue situation.
Choose one domain from this list:
- restaurant
- hotel
- attraction
- taxi
- train
- hospital
- police
Answer with only one word, the selected domain from the list.
You have to always select the closest possible domain.
Consider the last domain mentioned, so focus mainly on the last utterance.
"""

DOMAIN_TRACKING_FEWSHOT_EXAMPLES = [
    dict(
        context=[
            ["user", "I need a cheap place to eat"],
            ["sys", "We have several not expensive places available. What food are you interested in?"],
            ["user", "Chinese food."]
        ],
        active_domain="restaurant"
    ),
    dict(
        context=[
            ["user", "I also need a hotel in the north."],
            ["sys", "Ok, can I offer you the Molly's place?"],
            ["user", "What is the address?"]
        ],
        active_domain="hotel",
    ),
    dict(
        context=[
            ["user", "What is the address?"],
            ["sys", "It's 123 Northfolk Road."],
            ["user", "That's all. I also need a train from London."]
        ],
        active_domain="train"
    )
]

STATE_TRACKING_SYSTEM_PROMPTS = {
"hotel": """Capture entity values from last utterance of the converstation between a customer and a chatbot.
Focus only on the values mentioned in the last utterance.
Capture pairs of key and value in JSON format.
Values that should be captured are:
- "area": the area where the hotel is located (choose from north/east/west/south/centre)
- "internet": if the customer needs free internet (choose from yes/no)
- "parking": if the customer needs free parking (choose from yes/no)
- "stars": the number of stars the hotel has (choose from 0/1/2/3/4/5)
- "type": the type of the hotel (choose from hotel/guesthouse)
- "pricerange": the price range of the hotel (choose from cheap/moderate/expensive)
- "name": name of the hotel
- "bookstay": length of the stay
- "bookday": the day of the booking
- "bookpeople": how many people should be booked for
Do not capture any other values!
If not specified, do not include the key in the JSON.""",

"attraction": """Capture entity values from last utterance of the converstation between a customer and a chatbot.
Focus only on the values mentioned in the last utterance.
Capture pairs of key and value in JSON format.
Values that should be captured are:
- "type": the type of attraction (e.g., entertainment/nightclub/swimmingpool/museum/gallery/concert/stadium)
- "area": the area where the attraction is located (choose from north/east/west/south/centre)
- "name": the name of the attraction
Do not capture any other values!
If not specified, do not include the key in the JSON.""",

"restaurant": """Capture entity values from last utterance of the converstation between a customer and a chatbot.
Focus only on the values mentioned in the last utterance.
Capture pairs of key and value in JSON format.
Values that should be captured are:
- "pricerange": the price range of the restaurant (choose from cheap/moderate/expensive)
- "area": the area where the restaurant is located (choose from north/east/west/south/centre)
- "food": the type of food the restaurant serves
- "name": the name of the restaurant
- "bookday": the day of the booking
- "booktime": the time of the booking
- "bookpeople": for how many people is the booking made
Do not capture any other values!
If not specified, do not include the key in the JSON.""",

"train": """Capture entity values from last utterance of the converstation between a customer and a chatbot.
Focus only on the values mentioned in the last utterance.
Capture pairs of key and value in JSON format.
Values that should be captured are:
- "arriveBy": what time the train should arrive (format is HH:MM)
- "leaveAt": what time the train should leave (format is HH:MM)
- "day": what day the train should leave (choose from monday/tuesday/wednesday/thursday/friday/saturday/sunday)
- "departure": the departure station
- "destination": the destination station
- "bookpeople": how many people the booking is for
Do not capture any other values!
If not specified, do not include the key in the JSON.""",

"taxi": """Capture entity values from last utterance of the converstation between a customer and a chatbot.
Focus only on the values mentioned in the last utterance.
Capture pairs of key and value in JSON format.
Values that should be captured are:
- "arriveBy": what time the train should arrive (format is HH:MM)
- "leaveAt": what time the train should leave (format is HH:MM)
- "departure": the departure station
- "destination": the destination station
Do not capture any other values!
If not specified, do not include the key in the JSON.""",

"hospital": """Capture entity values from last utterance of the converstation between a customer and a chatbot.
Focus only on the values mentioned in the last utterance.
Capture pairs of key and value in JSON format.
Values that should be captured are:
- "department": the department of interest
Do not capture any other values!
If not specified, do not include the key in the JSON.""",
}

RESPONSE_GENERATION_SYSTEM_PROMPT = {
"hotel": """You are an assistant that helps people to book a hotel.
You can search for a restaurant by customer's constraints such as pricerange and area.
There is also a number of hotels in the database currently corresponding to the constraints.
Once you find a hotel, you can provide the customer with the following information if the customer asks for it.
Note that avoid listing out the hotel details separately. Instead, incorporate only the necessary information into a single, concise response.
- "address": the address of the hotel
- "area": the area where the hotel is located
- "choice": the number of hotels found
- "name": the name of the hotel
- "phone": the phone number of the hotel
- "postcode": the postcode of the hotel
- "pricerange": the price range of the hotel
- "stars": the number of stars the hotel has
- "type": the type of the hotel
- "Ref": the reference number of the booking
**Do not include specific values in the response text, but use placeholders instead so that the response can be used for different values. The placeholders should be in the format of the above keys enclosed in square brackets (e.g., [address], [choice]).**""",

"attraction": """You are an assistant that helps people to find an attraction.
You can search for an attraction by customer's constraints such as name and area.
There is also a number of attractions in the database currently corresponding to the constraints.
Once you find an attraction, you can provide the customer with the following information if the customer asks for it.
Note that avoid listing out the hotel details separately. Instead, incorporate only the necessary information into a single, concise response.
- "address": the address of the attraction
- "area": the area where the attraction is located
- "choice": the number of attractions found
- "entrance fee": the entrance fee of the attraction
- "name": the name of the attraction
- "phone": the phone number of the attraction
- "postcode": the postcode of the attraction
- "type": the type of the attraction
**Do not include specific values in the response text, but use placeholders instead so that the response can be used for different values. The placeholders should be in the format of the above keys enclosed in square brackets (e.g., [address], [entrance fee]).**""",

"restaurant": """You are an assistant that helps people to book a restaurant.
You can search for a restaurant by customer's constraints such as pricerange and area.
There is also a number of restaurants in the database currently corresponding to the constraints.
Once you find a restaurant, you can provide the customer with the following information if the customer asks for it.
Note that avoid listing out the hotel details separately. Instead, incorporate only the necessary information into a single, concise response.
- "address": the address of the restaurant
- "area": the area where the restaurant is located
- "choice": the number of restaurants found
- "food": the type of food the restaurant serves
- "name": the name of the restaurant
- "phone": the phone number of the restaurant
- "postcode": the postcode of the restaurant
- "pricerange": the price range of the restaurant
- "Ref": the reference number of the booking
**Do not include specific values in the response text, but use placeholders instead so that the response can be used for different values. The placeholders should be in the format of the above keys enclosed in square brackets (e.g., [address], [choice]).**""",

"train": """You are an assistant that helps people to book a train.
You can search for a train by customer's constraints such as departure and destination.
There is also a number of trains in the database currently corresponding to the constraints.
Once you find a train, you can provide the customer with the following information if the customer asks for it.
Note that avoid listing out the hotel details separately. Instead, incorporate only the necessary information into a single, concise response.
- "arriveBy": what time the train should arrive
- "leaveAt": what time the train should leave
- "choice": the number of trains found
- "day": what day the train should leave
- "departure": the departure station
- "destination": the destination station
- "duration": the duration of the train ride
- "price": the price of the ticket
- "trainID": the ID of the train
- "Ref": the reference number of the booking
**Do not include specific values in the response text, but use placeholders instead so that the response can be used for different values. The placeholders should be in the format of the above keys enclosed in square brackets (e.g., [arriveBy], [choice]).**""",

"taxi": """You are an assistant that helps people to book a taxi.
You can search for a taxi by customer's constraints such as departure and destination.
There is also a number of taxis in the database currently corresponding to the constraints.
Once you find a taxi, you can provide the customer with the following information if the customer asks for it.
Note that avoid listing out the hotel details separately. Instead, incorporate only the necessary information into a single, concise response.
- "arriveBy": what time the taxi should arrive
- "leaveAt": what time the taxi should leave
- "departure": the pickup location of the taxi
- "destination": the drop-off location of the taxi
- "phone": the phone number of the taxi
- "type": the type of the taxi
**Do not include specific values in the response text, but use placeholders instead so that the response can be used for different values. The placeholders should be in the format of the above keys enclosed in square brackets (e.g., [arriveBy], [type]).**""",

"hospital": """You are an assistant that helps people to find a hospital.
You can search for a hospital by customer's constraints such as department.
There is also a number of hospitals in the database currently corresponding to the constraints.
Once you find a hospital, you can provide the customer with the following information if the customer asks for it.
Note that avoid listing out the hotel details separately. Instead, incorporate only the necessary information into a single, concise response.
- "address": the address of the hospital
- "department": the department of the hospital
- "phone": the phone number of the hospital
- "postcode": the postcode of the hospital
**Do not include specific values in the response text, but use placeholders instead so that the response can be used for different values. The placeholders should be in the format of the above keys enclosed in square brackets (e.g., [address], [department]).**""",

"police": """You are an assistant that helps people to find a police station.
You can search for a police station by customer's constraints such as name.
There is also a number of police stations in the database currently corresponding to the constraints.
Once you find a police station, you can provide the customer with the following information if the customer asks for it.
Note that avoid listing out the hotel details separately. Instead, incorporate only the necessary information into a single, concise response.
- "address": the address of the police station
- "name": the name of the police station
- "phone": the phone number of the police station
- "postcode": the postcode of the police station
**Do not include specific values in the response text, but use placeholders instead so that the response can be used for different values. The placeholders should be in the format of the above keys enclosed in square brackets (e.g., [address], [name]).**""",
}

def context_tupes2str(
        context_tuples: List[Tuple[str, str]],
        speaker_prefix_mapping: Dict[str, str],
        max_context_turns: int
    ) -> str:
    context_pieces = []
    for speaker, text in context_tuples:
        if text == "null":
            continue
        prefix = speaker_prefix_mapping[speaker]
        context_pieces.append(f"{prefix} {text}")
    if len(context_pieces) > max_context_turns:
        context_pieces = context_pieces[-max_context_turns:]
    return "\n".join(context_pieces)

def domain_belief_state_dict2jsonstr(domain_belief_state: dict) -> str:
    non_values = ["", "not mentioned"]
    flat_domain_bs = {}
    for constraint_type, constraints in domain_belief_state.items():
        for slot, value in constraints.items():
            if slot == "booked" or value in non_values:
                continue
            flat_domain_bs[slot] = value
    return json.dumps(flat_domain_bs)

def remove_domain_key_from_delex_response(delexicalized_system_response: str):
    """
    Remove domain key from delexicalized system response
    Args:
        delexicalized_system_response: Delexicalized system response
        e.g., "[hotel_name] is a good choice."
    Returns:
        Delexicalized system response without domain key
        e.g., "[name] is a good choice."
    """
    domains = ["restaurant", "hotel", "attraction", "taxi", "train", "hospital", "police"]
    for domain in domains:
        delexicalized_system_response = re.sub(fr"\[{domain}_", "[", delexicalized_system_response)
        # Special case for taxi domain
        delexicalized_system_response = re.sub(r"\[taxi_types", "[type", delexicalized_system_response)
        delexicalized_system_response = re.sub(r"\[taxi_phone", "[phone", delexicalized_system_response)
    return delexicalized_system_response

def remove_white_spaces_before_punctuation(text: str) -> str:
    """
    Remove white spaces before punctuation
    Args:
        - text: e.g., "I found [choice] hotels in the [area] . How about a [type] : [name] ?"
    Returns:
        - e.g., "I found [choice] hotels in the [area]. How about a [type]: [name]?"
    """
    return re.sub(r"\s+([.,:;!?])", r"\1", text)


def lexicalize_response(delexicalized_response: str, belief_state: dict, entities: Dict[str, list], active_domain: str) -> str:
    """
    Lexicalize delexicalized system response generated by the model
    Args:
        - delex_response: Delexicalized system response
            e.g., I found [choice] hotels in the [area]. How about we start with [name]?
        - entities: Entities found in the database
            e.g., {"Hotel": [{"name": "Molly's", "area": ...}, ...], "Restaurant": [...], ...}
    """
    # 1. Find all placeholders in the delexicalized response
    placeholders = re.findall(r"\[([^\[\]]+)\]", delexicalized_response)

    # 2. Set up (domain slot)-values mapping for each placeholder
    domain_slot_tuples = []
    for placeholder in placeholders:
        # Special case for taxi domain
        if (active_domain, placeholder) == ("taxi", "phone"):
            slot = f"taxi_phone"
        elif (active_domain, placeholder) == ("taxi", "type"):
            slot = f"taxi_types"
        else:
            slot = placeholder

        domain_slot_tuples.append((active_domain, slot))

    domain_slot_values = lexicalize_domain_slot_tuples(
        domain_slot_tuples=domain_slot_tuples,
        belief_state=belief_state,
        entities={d.lower(): e for d, e in entities.items()}
    )

    # 3. Replace placeholders with values
    lex_response = delexicalized_response
    for placeholder in placeholders:
        # Special case for taxi domain
        if (active_domain, placeholder) == ("taxi", "phone"):
            slot = f"taxi_phone"
        elif (active_domain, placeholder) == ("taxi", "type"):
            slot = f"taxi_types"
        else:
            slot = placeholder
        for value in domain_slot_values[(active_domain, slot)]:
            lex_response = lex_response.replace(f"[{placeholder}]", value, 1)

    return lex_response


def process_active_domain_str(active_domain_str: str) -> Union[str, None]:
    active_domain = active_domain_str.strip()
    if active_domain in ["restaurant", "hotel", "attraction", "taxi", "train", "hospital", "police"]:
        return active_domain
    return None

def domain_belief_state_diff_dict2jsonstr(domain_belief_state_diff: dict) -> str:
    flat_domain_bs = {}
    for constraint_type, constraints in domain_belief_state_diff.items():
        for slot, value in constraints.items():
            if constraint_type == "book":
                slot = f"book{slot}"
            flat_domain_bs[slot] = value
    return json.dumps(flat_domain_bs)

def domain_belief_state_diff_jsonstr2dict(domain_belief_state_diff_str: str, domain: str) -> dict:
    try:
        flat_domain_bs_diff = json.loads(domain_belief_state_diff_str)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse belief state JSON string: {domain_belief_state_diff_str}")
        return {}
    
    domain_bs = default_state()["belief_state"][domain]
    domain_bs_diff = {}
    for slot, value in flat_domain_bs_diff.items():
        if "book" in slot:
            constraint_type = "book"
            slot = slot.replace("book", "")
        else:
            constraint_type = "semi"
        if slot in domain_bs[constraint_type]:
            if constraint_type not in domain_bs_diff:
                domain_bs_diff[constraint_type] = {}
            domain_bs_diff[constraint_type][slot] = str(value)
        else:
            logger.error(f"Invalid slot in belief state: {slot} in {domain}")

    return domain_bs_diff

class PromptFormatter:
    speaker_prefix_mapping={"user": "Customer:", "sys": "Chatbot:"}

    def __init__(self):
        pass

    def active_domain_tracking(
            self,
            context: List[Tuple[str, str]],
            max_context_turns: int
        ) -> Tuple[List[Dict[str, str]], Union[ResponseFormat, NotGiven]]:
        """
        Generate prompt messages for active domain tracking
        """
        messages = []

        # 1. System prompt for domain tracking instruction
        messages.append(
            {"role": "system", "content": DOMAIN_TRACKING_SYSTEM_PROMPT}
        )

        # 2. Few-shot examples
        for example in DOMAIN_TRACKING_FEWSHOT_EXAMPLES:
            context_str = context_tupes2str(
                context_tuples=example["context"],
                speaker_prefix_mapping=self.speaker_prefix_mapping,
                max_context_turns=max_context_turns
            )
            messages += [
                {"role": "user", "content": context_str},
                {"role": "assistant", "content": example["active_domain"]}
            ]
        
        # 3. Current input prompt
        context_str = context_tupes2str(
            context_tuples=context,
            speaker_prefix_mapping=self.speaker_prefix_mapping,
            max_context_turns=max_context_turns
        )
        messages.append(
            {"role": "user", "content": context_str}
        )

        return messages, NOT_GIVEN

    def belief_state_tracking(
            self,
            context: List[Tuple[str, str]],
            active_domain: str,
            fewshot_examples: List[Example],
            max_context_turns: int
        ) -> Tuple[List[Dict[str, str]], Union[ResponseFormat, NotGiven]]:
        """
        Generate prompt messages for belief state tracking
        Return:
            - List of prompt messages
                e.g. [
                    {"role": "system", "content": "Capture entity values ..."},
                    {"role": "user", "content": "<few-shot example 1 prompt>"},
                    {"role": "assistant", "content": "<few-shot example 1 response>"},
                    ...
                    {"role": "user", "content": "<current input prompt>"}
                ]
            - Response format for OpenAI API
                e.g., {"type": "json_object"}
        """
        messages = []

        # 1. System prompt for belief state tracking instruction
        messages.append(
            {"role": "system", "content": STATE_TRACKING_SYSTEM_PROMPTS[active_domain]}
        )
        
        # 2. Few-shot examples
        for example in fewshot_examples:
            context_str = context_tupes2str(
                context_tuples=example.context,
                speaker_prefix_mapping=self.speaker_prefix_mapping,
                max_context_turns=max_context_turns
            )
            domain_bs_diff_str = domain_belief_state_diff_dict2jsonstr(
                domain_belief_state_diff=example.belief_state_diff.get(active_domain, {})
            )

            messages += [
                {"role": "user", "content": context_str},
                {"role": "assistant", "content": domain_bs_diff_str}
            ]

        # 3. Current input prompt
        context_str = context_tupes2str(
            context_tuples=context,
            speaker_prefix_mapping=self.speaker_prefix_mapping,
            max_context_turns=max_context_turns
        )
        messages.append(
            {"role": "user", "content": context_str}
        )

        return messages, {"type": "json_object"}

    def response_generation(
            self,
            context: List[Tuple[str, str]],
            belief_state: dict,
            db_results: Dict[str, int],
            active_domain: str,
            fewshot_examples: List[Example],
            max_context_turns: int,
        ) -> Tuple[List[Dict[str, str]], Union[ResponseFormat, NotGiven]]:
        """
        Generate prompt messages for response generation
        Return:
            - List of prompt messages
                e.g. [
                    {"role": "system", "content": "Capture entity values ..."},
                    {"role": "user", "content": "<few-shot example 1 prompt>"},
                    {"role": "assistant", "content": "<few-shot example 1 response>"},
                    ...
                    {"role": "user", "content": "<current input prompt>"}
                ]
            - Response format for OpenAI API
                e.g., NOT_GIVEN
        """
        messages = []

        # 1. System prompt for response generation instruction
        messages.append(
            {"role": "system", "content": RESPONSE_GENERATION_SYSTEM_PROMPT[active_domain]}
        )

        # 2. Few-shot examples
        for example in fewshot_examples:
            context_str = context_tupes2str(
                context_tuples=example.context,
                speaker_prefix_mapping=self.speaker_prefix_mapping,
                max_context_turns=max_context_turns
            )
            domain_bs_str = domain_belief_state_dict2jsonstr(
                domain_belief_state=example.belief_state[active_domain]
            )
            user_input = (
                f"{context_str}\n"
                f"Constraints: {domain_bs_str}\n"
                f"Number of {active_domain}s found: {example.db_results[active_domain.capitalize()]}"
            )
            assistant_response = remove_domain_key_from_delex_response(
                delexicalized_system_response=example.delexicalized_system_response
            )
            assistant_response = remove_white_spaces_before_punctuation(assistant_response)
            messages += [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_response}
            ]

        # 3. Current input prompt
        context_str = context_tupes2str(
            context_tuples=context,
            speaker_prefix_mapping=self.speaker_prefix_mapping,
            max_context_turns=max_context_turns
        )
        domain_bs_str = domain_belief_state_dict2jsonstr(
            domain_belief_state=belief_state[active_domain]
        )
        user_input = (
            f"{context_str}\n"
            f"Constraints: {domain_bs_str}\n"
            f"Number of {active_domain}s found: {db_results[active_domain.capitalize()]}"
        )
        messages.append(
            {"role": "user", "content": user_input}
        )

        return messages, NOT_GIVEN
