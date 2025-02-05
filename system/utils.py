from collections import defaultdict
from typing import List, Tuple, Dict, Any

def lexicalize_domain_slot_tuples(
        domain_slot_tuples: List[Tuple[str, str]],
        belief_state: dict,
        entities: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[Tuple[str, str], str]:
    """
    Lexicalize delexicalized domain-slot tuples in dialogue act.
    Args:
        - domain_slot_tuples: List of domain-slot tuples in dialogue act
            e.g., [('restaurant', 'name'), ('restaurant', 'area')]
        - belief_state: Belief state
        - entities: Retrieved entities by domain
            e.g., {'restaurant': [{'name': 'The Golden Curry', 'area': 'centre'}, ...], ...}
    Returns:
        - domain_slot_dict: Lexicalized domain-slot dictionary
            e.g., {(restaurant, name): ['The Golden Curry'], (restaurant, area): ['centre']}
    """
    # 1. count domain-slot
    domain_slot_count = {}
    for domain, slot in domain_slot_tuples:
        domain_slot_count[(domain, slot)] = domain_slot_count.get((domain, slot), 0) + 1

    # 1. lexicalize domain-slot
    domain_slot_dict = defaultdict(list)
    for (domain, slot), count in domain_slot_count.items():
        for entity_index in range(count):
            # initialize with belief state
            slot_value_mapping = {
                'choice': str(len(entities[domain])),
                **belief_state[domain]['semi'],
                **belief_state[domain]['book']
            }
            # then overwrite with entity's value
            if len(entities[domain]) != 0:
                entity = entities[domain][entity_index % len(entities[domain])]
                slot_value_mapping.update(entity)

            value = slot_value_mapping.get(slot, "unknown")
            if not isinstance(value, (str, int)):
                value = "unknown"
            domain_slot_dict[(domain, slot)].append(value)

    return domain_slot_dict
