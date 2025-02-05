import os
import re
import json
import pickle
from copy import deepcopy
from logging import getLogger

import numpy as np
import torch

from convlab2.util.multiwoz.state import default_state
from convlab2.util.multiwoz.dbquery import Database

from system.policy.lava.latent_dialog.models_task import SysPerfectBD2Cat
from system.policy.lava.utils.nlp import normalize
from system.policy.lava.utils import delexicalize
from system.policy.lava.latent_dialog import domain
from system.policy.lava.latent_dialog.utils import (
    Pack,
    cast_type,
    get_detokenize
)

from system.policy.lava.corpora_inference import (
    NormMultiWozCorpus,
    BOS,
    EOS,
    PAD
)

from system.module_base import PolicyBase, WordPolicyOutput
from utils import set_logger

logger = getLogger(__name__)
set_logger(logger)

TEACH_FORCE = 'teacher_forcing'
TEACH_GEN = 'teacher_gen'
GEN = 'gen'
GEN_VALID = 'gen_valid'

placeholder_re = re.compile(r'\[(\s*[\w_\s]+)\s*\]')
number_re = re.compile(
    r'.*(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s$')
time_re = re.compile(
    r'((?:\d{1,2}[:]\d{2,3})|(?:\d{1,2} (?:am|pm)))', re.IGNORECASE)

REQ_TOKENS = {}
# DOMAIN_REQ_TOKEN = ['restaurant', 'hospital', 'hotel','attraction', 'train', 'police', 'taxi']
DOMAIN_REQ_TOKEN = ['taxi','restaurant', 'hospital', 'hotel','attraction','train','police']
ACTIVE_BS_IDX = [13, 30, 35, 61, 72, 91, 93] #indexes in the BS indicating if domain is active
NO_MATCH_DB_IDX = [-1, 0, -1, 6, 12, 18, -1] # indexes in DB pointer indicating 0 match is found for that domain, -1 mean that domain has no DB
REQ_TOKENS['attraction'] = ["[attraction_address]", "[attraction_name]", "[attraction_phone]", "[attraction_postcode]", "[attraction_reference]", "[attraction_type]"]
REQ_TOKENS['hospital'] = ["[hospital_address]", "[hospital_department]", "[hospital_name]", "[hospital_phone]", "[hospital_postcode]"] #, "[hospital_reference]"
REQ_TOKENS['hotel'] = ["[hotel_address]", "[hotel_name]", "[hotel_phone]", "[hotel_postcode]", "[hotel_reference]", "[hotel_type]"]
REQ_TOKENS['restaurant'] = ["[restaurant_name]", "[restaurant_address]", "[restaurant_phone]", "[restaurant_postcode]", "[restaurant_reference]"]
REQ_TOKENS['train'] = ["[train_id]", "[train_reference]"]
REQ_TOKENS['police'] = ["[police_address]", "[police_phone]",
                        "[police_postcode]"]  # "[police_name]",
REQ_TOKENS['taxi'] = ["[taxi_phone]", "[taxi_type]"]

REQ_DB_ATTRIBUTES = {}
REQ_DB_ATTRIBUTES['attraction'] = {"address", "area", "entrance fee", "name", "phone", "postcode", "type"}
REQ_DB_ATTRIBUTES['hospital'] = {"address", "department", "phone", "postcode"}
REQ_DB_ATTRIBUTES['hotel'] = {"address", "area", "internet", "parking", "name", "phone", "postcode", "pricerange", "stars", "type"}
REQ_DB_ATTRIBUTES['police'] = {"address", "phone", "postcode", "name"}
REQ_DB_ATTRIBUTES['restaurant'] = {"address", "area", "food", "name", "phone", "postcode", "pricerange"}
REQ_DB_ATTRIBUTES['taxi'] = {}
REQ_DB_ATTRIBUTES['train'] = {"arriveBy", "day", "departure", "destination", "duration", "leaveAt", "price", "trainID"}


def oneHotVector(num, domain, vector):
    """Return number of available entities for particular domain."""
    domains = ['restaurant', 'hotel', 'attraction', 'train']
    number_of_options = 6
    if domain != 'train':
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
        elif num == 1:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num == 2:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num == 3:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num == 4:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        elif num >= 5:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])
    else:
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
        elif num <= 2:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num <= 5:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num <= 10:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num <= 40:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        elif num > 40:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])

    return vector

def addBookingPointer(state, pointer_vector):
    """Add information about availability of the booking option."""
    # Booking pointer
    rest_vec = np.array([1, 0])
    if "book" in state['restaurant']:
        if "booked" in state['restaurant']['book']:
            if state['restaurant']['book']["booked"]:
                if "reference" in state['restaurant']['book']["booked"][0]:
                    rest_vec = np.array([0, 1])

    hotel_vec = np.array([1, 0])
    if "book" in state['hotel']:
        if "booked" in state['hotel']['book']:
            if state['hotel']['book']["booked"]:
                if "reference" in state['hotel']['book']["booked"][0]:
                    hotel_vec = np.array([0, 1])

    train_vec = np.array([1, 0])
    if "book" in state['train']:
        if "booked" in state['train']['book']:
            if state['train']['book']["booked"]:
                if "reference" in state['train']['book']["booked"][0]:
                    train_vec = np.array([0, 1])

    pointer_vector = np.append(pointer_vector, rest_vec)
    pointer_vector = np.append(pointer_vector, hotel_vec)
    pointer_vector = np.append(pointer_vector, train_vec)

    # pprint(pointer_vector)
    return pointer_vector

def addDBPointer(state, db: Database):
    """Create database pointer for all related domains."""
    domains = ['restaurant', 'hotel', 'attraction', 'train']  
    pointer_vector = np.zeros(6 * len(domains))
    db_results = {}
    num_entities = {}
    for domain in domains:
        # entities = dbPointer.queryResultVenues(domain, {'metadata': state})
        constraints = [[slot, value] for slot, value in state[domain]['semi'].items() if value] if domain in state else []
        entities = db.query(domain, constraints) #, topk=10)
        num_entities[domain] = len(entities)
        if len(entities) > 0:
            # fields = dbPointer.table_schema(domain)
            # db_results[domain] = dict(zip(fields, entities[0]))
            db_results[domain] = entities
        # pointer_vector = dbPointer.oneHotVector(len(entities), domain, pointer_vector)
        pointer_vector = oneHotVector(len(entities), domain, pointer_vector)

    return list(pointer_vector), db_results, num_entities

def delexicaliseReferenceNumber(sent, state):
    """Based on the belief state, we can find reference number that
    during data gathering was created randomly."""
    domains = ['restaurant', 'hotel', 'attraction',
               'train', 'taxi', 'hospital']  # , 'police']

    if state['history'][-1][0]=="sys":
        # print(state["booked"])
        for domain in domains:
            if state['booked'][domain]:
                for slot in state['booked'][domain][0]:
                    val = '[' + domain + '_' + slot + ']'
                    key = normalize(state['booked'][domain][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' +
                                                      key + ' ', ' ' + val + ' ')

                    # try reference with hashtag
                    key = normalize("#" + state['booked'][domain][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' +
                                                      key + ' ', ' ' + val + ' ')

                    # try reference with ref#
                    key = normalize(
                        "ref#" + state['booked'][domain][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' +
                                                      key + ' ', ' ' + val + ' ')

    return sent

def domain_mark_not_mentioned(state, active_domain):
    if active_domain not in ['hospital', 'taxi', 'train', 'attraction', 'restaurant', 'hotel'] or active_domain is None:
        return

    for s in state[active_domain]['semi']:
        if state[active_domain]['semi'][s] == '':
            state[active_domain]['semi'][s] = 'not mentioned'

def mark_not_mentioned(state):
    for domain in state:
        # if domain == 'history':
        if domain not in ['police', 'hospital', 'taxi', 'train', 'attraction', 'restaurant', 'hotel']:
            continue
        try:
            # if len([s for s in state[domain]['semi'] if s != 'book' and state[domain]['semi'][s] != '']) > 0:
            # for s in state[domain]['semi']:
            #     if s != 'book' and state[domain]['semi'][s] == '':
            #         state[domain]['semi'][s] = 'not mentioned'
            for s in state[domain]:
                if state[domain][s] == '':
                    state[domain][s] = 'not mentioned'
        except Exception as e:
            # print(str(e))
            # pprint(state[domain])
            pass

def get_summary_bstate(bstate):
    """Based on the mturk annotations we form multi-domain belief state"""
    domains = [u'taxi', u'restaurant',  u'hospital',
               u'hotel', u'attraction', u'train', u'police']
    summary_bstate = []
    for domain in domains:
        domain_active = False

        booking = []
        for slot in sorted(bstate[domain]['book'].keys()):
            if slot == 'booked':
                if bstate[domain]['book']['booked']:
                    booking.append(1)
                else:
                    booking.append(0)
            else:
                if bstate[domain]['book'][slot] != "":
                    booking.append(1)
                else:
                    booking.append(0)
        if domain == 'train':
            if 'people' not in bstate[domain]['book'].keys():
                booking.append(0)
            if 'ticket' not in bstate[domain]['book'].keys():
                booking.append(0)
        summary_bstate += booking

        for slot in bstate[domain]['semi']:
            slot_enc = [0, 0, 0]
            if bstate[domain]['semi'][slot] == 'not mentioned':
                slot_enc[0] = 1
            elif bstate[domain]['semi'][slot] == 'dont care' or bstate[domain]['semi'][slot] == 'dontcare' or bstate[domain]['semi'][slot] == "don't care":
                slot_enc[1] = 1
            elif bstate[domain]['semi'][slot]:
                slot_enc[2] = 1
            if slot_enc != [0, 0, 0]:
                domain_active = True
            summary_bstate += slot_enc

        # quasi domain-tracker
        if domain_active:
            summary_bstate += [1]
        else:
            summary_bstate += [0]

    # print(len(summary_bstate))
    assert len(summary_bstate) == 94
    return summary_bstate

def get_summary_bstate_unifiedformat(state):
    """Based on the mturk annotations we form multi-domain belief state"""
    domains = [u'taxi', u'restaurant',  u'hospital',
               u'hotel', u'attraction', u'train']#, u'police']
    bstate = state['belief_state']
    # booked = state['booked']
    # how to make empty book this format instead of an empty dictionary?
    #TODO fix booked info update in state!
    booked = {
            "taxi": [],
            "hotel": [],
            "restaurant": [],
            "train": [],
            "attraction": [],
            "hospital": []
            }

    summary_bstate = []

    for domain in domains:
        domain_active = False

        booking = []
        if len(booked[domain]) > 0:
            booking.append(1)
        else:
            booking.append(0)
        if domain == 'train':
            if not bstate[domain]['book']['people']:
                booking.append(0)
            else:
                booking.append(1)
            if booked[domain] and 'ticket' in booked[domain][0].keys():
                booking.append(1)
            else:
                booking.append(0)
        summary_bstate += booking

        if domain == "restaurant":
            book_slots = ['day', 'people', 'time']
        elif domain == "hotel":
            book_slots = ['day', 'people', 'stay']
        else:
            book_slots = []
        for slot in book_slots:
            if bstate[domain]['book'][slot] == '':
                summary_bstate.append(0)
            else:
                summary_bstate.append(1)

        for slot in bstate[domain]['semi']:
            slot_enc = [0, 0, 0]
            if bstate[domain]['semi'][slot] == 'not mentioned':
                slot_enc[0] = 1
            elif any([
                bstate[domain]['semi'][slot] == 'dont care',
                bstate[domain]['semi'][slot] == 'dontcare',
                bstate[domain]['semi'][slot] == "don't care"
            ]):
                slot_enc[1] = 1
            elif bstate[domain]['semi'][slot]:
                slot_enc[2] = 1
            if slot_enc != [0, 0, 0]:
                domain_active = True
            summary_bstate += slot_enc

        # quasi domain-tracker
        if domain_active: # 7 domains
            summary_bstate += [1]
        else:
            summary_bstate += [0]


    # add manually from action as police is not tracked anymore in unified format
    if "police" in [d.lower() for i,d,s,v in state['user_action']]:
        summary_bstate += [0, 1]
    else:
        summary_bstate += [0, 0]

    assert len(summary_bstate) == 94
    return summary_bstate

def fill_unknown_to_requestables(entity, domain):
    """Fill the requestables with 'unknown'."""
    req_attributes = REQ_DB_ATTRIBUTES[domain]
    entity = deepcopy(entity)
    for slot in req_attributes:
        if slot not in entity:
            entity[slot] = 'unknown'
    return entity

class LAVA(PolicyBase):
    module_name = "lava"

    def __init__(self, device: torch.device):
        # Check torch's default device is the same as the specified device
        assert device.index == torch.cuda.current_device(), \
            (f"Specified device {device.index} is not the default "
             f"device {torch.cuda.current_device()}")
        
        temp_path = os.path.dirname(os.path.abspath(__file__))

        model_file = os.path.join(temp_path, "reward_best.model")
        config_path = os.path.join(temp_path, "config.json")
        train_data_path = os.path.join(temp_path, "data/norm-multi-woz/train_dials.json")

        self.prev_state = default_state()
        self.prev_active_domain = None

        self.db = Database()

        self.config = Pack(json.load(open(config_path)))
        self.corpus = NormMultiWozCorpus(
            train_data_path=train_data_path, max_vocab_size=self.config.max_vocab_size
        )

        self.model = SysPerfectBD2Cat(self.corpus, self.config)
        self.model.load_state_dict(torch.load(model_file))
        self.model.to(device)
        self.model.eval()

        self.temperature_params = {
            # "py_temp": 0.1, "dec_temp": 0.1 # Default setting
            "py_temp": 0.1, "dec_temp": 0.6 # This combination seems to work better
        }

        self.dic = pickle.load(open(os.path.join(temp_path, 'utils/svdic.pkl'), 'rb'))

    @property
    def dim_module_state(self) -> int:
        return 0
    
    @property
    def dim_module_output(self) -> int:
        return 0
    
    def init_session(self) -> None:
        self.prev_state = default_state()
        self.prev_active_domain = None
        self.prev_output = ""
        self.domains = []

    def is_active(self, domain, state):

        if domain in [d.lower() for i,d,s,v in state['user_action']]:
            return True
        else:
            return False

    def get_active_domain_unified(self, prev_active_domain, prev_state, state):
        domains = ['hotel', 'restaurant', 'attraction',
                   'train', 'taxi', 'hospital', 'police']
        active_domain = None
        # print("PREV_STATE:",prev_state)
        # print()
        # print("NEW_STATE",state)
        # print()
        for domain in domains:
            if not self.is_active(domain, prev_state) and self.is_active(domain, state):
                #print("case 1:",domain)
                return domain
            elif self.is_active(domain, prev_state) and self.is_active(domain, state):
                return domain
            # elif self.is_active(domain, prev_state) and not self.is_active(domain, state):
                #print("case 2:",domain)
                # return domain
            # elif prev_state['belief_state'][domain] != state['belief_state'][domain]:
                #print("case 3:",domain)
                # active_domain = domain
        if active_domain is None:
            active_domain = prev_active_domain
        return active_domain

    def predict(self, dialogue_state: dict) -> WordPolicyOutput:
        # try:
        response, delex_response, active_domain = self.predict_response(dialogue_state)
        # except Exception as e:
        #    print('Response generation error', e)
        #    response, active_domain = self.predict_response(dialogue_state)
        #    response = 'Can I help you with anything else?'
        #    active_domain = None

        self.prev_state = deepcopy(dialogue_state)
        self.prev_active_domain = active_domain

        policy_output = WordPolicyOutput(
            module_name=self.module_name,
            system_action=response,
            delexicalized_response=delex_response,
            active_domain=active_domain
        )

        return policy_output

    def need_multiple_results(self, template):
        words = template.split()
        if "first" in words and "last" in words:
            return True
        elif "i have" in template:
            if words.count("[restaurant_name]") > 1:
                return True
            elif words.count("[restaurant_pricerange]") > 1:
                return True
            elif words.count("[hotel_name]") > 1:
                return True
            elif words.count("[attraction_name]") > 1:
                return True
            elif words.count("[train_id]") > 1:
                return True
            else:
                return False
        else:
            return False

    def predict_response(self, state):
        # breakpoint()

        # input state is in convlab format
        history = []
        for i in range(len(state['history'])):
            history.append(state['history'][i][1])

        e_idx = len(history)
        s_idx = max(0, e_idx - self.config.backward_size)
        context = []
        for turn in history[s_idx: e_idx]:
            # turn = pad_to(config.max_utt_len, turn, do_pad=False)
            context.append(turn)

        prepared_data = {}
        prepared_data['context'] = []
        prepared_data['response'] = {}

        state_history = state['history']
        bstate = deepcopy(state['belief_state'])

        # mark_not_mentioned(prev_state)
        #active_domain = self.get_active_domain_convlab(self.prev_active_domain, prev_bstate, bstate)
        active_domain = self.get_active_domain_unified(self.prev_active_domain, self.prev_state, state)

        domain_mark_not_mentioned(bstate, active_domain)

        
        # add database pointer
        pointer_vector, top_results, num_results = addDBPointer(bstate, self.db)

        # add booking pointer
        pointer_vector = addBookingPointer(bstate, pointer_vector)
        belief_summary = get_summary_bstate_unifiedformat(state)

        for t_id in range(len(context)):
            usr = context[t_id]

            if t_id == 0: #system turns
                if usr == "null":
                    usr = "<d>"
                    # booked = {"taxi": [],
                            # "restaurant": [],
                            # "hospital": [],
                            # "hotel": [],
                            # "attraction": [],
                            # "train": []}

            usr = delexicalize.delexicalise(usr.lower(), self.dic)

            # parsing reference number GIVEN belief state
            usr = delexicaliseReferenceNumber(usr, state)

            # changes to numbers only here
            digitpat = re.compile('(^| )\d+( |$)')
            usr = re.sub(digitpat, '\\1[value_count]\\2', usr)

            usr_utt = [BOS] + usr.split() + [EOS]
            packed_val = {}
            packed_val['bs'] = belief_summary
            packed_val['db'] = pointer_vector
            packed_val['utt'] = self.corpus._sent2id(usr_utt)

            prepared_data['context'].append(packed_val)

        prepared_data['response']['bs'] = prepared_data['context'][-1]['bs']
        prepared_data['response']['db'] = prepared_data['context'][-1]['db']
        results = [Pack(context=prepared_data['context'], response=prepared_data['response'])]

        # data_feed is in LaRL format
        data_feed = prepare_batch_gen(results, self.config)

        outputs = self.model_predict(data_feed)
        self.prev_output = outputs

        state_with_history = deepcopy(bstate)
        state_with_history['history'] = deepcopy(state_history)

        # fill the unknown values to the requestables
        for domain in top_results:
            for i, entity in enumerate(top_results[domain]):
                top_results[domain][i] = fill_unknown_to_requestables(entity, domain)

        # Lexicalize the output
        if active_domain in ["hotel", "attraction", "train", "restaurant"] \
            and num_results[active_domain] == 0 \
                and any([p in outputs for p in REQ_TOKENS[active_domain]]):
            # if no entities found in the DB of the active domain and the template requires some
            # values, then return a default response
            response = "I am sorry, can you say that again?"
        elif self.need_multiple_results(outputs) and num_results.get(active_domain,0) > 1:
            response = self.lexicalize_multioptions(outputs, top_results, num_results, state_with_history)
        else:
            response = self.lexicalize(outputs, top_results, num_results, state_with_history, active_domain)

        response = response.replace("free pounds", "free")
        response = response.replace("pounds pounds", "pounds")

        return response, outputs, active_domain

    def lexicalize(self, template, top_results, num_results, state_with_history, active_domain):
        # select the top result for each domain
        top_result = {}
        for domain, entities in top_results.items():
            if domain == 'train' and state_with_history['train']['semi']['arriveBy'] not in ["not mentioned", ""]:
                top_result[domain] = entities[-1]
            else:
                top_result[domain] = entities[0]
        top_result['hospital'] = self.db.query('hospital', [])[0]
        top_result['police'] = self.db.query('police', [])[0]
        top_result['taxi'] = self.db.query('taxi', [])[0]

        # populate the template with the top result
        response = self.populate_template(template, top_result, num_results, state_with_history, active_domain)

        # use other domains' values if the active domain does not have the required values 
        other_domains = list(top_result)
        if active_domain in other_domains:
            other_domains.remove(active_domain)
        while all([p in response for p in ["[", "]"]]) and other_domains:
            domain = other_domains.pop()
            if domain not in top_result:
                continue
            response = self.populate_template(template, top_result, num_results, state_with_history, domain)

        # check some common lexicalization errors
        if any([p in response for p in ["not mentioned", "dontcare"]]):
            logger.warning(f"Detected lexicalization error in {response}")

        return response

    def populate_template(self, template, top_result, num_results, state, active_domain):
        # print("template:",template)
        # print("top_results:",top_results)
        # active_domain = None if len(
        #    top_results.keys()) == 0 else list(top_results.keys())[0]
        num_results_str = str(num_results.get(active_domain, 0))

        template = template.replace('book [value_count] of', 'book one of')
        tokens = [""] + template.split()
        response = []
        for index, token in enumerate(tokens):
            if token.startswith('[') \
                and any([token.endswith(']'),
                         token.endswith('].'), token.endswith('],'),
                         token.endswith(']!'), token.endswith(']?')]):

                if len(token.split('_')) > 2:
                    # Probably a series of special tokens, e.g., "[value_count],[value_count]"
                    # find the first "]" and split there
                    token = token[:token.find(']')] + ']'

                domain, slot = re.sub(r'[\[\]\.,!\?]', '', token).split('_')
                punct = '' if token.endswith(']') else token[-1]
                token = token if token.endswith(']') else token[:-1]

                if domain == 'train' and slot == 'id':
                    slot = 'trainID'
                elif active_domain != 'train' and slot == 'price':
                    slot = 'pricerange'
                elif slot == 'reference':
                    slot = 'Ref'

                if domain in top_result and len(top_result[domain]) > 0 and slot in top_result[domain]:
                    # print('{} -> {}'.format(token, top_results[domain][slot]))
                    response.append(top_result[domain][slot])
                elif domain == 'value':
                    if slot == 'count':
                        if "there are" in " ".join(tokens[index-2:index]) or "i have" in " ".join(tokens[index-2:index]):
                            response.append(num_results_str)
                        # the first [value_count], the last [value_count]
                        elif "the" in tokens[index-2]:
                            response.append("one")
                        elif active_domain == "restaurant":
                            if "people" in tokens[index:index+1] or "table" in tokens[index-2:index]:
                                response.append(state["restaurant"]["book"]["people"])
                        elif active_domain == "train":
                            if "ticket" in " ".join(tokens[index-2:index+1]) or "people" in tokens[index:]:
                                response.append(state["train"]["book"]["people"])
                            elif index+1 < len(tokens) and "minute" in tokens[index+1]:
                                if 'train' in top_result:
                                    response.append(top_result['train']['duration'].split()[0])
                                else:
                                    response.append(token)
                        elif active_domain == "hotel":
                            if index+1 < len(tokens) and 'hotel' in top_result:
                                if "star" in tokens[index+1]:
                                    if 'stars' in top_result['hotel']:
                                        response.append(top_result['hotel']['stars'])
                                    else:
                                        response.append(token)
                                elif "nights" in tokens[index+1]:
                                    response.append(state["hotel"]["book"]["stay"])
                                elif "people" in tokens[index+1]:
                                    response.append(state["hotel"]["book"]["people"])
                            else:
                                response.append(token)
                        elif active_domain == "attraction":
                            if index + 1 < len(tokens):
                                if "pounds" in tokens[index+1] and "entrance fee" in " ".join(tokens[index-3:index]):
                                    if "attraction" in top_result:
                                        value = top_result["attraction"]['entrance fee']
                                    else:
                                        value = "?"
                                    if "?" in value:
                                        value = "unknown"
                                    # if "?" not in value:
                                    #    try:
                                    #        value = str(int(value))
                                    #    except:
                                    #        value = 'free'
                                    # else:
                                    #    value = "unknown"
                                    response.append(value)
                        # if "there are" in " ".join(tokens[index-2:index]):
                            # response.append(str(num_results))
                        # elif "the" in tokens[index-2]: # the first [value_count], the last [value_count]
                            # response.append("1")
                        else:
                            response.append(num_results_str)
                    elif slot == 'place':
                        if 'arriv' in " ".join(tokens[index-2:index]) or "to" in " ".join(tokens[index-2:index]):
                            if active_domain == "train":
                                if "train" in top_result:
                                    response.append(top_result["train"]["destination"])
                                else:
                                    response.append(token)
                            elif active_domain == "taxi":
                                response.append(state["taxi"]["semi"]["destination"])
                        elif 'leav' in " ".join(tokens[index-2:index]) or "from" in tokens[index-2:index] or "depart" in " ".join(tokens[index-2:index]):
                            if active_domain == "train":
                                if "train" in top_result:
                                    response.append(top_result["train"]["departure"])
                                else:
                                    response.append(token)
                            elif active_domain == "taxi":
                                response.append(state["taxi"]["semi"]["departure"])
                        elif "hospital" in template:
                            response.append("Cambridge")
                        else:
                            # try:
                            #     for d in state:
                            #         if d == 'history':
                            #             continue
                            #         for s in ['destination', 'departure']:
                            #             if s in state[d]["semi"]:
                            #                 response.append(state[d]["semi"][s])
                            #                 raise ValueError(f"Found {s} in {d}")
                            # except:
                            #     pass
                            # else:
                            #     response.append(token)
                            response.append("Cambridge")
                    elif slot == 'time':
                        if 'arrive' in ' '.join(response[-5:]) or 'arrival' in ' '.join(response[-5:]) or 'arriving' in ' '.join(response[-3:]):
                            if active_domain == "train" and 'train' in top_result and 'arriveBy' in top_result["train"]:
                                # print('{} -> {}'.format(token, top_results[active_domain]['arriveBy']))
                                response.append(top_result["train"]['arriveBy'])
                                continue
                            for d in state:
                                if d == 'history':
                                    continue
                                if 'arriveBy' in state[d]["semi"]:
                                    response.append(state[d]["semi"]['arriveBy'])
                                    break
                        elif 'leave' in ' '.join(response[-5:]) or 'leaving' in ' '.join(response[-5:]) or 'departure' in ' '.join(response[-3:]):
                            if active_domain == "train" and 'train' in top_result and 'leaveAt' in top_result["train"]:
                                # print('{} -> {}'.format(token, top_results[active_domain]['leaveAt']))
                                response.append(top_result["train"]['leaveAt'])
                                continue
                            for d in state:
                                if d == 'history':
                                    continue
                                if 'leaveAt' in state[d]["semi"]:
                                    response.append(state[d]["semi"]['leaveAt'])
                                    break
                        elif 'book' in response or "booked" in response:
                            if state['restaurant']['book']['time'] != "":
                                response.append(state['restaurant']['book']['time'])
                        else:
                            try:
                                for d in state:
                                    if d == 'history':
                                        continue
                                    for s in ['arriveBy', 'leaveAt']:
                                        if s in state[d]["semi"]:
                                            response.append(state[d]["semi"][s])
                                            raise ValueError(f"Found {s} in {d}")
                            except:
                                pass
                            else:
                                response.append(token)
                    elif slot == 'price':
                        if active_domain == 'attraction':
                            # .split()[0]
                            if "attraction" in top_result:
                                value = top_result['attraction']['entrance fee']
                            else:
                                value = "?"
                            if "?" in value:
                                value = "unknown"
                            # if "?" not in value:
                            #    try:
                            #        value = str(int(value))
                            #    except:
                            #        value = 'free'
                            # else:
                            #    value = "unknown"
                            response.append(value)
                        elif active_domain == "train":
                            if "train" in top_result and "price" in top_result["train"]:
                                response.append(top_result["train"][slot].split()[0])
                            else:
                                response.append(token)
                    elif slot == "day" and active_domain in ["restaurant", "hotel"]:
                        if state[active_domain]['book']['day'] != "":
                            response.append(state[active_domain]['book']['day'])

                    else:
                        # slot-filling based on query results
                        for d in top_result:
                            if slot in top_result[d]:
                                response.append(top_result[d][slot])
                                break
                        else:
                            # slot-filling based on belief state
                            for d in state:
                                if d == 'history':
                                    continue
                                if slot in state[d]["semi"]:
                                    response.append(state[d]["semi"][slot])
                                    break
                                if slot in state[d]["book"]:
                                    response.append(state[d]["book"][slot])
                                    break
                            else:
                                response.append(token)
                else:
                    if domain == 'hospital':
                        if slot == 'name':
                            response.append('The hospital')
                        else:
                            response.append(top_result['hospital'][slot])
                    elif domain == 'police':
                        response.append(top_result['police'][slot])
                    elif domain == 'taxi':
                        if slot == 'phone':
                            response.append(top_result['taxi']['taxi_phone'])
                        elif slot == 'color':
                            response.append(top_result['taxi']['taxi_colors'])
                        elif slot == 'type':
                            response.append(top_result['taxi']['taxi_types'])
                    else:
                        # print(token)
                        response.append(token)
                if response:
                    response[-1] += punct
            else:
                if token == "pounds" and ("pounds" in response[-1] or "unknown" in response[-1] or "free" in response[-1]):
                    pass
                else:
                    response.append(token)

        response = ' '.join(response)
        response = response.replace(' -s', 's')
        response = response.replace(' -ly', 'ly')
        response = response.replace(' .', '.')
        response = response.replace(' ?', '?')

        # if "not mentioned" in response:
        #    pdb.set_trace()
        # print("lexicalized: ", response)

        return response

    def lexicalize_multioptions(self, template, top_results, num_results, state_with_history):
        top_results['hospital'] = self.db.query('hospital', [])
        top_results['police'] = self.db.query('police', [])
        top_results['taxi'] = self.db.query('taxi', [])
        
        response = self.populate_template_multioptions(template, top_results, num_results, state_with_history)

        # check some common lexicalization errors
        if any([p in response for p in ["not mentioned", "dontcare"]]):
            logger.warning(f"Detected lexicalization error in {response}")

        return response

    def populate_template_multioptions(self, template, top_results, num_results, state):
        # print("template:",template)
        # print("top_results:",top_results)
        active_domain = None if len(top_results.keys()) == 0 else list(top_results)[0]
        num_results_str = str(num_results.get(active_domain, 0))
        # if active_domain != "train":
        #    pdb.set_trace()

        template = template.replace('book [value_count] of', 'book one of')
        tokens = template.split()
        response = []
        result_idx = 0
        for index, token in enumerate(tokens):
            if token.startswith('[') \
                and any([token.endswith(']'),
                         token.endswith('].'), token.endswith('],'),
                         token.endswith(']!'), token.endswith(']?')]):

                if "first" in tokens[index - 4:index]:
                    result_idx = 0
                elif "last" in tokens[index - 4:index] or "latest" in tokens[index-4:index]:
                    # pdb.set_trace()
                    result_idx = -1
                # this token has appeared before
                elif "name" in token and tokens[:index+1].count(token) > 1:
                    result_idx += 1

                if len(token.split('_')) > 2:
                    # Probably a series of special tokens, e.g., "[value_count],[value_count]"
                    # find the first "]" and split there
                    token = token[:token.find(']')] + ']'

                domain, slot = re.sub(r'[\[\]\.,!\?]', '', token).split('_')
                punct = '' if token.endswith(']') else token[-1]
                token = token if token.endswith(']') else token[:-1]

                if domain == 'train' and slot == 'id':
                    slot = 'trainID'
                elif active_domain != 'train' and slot == 'price':
                    slot = 'pricerange'
                elif slot == 'reference':
                    slot = 'Ref'

                if domain in top_results and len(top_results[domain]) > result_idx and slot in top_results[domain][result_idx]:
                    # print('{} -> {}'.format(token, top_results[domain][slot]))
                    response.append(top_results[domain][result_idx][slot])
                elif domain == 'value':
                    if slot == 'count':
                        if "there are" in " ".join(tokens[index-2:index]) or "i have" in " ".join(tokens[index-2:index]):
                            response.append(num_results_str)
                        # the first [value_count], the last [value_count]
                        elif "the" in tokens[index-2] or "which" in tokens[index-1]:
                            response.append("one") in top_results
                        elif active_domain == "train":
                            if index+1 < len(tokens) and "minute" in tokens[index+1]:
                                response.append(top_results['train'][result_idx]['duration'].split()[0])
                        elif active_domain == "hotel":
                            if index+1 < len(tokens):
                                if "star" in tokens[index+1]:
                                    response.append(top_results['hotel'][result_idx]['stars'])
                                # elif "nights" in tokens[index+1]:
                                #    response.append(state[active_domain]["book"]["stay"])
                                # elif "people" in tokens[index+1]:
                                #    response.append(state[active_domain]["book"]["people"])
                        elif active_domain == "attraction":
                            if "pounds" in tokens[index+1] and "entrance fee" in " ".join(tokens[index-3:index]):
                                value = top_results[active_domain][result_idx]['entrance fee'].split()[0]
                                if "?" in value:
                                    value = "unknown"
                                # if "?" not in value:
                                #     try:
                                #         value = str(int(value))
                                #     except:
                                #         value = 'free'
                                # else:
                                #     value = "unknown"
                                response.append(value)
                        # if "there are" in " ".join(tokens[index-2:index]):
                            # response.append(str(num_results))
                        # elif "the" in tokens[index-2]: # the first [value_count], the last [value_count]
                            # response.append("1")
                        else:
                            response.append(num_results_str)
                    elif slot == 'place':
                        if 'arriv' in " ".join(tokens[index-2:index]) or "to" in " ".join(tokens[index-2:index]):
                            if active_domain == "train":
                                response.append(top_results["train"][result_idx]["destination"])
                            else:
                                response.append(state["taxi"]["semi"]["destination"])
                        elif 'leav' in " ".join(tokens[index-2:index]) or "from" in tokens[index-2:index] or "depart" in " ".join(tokens[index-2:index]):
                            if active_domain == "train":
                                response.append(top_results["train"][result_idx]["departure"])
                            else:
                                response.append(state["taxi"]["semi"]["departure"])
                        else:
                            # try:
                            #     for d in state:
                            #         if d == 'history':
                            #             continue
                            #         for s in ['destination', 'departure']:
                            #             if s in state[d]["semi"]:
                            #                 response.append(state[d]["semi"][s])
                            #                 raise
                            # except:
                            #     pass
                            # else:
                            #     response.append(token)
                            response.append("Cambridge")
                    elif slot == 'time':
                        if 'arriv' in ' '.join(response[-7:]) or 'arriving' in ' '.join(response[-7:]):
                            if active_domain is not None and 'arriveBy' in top_results[active_domain][result_idx]:
                                # print('{} -> {}'.format(token, top_results[active_domain]['arriveBy']))
                                response.append(top_results[active_domain][result_idx]['arriveBy'])
                                continue
                            for d in state:
                                if d == 'history':
                                    continue
                                if 'arriveBy' in state[d]["semi"]:
                                    response.append(state[d]["semi"]['arriveBy'])
                                    break
                        elif 'leav' in ' '.join(response[-7:]) or 'depart' in ' '.join(response[-7:]):
                            if active_domain is not None and 'leaveAt' in top_results[active_domain][result_idx]:
                                # print('{} -> {}'.format(token, top_results[active_domain]['leaveAt']))
                                response.append(top_results[active_domain][result_idx]['leaveAt'])
                                continue
                            for d in state:
                                if d == 'history':
                                    continue
                                if 'leaveAt' in state[d]["semi"]:
                                    response.append(state[d]["semi"]['leaveAt'])
                                    break
                        elif 'book' in response or "booked" in response:
                            if state['restaurant']['book']['time'] != "":
                                response.append(state['restaurant']['book']['time'])
                        else:
                            try:
                                for d in state:
                                    if d == 'history':
                                        continue
                                    for s in ['arriveBy', 'leaveAt']:
                                        if s in state[d]["semi"]:
                                            response.append(state[d]["semi"][s])
                                            raise ValueError(f"Found {s} in {d}")
                            except:
                                pass
                            else:
                                response.append(token)
                    elif slot == 'price':
                        if active_domain == 'attraction':
                            value = top_results['attraction'][result_idx]['entrance fee'].split()[0]
                            if "?" in value:
                                value = "unknown"
                            # if "?" not in value:
                            #     try:
                            #         value = str(int(value))
                            #     except:
                            #         value = 'free'
                            # else:
                            #     value = "unknown"
                            response.append(value)
                        elif active_domain == "train":
                            response.append(top_results[active_domain][result_idx][slot].split()[0])
                    elif slot == "day" and active_domain in ["restaurant", "hotel"]:
                        if state[active_domain]['book']['day'] != "":
                            response.append(state[active_domain]['book']['day'])

                    else:
                        # slot-filling based on query results
                        for d, entities in top_results.items():
                            if result_idx < len(entities) and slot in entities[result_idx]:
                                response.append(entities[result_idx][slot])
                                break
                        else:
                            # slot-filling based on belief state
                            for d in state:
                                if d == 'history':
                                    continue
                                if slot in state[d]["semi"]:
                                    response.append(state[d]["semi"][slot])
                                    raise ValueError(f"Found {s} in {d}")
                            else:
                                response.append(token)
                else:
                    if domain == 'hospital':
                        if slot == 'name':
                            response.append('The hospital')
                        else:
                            response.append(top_results['hospital'][result_idx][slot])
                    elif domain == 'police':
                        response.append(top_results['police'][result_idx][slot])
                    elif domain == 'taxi':
                        if slot == 'phone':
                            response.append(top_results['taxi'][0]['taxi_phone'])
                        elif slot == 'color':
                            response.append(top_results['taxi'][0]['taxi_colors'])
                        elif slot == 'type':
                            response.append(top_results['taxi'][0]['taxi_types'])
                    else:
                        # print(token)
                        response.append(token)
                response[-1] += punct
            else:
                response.append(token)
        
        response = ' '.join(response)

        response = response.replace(' -s', 's')
        response = response.replace(' -ly', 'ly')
        response = response.replace(' .', '.')
        response = response.replace(' ?', '?')
        # print(template, response)

        return response

    def populate_template_options(self, template, top_results, num_results, state):
        # print("template:",template)
        # print("top_results:",top_results)
        active_domain = None if len(
            top_results.keys()) == 0 else list(top_results.keys())[0]
        # if active_domain != "train":
        #    pdb.set_trace()

        template = template.replace('book [value_count] of', 'book one of')
        tokens = template.split()
        response = []
        result_idx = 0
        for index, token in enumerate(tokens):
            if token.startswith('[') and (token.endswith(']') or token.endswith('].') or token.endswith('],')):
                if "first" in tokens[index - 4:index]:
                    result_idx = 0
                elif "last" in tokens[index - 4:index] or "latest" in tokens[index-4:index]:
                    # pdb.set_trace()
                    result_idx = -1
                # this token has appeared before
                elif "name" in token and tokens[:index+1].count(token) > 1:
                    result_idx += 1
                domain = token[1:-1].split('_')[0]
                slot = token[1:-1].split('_')[1]
                if slot.endswith(']'):
                    slot = slot[:-1]
                if domain == 'train' and slot == 'id':
                    slot = 'trainID'
                elif active_domain != 'train' and slot == 'price':
                    slot = 'pricerange'
                elif slot == 'reference':
                    slot = 'Ref'
                if domain in top_results and len(top_results[domain]) > 0 and slot in top_results[domain][result_idx]:
                    # print('{} -> {}'.format(token, top_results[domain][slot]))
                    response.append(top_results[domain][result_idx][slot])
                elif domain == 'value':
                    if slot == 'count':
                        if "there are" in " ".join(tokens[index-2:index]) or "i have" in " ".join(tokens[index-2:index]):
                            response.append(str(num_results))
                        # the first [value_count], the last [value_count]
                        elif "the" in tokens[index-2] or "which" in tokens[index-1]:
                            response.append("one")
                        elif active_domain == "train":
                            if index+1 < len(tokens) and "minute" in tokens[index+1]:
                                response.append(top_results['train'][result_idx]['duration'].split()[0])
                        elif active_domain == "hotel":
                            if index+1 < len(tokens):
                                if "star" in tokens[index+1]:
                                    response.append(top_results['hotel'][result_idx]['stars'])
                                # elif "nights" in tokens[index+1]:
                                #    response.append(state[active_domain]["book"]["stay"])
                                # elif "people" in tokens[index+1]:
                                #    response.append(state[active_domain]["book"]["people"])
                        elif active_domain == "attraction":
                            if "pounds" in tokens[index+1] and "entrance fee" in " ".join(tokens[index-3:index]):
                                value = top_results[active_domain][result_idx]['entrance fee'].split()[0]
                                if "?" not in value:
                                    try:
                                        value = str(int(value))
                                    except:
                                        value = 'free'
                                else:
                                    value = "unknown"
                                response.append(value)
                        # if "there are" in " ".join(tokens[index-2:index]):
                            # response.append(str(num_results))
                        # elif "the" in tokens[index-2]: # the first [value_count], the last [value_count]
                            # response.append("1")
                        else:
                            response.append(str(num_results))
                    elif slot == 'place':
                        if 'arriv' in " ".join(tokens[index-2:index]) or "to" in " ".join(tokens[index-2:index]):
                            response.append(
                                top_results[active_domain][result_idx]["destination"])
                        elif 'leav' in " ".join(tokens[index-2:index]) or "from" in tokens[index-2:index] in "depart" in " ".join(tokens[index-2:index]):
                            response.append(
                                top_results[active_domain][result_idx]["departure"])
                        else:
                            try:
                                for d in state:
                                    if d == 'history':
                                        continue
                                    for s in ['destination', 'departure']:
                                        if s in state[d]["semi"]:
                                            response.append(state[d]["semi"][s])
                                            raise
                            except:
                                pass
                            else:
                                response.append(token)
                    elif slot == 'time':
                        if 'arriv' in ' '.join(response[-7:]) or 'arriving' in ' '.join(response[-7:]):
                            if active_domain is not None and 'arriveBy' in top_results[active_domain][result_idx]:
                                # print('{} -> {}'.format(token, top_results[active_domain]['arriveBy']))
                                response.append(top_results[active_domain][result_idx]['arriveBy'])
                                continue
                            for d in state:
                                if d == 'history':
                                    continue
                                if 'arriveBy' in state[d]["semi"]:
                                    response.append(state[d]["semi"]['arriveBy'])
                                    break
                        elif 'leav' in ' '.join(response[-7:]) or 'depart' in ' '.join(response[-7:]):
                            if active_domain is not None and 'leaveAt' in top_results[active_domain][result_idx]:
                                # print('{} -> {}'.format(token, top_results[active_domain]['leaveAt']))
                                response.append(top_results[active_domain][result_idx]['leaveAt'])
                                continue
                            for d in state:
                                if d == 'history':
                                    continue
                                if 'leaveAt' in state[d]["semi"]:
                                    response.append(state[d]["semi"]['leaveAt'])
                                    break
                        elif 'book' in response or "booked" in response:
                            if state['restaurant']['book']['time'] != "":
                                response.append(state['restaurant']['book']['time'])
                        else:
                            try:
                                for d in state:
                                    if d == 'history':
                                        continue
                                    for s in ['arriveBy', 'leaveAt']:
                                        if s in state[d]:
                                            response.append(state[d][s])
                                            raise
                            except:
                                pass
                            else:
                                response.append(token)
                    elif slot == 'price':
                        if active_domain == 'attraction':
                            value = top_results['attraction'][result_idx]['entrance fee'].split()[
                                0]
                            if "?" not in value:
                                try:
                                    value = str(int(value))
                                except:
                                    value = 'free'
                            else:
                                value = "unknown"
                            response.append(value)
                        elif active_domain == "train":
                            response.append(top_results[active_domain][result_idx][slot].split()[0])
                    elif slot == "day" and active_domain in ["restaurant", "hotel"]:
                        if state[active_domain]['book']['day'] != "":
                            response.append(state[active_domain]['book']['day'])

                    else:
                        # slot-filling based on query results
                        for d in top_results:
                            if slot in top_results[d][result_idx]:
                                response.append(
                                    top_results[d][result_idx][slot])
                                break
                        else:
                            # slot-filling based on belief state
                            for d in state:
                                if d == 'history':
                                    continue
                                if slot in state[d]:
                                    response.append(state[d][slot])
                                    break
                            else:
                                response.append(token)
                else:
                    if domain == 'hospital':
                        if slot == 'phone':
                            response.append('01223216297')
                        elif slot == 'department':
                            response.append('neurosciences critical care unit')
                        elif slot == 'address':
                            response.append("56 Lincoln street")
                        elif slot == "postcode":
                            response.append('533421')
                    elif domain == 'police':
                        if slot == 'phone':
                            response.append('01223358966')
                        elif slot == 'name':
                            response.append('Parkside Police Station')
                        elif slot == 'address':
                            response.append('Parkside, Cambridge')
                        elif slot == 'postcode':
                            response.append('533420')
                    elif domain == 'taxi':
                        if slot == 'phone':
                            response.append('01223358966')
                        elif slot == 'color':
                            response.append('white')
                        elif slot == 'type':
                            response.append('toyota')
                    else:
                        # print(token)
                        response.append(token)
            else:
                response.append(token)

        try:
            response = ' '.join(response)
        except Exception as e:
            # pprint(response)
            raise
        response = response.replace(' -s', 's')
        response = response.replace(' -ly', 'ly')
        response = response.replace(' .', '.')
        response = response.replace(' ?', '?')
        # print(template, response)

        return response

    def model_predict(self, data_feed):
        self.logprobs = []
        logprobs, pred_labels, joint_logpz, sample_y = self.model.forward_rl(
            data_feed, self.model.config.max_dec_len,
            **self.temperature_params
        )

        self.logprobs.extend(joint_logpz)

        pred_labels = np.array(
            [pred_labels], dtype=int)
        de_tknize = get_detokenize()
        pred_str = get_sent(self.model.vocab, de_tknize, pred_labels, 0)

        return pred_str
 
def get_sent(vocab, de_tknize, data, b_id, stop_eos=True, stop_pad=True):
    ws = []
    for t_id in range(data.shape[1]):
        w = vocab[data[b_id, t_id]]
        if (stop_eos and w == EOS) or (stop_pad and w == PAD):
            break
        if w != PAD:
            ws.append(w)

    return de_tknize(ws)

def pad_to(max_len, tokens, do_pad):
    if len(tokens) >= max_len:
        return tokens[: max_len-1] + [tokens[-1]]
    elif do_pad:
        return tokens + [0] * (max_len - len(tokens))
    else:
        return tokens

def prepare_batch_gen(rows, config, pad_context=True):
    ctx_utts, ctx_lens = [], []

    out_bs, out_db = [], []

    for row in rows:
        in_row, out_row = row['context'], row['response']
        # source context
        batch_ctx = []
        for turn in in_row:
            batch_ctx.append(
                pad_to(config.max_utt_len, turn['utt'], do_pad=pad_context))
        ctx_utts.append(batch_ctx)
        ctx_lens.append(len(batch_ctx))

        out_bs.append(out_row['bs'])
        out_db.append(out_row['db'])

    batch_size = len(ctx_lens)
    vec_ctx_lens = np.array(ctx_lens)  # (batch_size, ), number of turns
    max_ctx_len = np.max(vec_ctx_lens)
    if pad_context:
        vec_ctx_utts = np.zeros(
            (batch_size, max_ctx_len, config.max_utt_len), dtype=np.int32)
    else:
        vec_ctx_utts = []
    vec_out_bs = np.array(out_bs)  # (batch_size, 94)
    vec_out_db = np.array(out_db)  # (batch_size, 30)

    for b_id in range(batch_size):
        if pad_context:
            vec_ctx_utts[b_id, :vec_ctx_lens[b_id], :] = ctx_utts[b_id]
        else:
            vec_ctx_utts.append(ctx_utts[b_id])


    return Pack(context_lens=vec_ctx_lens,  # (batch_size, )
                # (batch_size, max_ctx_len, max_utt_len)
                contexts=vec_ctx_utts,
                bs=vec_out_bs,  # (batch_size, 94)
                db=vec_out_db  # (batch_size, 30)
                )


if __name__ == '__main__':

    domain_name = 'object_division'
    domain_info = domain.get_domain(domain_name)

    train_data_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'data/norm-multi-woz/train_dials.json')

    config = Pack(
        seed=10,
        train_path=train_data_path,
        max_vocab_size=1000,
        last_n_model=5,
        max_utt_len=50,
        max_dec_len=50,
        backward_size=2,
        batch_size=1,
        use_gpu=True,
        op='adam',
        init_lr=0.001,
        l2_norm=1e-05,
        momentum=0.0,
        grad_clip=5.0,
        dropout=0.5,
        max_epoch=100,
        embed_size=100,
        num_layers=1,
        utt_rnn_cell='gru',
        utt_cell_size=300,
        bi_utt_cell=True,
        enc_use_attn=True,
        dec_use_attn=True,
        dec_rnn_cell='lstm',
        dec_cell_size=300,
        dec_attn_mode='cat',
        y_size=10,
        k_size=20,
        beta=0.001,
        simple_posterior=True,
        contextual_posterior=True,
        use_mi=False,
        use_pr=True,
        use_diversity=False,
        #
        beam_size=20,
        fix_batch=True,
        fix_train_batch=False,
        avg_type='word',
        print_step=300,
        ckpt_step=1416,
        improve_threshold=0.996,
        patient_increase=2.0,
        save_model=True,
        early_stop=False,
        gen_type='greedy',
        preview_batch_num=None,
        k=domain_info.input_length(),
        init_range=0.1,
        pretrain_folder='2019-09-20-21-43-06-sl_cat',
        forward_only=False
    )

    state = {'user_action': [["Inform", "Hotel", "Area", "east"], ["Inform", "Hotel", "Stars", "4"]],
             'system_action': [],
             'belief_state': {'police': {'book': {'booked': []}, 'semi': {}},
                              'hotel': {'book': {'booked': [], 'people': '', 'day': '', 'stay': ''},
                                        'semi': {'name': '',
                                                 'area': 'east',
                                                 'parking': '',
                                                 'pricerange': '',
                                                 'stars': '4',
                                                 'internet': '',
                                                 'type': ''}},
                              'attraction': {'book': {'booked': []},
                                             'semi': {'type': '', 'name': '', 'area': ''}},
                              'restaurant': {'book': {'booked': [], 'people': '', 'day': '', 'time': ''},
                                             'semi': {'food': '', 'pricerange': '', 'name': '', 'area': ''}},
                              'hospital': {'book': {'booked': []}, 'semi': {'department': ''}},
                              'taxi': {'book': {'booked': []},
                                       'semi': {'leaveAt': '',
                                                'destination': '',
                                                'departure': '',
                                                'arriveBy': ''}},
                              'train': {'book': {'booked': [], 'people': ''},
                                        'semi': {'leaveAt': '',
                                                 'destination': '',
                                                 'day': '',
                                                 'arriveBy': '',
                                                 'departure': ''}}},
             'request_state': {},
             'terminated': False,
             'history': [['sys', ''],
                         ['user', 'Could you book a 4 stars hotel east of town for one night, 1 person?']]}

    model_file="path/to/model" # points to model from lava repo
    cur_model = LAVA(model_file)

    response = cur_model.predict(state)
    # print(response)
