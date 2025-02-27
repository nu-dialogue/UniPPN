import numpy as np
from fuzzywuzzy import fuzz
from convlab2.util.multiwoz.state import default_state
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator, belief_domains
from convlab2.util.multiwoz.multiwoz_slot_trans import REF_USR_DA, REF_SYS_DA

NONE_VALUE = ""

def get_default_belief_state():
    belief_state = {}
    for domain, domain_bs in default_state()["belief_state"].items():
        belief_state[domain] = {}
        for slot_values in domain_bs.values():
            for slot in slot_values:
                if slot == "booked":
                    continue
                belief_state[domain][slot] = [] # Available multi values for fail constraint
    return belief_state

def get_default_request_state():
    return {domain: [] for domain in default_state()["belief_state"]}

def convert_slot_name_from_usr_to_sys(domain, usr_slot):
    base_slot = REF_USR_DA[domain.capitalize()][usr_slot]

    if base_slot not in REF_SYS_DA[domain.capitalize()]:
        domain = "booking"
    sys_slot = REF_SYS_DA[domain.capitalize()][base_slot]

    return sys_slot

def normalize_value(value):
    value = value.strip().lower()
    if value in ["dontcare"]:
        value = NONE_VALUE
    return value

def div(a, b, except_value=0):
    return a / b if b != 0 else except_value

def flatten_da(da, include_value=True):
    flat_da = []
    for act in da:
        if include_value:
            flat_da.append( "-".join(act).lower() )
        else:
            flat_da.append( "-".join(act[:-1]).lower() )
    return flat_da

def is_equal_value(true_value, pred_value, slot_name=None, fuzzy_match_ratio=60):
    """
    Check if true_value and pred_value are equal.
    Reference: convlab2.util.multiwoz.dbquery.Database.query()
    """
    true_value = true_value.strip().lower()
    pred_value = pred_value.strip().lower()

    if slot_name in ["Addr"]:
        return fuzz.partial_ratio(true_value, pred_value) >= fuzzy_match_ratio
    else:
        return true_value == pred_value

def calc_da_accuracy(true_da, pred_da):
    if pred_da is None or not isinstance(true_da, list):
        # if system does not generate any da, return nan
        return {
            "tp_da": [], "fn_da": [], "fp_da": [],
            "recall": None, "precision": None, "f1": None, "acc": None
        }

    def convert_da_to_dict(da):
        da_dict = {}
        for intent, domain, slot, value in da:
            da_dict["-".join([intent, domain, slot])] = value
        return da_dict
    def convert_dict_to_da(da_dict):
        da = []
        for act, value in da_dict.items():
            intent, domain, slot = act.split("-", 2)
            da.append([intent, domain, slot, value])
        return da
    
    true_da = convert_da_to_dict(true_da)
    pred_da = convert_da_to_dict(pred_da)

    tp_da, fn_da, fp_da = {}, {}, {}
    for act in true_da:
        if act in pred_da and is_equal_value(true_da[act], pred_da[act], slot_name=act.split("-")[2]):
            tp_da[act] = true_da[act]
        else:
            fn_da[act] = true_da[act]
    for act in pred_da:
        if act not in true_da or not is_equal_value(true_da[act], pred_da[act], slot_name=act.split("-")[2]):
            fp_da[act] = pred_da[act]

    tp_da = convert_dict_to_da(tp_da)
    fn_da = convert_dict_to_da(fn_da)
    fp_da = convert_dict_to_da(fp_da)

    tp = len(tp_da)
    fn = len(fn_da)
    fp = len(fp_da)

    acc, recall, precision, f1 = 0, 0, 0, 0
    if tp + fn:
        recall = tp / (tp + fn)
    if tp + fp:
        precision = tp / (tp + fp)
    if recall + precision:
        f1 = (2*recall*precision) / (recall+precision)
    if tp + fn + fp:
        acc = tp / (tp + fn + fp)

    result = {"tp_da": tp_da, "fn_da": fn_da, "fp_da": fp_da, "recall":recall, "precision":precision, "f1":f1, "acc": acc}
    return result

class Evaluator(MultiWozEvaluator):
    def __init__(self, max_turn):
        super().__init__()
        self.max_turn = max_turn
        
    def add_goal(self, goal):
        self.belief_state_goal = get_default_belief_state()
        self.request_state_goal = get_default_request_state()
        for domain, domain_goal in goal.items():
            for task, slot_values in domain_goal.items():
                if task in ["info", "book", "fail_info", "fail_book"]:
                    for slot, value in slot_values.items():
                        slot = convert_slot_name_from_usr_to_sys(domain, slot)
                        value = normalize_value(value)
                        self.belief_state_goal[domain][slot].append(value)

                elif task in ["reqt"]:
                    for slot, _ in slot_values.items():
                        slot = convert_slot_name_from_usr_to_sys(domain, slot)
                        self.request_state_goal[domain].append(slot)
        
        for domain, domain_goal in self.belief_state_goal.items():
            for slot, values in domain_goal.items():
                if not values:
                    values.append(NONE_VALUE)

        return super().add_goal(goal)

    def get_reward_v2(self, user_agent):
        reward = self.get_reward()
        session_over = user_agent.is_terminated()
        if reward == 40 and not session_over:
            # don't give full reward if session is not over
            reward = 5
        return reward

    def _domain_eval(self, domain):
        result = {
            "inform_precision": None,
            "inform_recall": None,
            "inform_f1": None,
            "book_rate": None,
            "goal_match_rate": None,
            "task_success": None
        }
        if domain not in self.goal:
            return result
        
        goal = {domain: self._expand(self.goal)[domain]}

        book_rate = self._book_rate_goal(goal, self.booked, [domain])
        book_rate = np.mean(book_rate) if book_rate else None

        inf_tp, inf_fp, inf_fn, *_ = self._inform_F1_goal(goal, self.sys_da_array, [domain])
        inf_prec = div(inf_tp, inf_tp + inf_fp, except_value=None)
        inf_rec = div(inf_tp, inf_tp + inf_fn, except_value=None)
        if None not in [inf_prec, inf_rec]:
            inf_f1 = div(2 * inf_prec * inf_rec, inf_prec + inf_rec, except_value=None)
        else:
            inf_f1 = None

        matched, _ = self._final_goal_analyze()
        if domain in matched:
            goal_match_rate = 1
        else:
            goal_match_rate = 0

        task_success = any([
            book_rate == 1 and inf_rec == 1,
            book_rate == 1 and inf_prec is None,
            book_rate is None and inf_rec == 1,
        ]) and goal_match_rate == 1

        result = {
            "inform_precision": inf_prec,
            "inform_recall": inf_rec,
            "inform_f1": inf_f1,
            "book_rate": book_rate,
            "goal_match_rate": goal_match_rate,
            "task_success": task_success
        }
        return result

    def domain_eval(self):
        results = {
            domain: self._domain_eval(domain)
            for domain in belief_domains
        }
        return results

    def evaluate_from_user(self, user_agent):
        session_over = user_agent.is_terminated()
        prec, rec, f1 = self.inform_F1()
        book_rate = self.book_rate()
        task_success = self.task_success()
        goal_match_rate = self.final_goal_analyze()

        eval_result = {
            "session_over": session_over,
            "inform_precision": prec,
            "inform_recall": rec,
            "inform_f1": f1,
            "book_rate": book_rate,
            "goal_match_rate": goal_match_rate,
            "task_success": task_success
        }

        return eval_result
    
    def evaluate_from_system(self, sys_agent):
        belief_state_num_slots = 0
        belief_state_num_correct = 0
        current_dialogue_state = sys_agent.get_dialogue_state()
        for domain, domain_bs in self.belief_state_goal.items():
            for slot, values in domain_bs.items():
                if not any(values): continue # TODO: Need to verify this is ok for accuracy
                belief_state_num_slots += 1
                for task in ["semi", "book"]:
                    if slot in current_dialogue_state["belief_state"][domain][task]:
                        pred_value = normalize_value(current_dialogue_state["belief_state"][domain][task][slot])
                        break
                belief_state_num_correct += int(pred_value in values)

        request_state_num_slots = 0
        request_state_num_correct = 0
        for domain, domain_rs in self.request_state_goal.items():
            for slot in domain_rs:
                request_state_num_slots += 1
                if domain in current_dialogue_state["request_state"]:
                    request_state_num_correct += int(slot in current_dialogue_state["request_state"][domain])
    
        eval_result = {
            "belief_state": {
                "num_slots": belief_state_num_slots,
                "num_correct": belief_state_num_correct,
                "accuracy": div(belief_state_num_correct, belief_state_num_slots)
            },
            "request_state": {
                "num_slots": request_state_num_slots,
                "num_correct": request_state_num_correct,
                "accuracy": div(request_state_num_correct, request_state_num_slots)
            }
        }

        return eval_result
    
    def evaluate_da_accuracy(self, user_agent, sys_agent):
        user_turn_dicts = user_agent.get_turn_dicts()
        sys_turn_dicts = sys_agent.get_turn_dicts()
        assert len(user_turn_dicts) == len(sys_turn_dicts), "The number of turns between user and system are different."

        results = []
        num_turns = len(user_turn_dicts)
        for i in range(num_turns):
            result = {}
            true_user_da = user_turn_dicts[i]["policy"]["user_action"]
            if sys_turn_dicts[i]["ppn_nlu"]["user_action"] is not None:
                pred_user_da = sys_turn_dicts[i]["ppn_nlu"]["user_action"]
            else:
                pred_user_da = sys_turn_dicts[i]["nlu"]["user_action"]
            result["user_da"] = calc_da_accuracy(true_da=true_user_da, pred_da=pred_user_da)
            
            if i < num_turns - 1: # The last system response is not observed by the user
                if sys_turn_dicts[i]["ppn_policy"]["system_action"] is not None:
                    true_sys_da = sys_turn_dicts[i]["ppn_policy"]["system_action"]
                else:
                    true_sys_da = sys_turn_dicts[i]["policy"]["system_action"]
                pred_sys_da = user_turn_dicts[i+1]["nlu"]["system_action"]
                result["system_da"] = calc_da_accuracy(true_da=true_sys_da, pred_da=pred_sys_da)
                
            results.append(result)

        return results

    def evaluate_from_reqt_goal(self, sys_agent):
        final_reqt_goal = {}
        for domain, domain_goal in self.goal.items():
            if "reqt" not in domain_goal:
                continue
            final_reqt_goal[domain] = {}
            for slot, value in domain_goal["reqt"].items():
                slot = convert_slot_name_from_usr_to_sys(domain, slot)
                value = normalize_value(value)
                final_reqt_goal[domain][slot] = {"value": value, "informed": False}
        final_reqt_goal["booking"] = []
        for domain, booked_entity in self.booked.items():
            if booked_entity and domain in ["hotel", "restaurant", "train"]:
                final_reqt_goal["booking"].append(booked_entity["Ref"])
            
        results = []
        sys_turn_dicts = sys_agent.get_turn_dicts()
        for turn in sys_turn_dicts[::-1]:
            sys_da = turn["policy"]["system_action"]
            slots = []
            correct_slots = []
            for intent, domain, slot, value in sys_da:
                if domain in ["general"] or slot in ["Choice"]:
                    continue
                intent = intent.lower()
                domain = domain.lower()
                if slot == "Ref":
                    slots.append(f"{domain}-{slot}-{value}")
                    if value in final_reqt_goal["booking"]:
                        correct_slots.append(f"{domain}-{slot}-{value}")
                        final_reqt_goal["booking"].remove(value)
                else:
                    slot = REF_SYS_DA[domain.capitalize()][slot]
                    value = normalize_value(value)
                    if intent in ["inform", "recommend", "book", "offerbook", "offerbooked"] and \
                        domain in final_reqt_goal and \
                            slot in final_reqt_goal[domain]:
                        slots.append(f"{domain}-{slot}-{value}")
                        if not final_reqt_goal[domain][slot]["informed"] and \
                            is_equal_value(value, final_reqt_goal[domain][slot]["value"]):
                            correct_slots.append(f"{domain}-{slot}-{value}")
                            final_reqt_goal[domain][slot]["informed"] = True
            results.append({
                "slots": slots,
                "correct_slots": correct_slots,
                "accuracy": len(correct_slots) / len(slots) if slots else None
            })
        results = results[::-1]
        return results