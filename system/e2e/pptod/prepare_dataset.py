import os
import sys
import json
import argparse

root_dpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
sys.path.append(root_dpath)

from dataset.multiwoz23 import MultiWOZ23Dataset
from system.e2e.pptod.pptod import (
    PPTODSpecialTokens,
    PPTODPrefixes,
    clean_utterance,
    context_tuples2str,
    belief_state_dict2str,
    db_result_dict2str
)

def main(args):
    mwoz23 = MultiWOZ23Dataset()
    processed_data = {}
    for split in mwoz23.split_keys:
        dicts = []
        for dialogue_id in mwoz23.iter_dialogue(split=split, progress_bar=True):
            for turn in mwoz23.iter_system_turn(dialogue_id=dialogue_id):

                context_str = context_tuples2str(turn.context, args.max_context_turns)
                belief_state_str = belief_state_dict2str(turn.belief_state)
                db_results_str = db_result_dict2str(turn.db_results)
                response_str = " ".join([
                    PPTODSpecialTokens.system_response_sos,
                    clean_utterance(turn.delexicalized_system_response),
                    PPTODSpecialTokens.system_response_eos
                ])

                # Belief state tracking
                dicts.append({
                    "dialogue_id": turn.dialogue_id,
                    "turn_id": turn.turn_id,
                    "input": f"{PPTODPrefixes.dst} {context_str}",
                    "target": belief_state_str
                })

                # Response generation
                dicts.append({
                    "dialogue_id": turn.dialogue_id,
                    "turn_id": turn.turn_id,
                    "input": f"{PPTODPrefixes.rg} {context_str} {db_results_str}",
                    "target": response_str
                })

        # Convert dicts to jsonlines
        processed_data[split] = [json.dumps(d)+'\n' for d in dicts]

    # Save processed data
    os.makedirs(args.output_dir, exist_ok=True)
    for split, jsonlines in processed_data.items():
        with open(os.path.join(args.output_dir, f"{split}.json"), "w") as f:
            f.writelines(jsonlines)

    # Save delexicalization vocabulary
    with open(os.path.join(args.output_dir, "delex_vocab.json"), "w") as f:
        json.dump(mwoz23.delex_vocab, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory to save processed data.")
    parser.add_argument("--max_context_turns", type=int, default=0,
                        help="Maximum number of context turns to keep.")
    args = parser.parse_args()

    main(args)
