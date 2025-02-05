import os
import sys
import argparse
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Union, Any, Tuple, Optional, Dict

import torch
from sentence_transformers import SentenceTransformer

from convlab2.util.multiwoz.state import default_state

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
sys.path.append(root_dir)

from dataset.multiwoz23 import MultiWOZ23Dataset
from system.e2e.llmicl.prompts import (
    Example,
    context_tupes2str,
    PromptFormatter
)

def extract_belief_state_diff(
        prev_belief_state: Dict[str, Dict[str, str]],
        belief_state: Dict[str, Dict[str, str]]
    ) -> Dict[str, Dict[str, str]]:
    exclude_domains = ["bus"]
    exclude_slots = ["booked"]
    non_values = ["", "not mentioned"]

    belief_state_diff = {}
    for domain, constraints in belief_state.items():
        if domain in exclude_domains:
            continue
        for constraint_type, constraints in constraints.items():
            for slot, value in constraints.items():
                if slot not in prev_belief_state[domain][constraint_type]:
                    raise ValueError(f"Slot {slot} not found in previous belief state.")
                if slot in exclude_slots:
                    continue
                if value in ["not mentioned", prev_belief_state[domain][constraint_type][slot]]:
                    continue
                if domain not in belief_state_diff:
                    belief_state_diff[domain] = {}
                if constraint_type not in belief_state_diff[domain]:
                    belief_state_diff[domain][constraint_type] = {}
                belief_state_diff[domain][constraint_type][slot] = value
    return belief_state_diff

class FewshotExampler:
    def __init__(self, model_name: str, device: torch.device, max_context_turns: int,
                 examples: Optional[List[Example]] = None, embeddings: Optional[torch.Tensor] = None):
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name,
            device=device
        )

        self.device = device
        self.max_context_turns = max_context_turns

        self.examples = examples
        self.embeddings = embeddings

    def encode(self, sentences: Union[str, List[str]], batch_size: int = 1) -> torch.Tensor:
        if isinstance(sentences, str):
            sentences = [sentences]
            show_progress_bar = False
        else:
            show_progress_bar = True

        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True
        )
        return embeddings

    def create_vector_db(self, examples: List[Example]) -> torch.Tensor:
        batch_context_str = []
        for example in examples:
            context_str = context_tupes2str(
                context_tuples=example.context,
                speaker_prefix_mapping=PromptFormatter.speaker_prefix_mapping,
                max_context_turns=self.max_context_turns
            )
            batch_context_str.append(context_str)

        embeddings = self.encode(batch_context_str, batch_size=128)

        self.examples = examples
        self.embeddings = embeddings

    def save(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save({
            "model_name": self.model_name,
            "max_context_turns": self.max_context_turns,
            "examples": self.examples,
            "embeddings": self.embeddings.cpu()
        }, output_path)

    @classmethod
    def load(cls, saved_path: str, device: torch.device):
        state = torch.load(saved_path)
        return cls(
            model_name=state["model_name"],
            device=device,
            max_context_turns=state["max_context_turns"],
            examples=state["examples"],
            embeddings=state["embeddings"].to(device)
        )

    def retrieve(self, context: List[Tuple[str, str]], num_examples: int, sampling_top_k: Optional[int]) -> List[Example]:
        context_str = context_tupes2str(
            context_tuples=context,
            speaker_prefix_mapping=PromptFormatter.speaker_prefix_mapping,
            max_context_turns=self.max_context_turns
        )
        query_embedding = self.encode(context_str)

        scores = torch.nn.functional.cosine_similarity(
            x1=query_embedding, x2=self.embeddings
        )

        sorted_indices = scores.argsort(descending=True)

        if sampling_top_k is not None:
            assert sampling_top_k <= len(sorted_indices), \
                (f"sampling_top_k ({sampling_top_k}) must be less than or equal to the "
                 f"number of examples ({len(sorted_indices)}).")
            sorted_indices = sorted_indices[:sampling_top_k][torch.randperm(sampling_top_k)]

        selected_indices = sorted_indices[:num_examples]

        return [self.examples[i] for i in selected_indices]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="vector_db/mwoz23-gte_base-ctx3-dpd20.pt",
                        help="Path to save the vector database.")
    parser.add_argument("--model_name", type=str, default="thenlper/gte-base",
                        help="Name of the sentence transformer model.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train",
                        help="Split of the dataset to use.")
    parser.add_argument('--context_turns', type=int, default=3,
                        help='Number of context turns to use.')
    parser.add_argument('--dialogues_per_domain', type=int, default=20,
                        help='Number of dialogues per domain to use.')

    args = parser.parse_args()

    # Load MultiWOZ dataset
    dataset = MultiWOZ23Dataset()

    # Initialize FewshotExampler
    fewshot_exampler = FewshotExampler(
        model_name=args.model_name,
        device=torch.device("cuda:0"),
        max_context_turns=args.context_turns
    )

    # Create vector database
    domain_counter = {domain: 0 for domain in dataset.available_domains}
    examples = []
    for dialogue_id in dataset.iter_dialogue(split=args.split):
        goal = dataset.get_goal(dialogue_id)
        if all([domain_counter[domain.capitalize()] >= args.dialogues_per_domain for domain in goal]):
            # Skip if all domains in the goal have enough dialogues
            continue
        for domain in goal:
            domain_counter[domain.capitalize()] += 1

        prev_belief_state = default_state()["belief_state"]
        for turn in dataset.iter_system_turn(dialog_id=dialogue_id):
            try:
                belief_state_diff = extract_belief_state_diff(
                    prev_belief_state=prev_belief_state,
                    belief_state=turn.belief_state
                )
                examples.append(Example(
                    dialogue_id=dialogue_id,
                    turn_id=turn.turn_id,
                    context=turn.context,
                    belief_state_diff=belief_state_diff,
                    belief_state=turn.belief_state,
                    db_results=turn.db_results,
                    delexicalized_system_response=turn.delexicalized_system_response
                ))
            except ValueError as e:
                print(f"Skipping turn {dialogue_id} {turn.turn_id}: {e}")

            prev_belief_state = turn.belief_state

        print(", ".join([f"{domain}: {cnt:02d}" for domain, cnt in domain_counter.items()]))
        if all([cnt >= args.dialogues_per_domain for cnt in domain_counter.values()]):
            # Break if all domains have enough dialogues
            break

    fewshot_exampler.create_vector_db(examples)
    fewshot_exampler.save(args.output_path)

    breakpoint()
