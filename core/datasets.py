"""
MCQ Dataset Loading

Loads MCQ datasets from HuggingFace for benchmarking.
Adapted from NNCONFIDENCE/hlcc_mcq_system.py.

Supported datasets: truthfulqa, arc-easy, arc-challenge, mmlu, hellaswag,
commonsenseqa, openbookqa, sciq, winogrande, medmcqa, boolq, mmlu-pro
"""

from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np


@dataclass
class MCQExample:
    """A single MCQ example."""
    question_id: str
    question: str
    choices: List[str]
    correct_answer: int  # Index of correct choice
    subject: str = "general"
    difficulty: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)


def load_mcq_dataset(
    name: str,
    split: str = None,
    max_examples: Optional[int] = None
) -> List[MCQExample]:
    """
    Load an MCQ dataset from HuggingFace.

    Args:
        name: Dataset name (e.g. "truthfulqa", "arc-easy")
        split: Dataset split override (default: uses appropriate split per dataset)
        max_examples: Maximum number of examples to load

    Returns:
        List of MCQExample objects
    """
    from datasets import load_dataset

    examples = []

    if name == "mmlu":
        dataset = load_dataset("cais/mmlu", "all")
        ds_split = split or "test"
        for i, item in enumerate(dataset[ds_split]):
            if max_examples and i >= max_examples:
                break
            examples.append(MCQExample(
                question_id=f"mmlu_{i}",
                question=item["question"],
                choices=item["choices"],
                correct_answer=item["answer"],
                subject=item.get("subject", "general")
            ))

    elif name == "arc-easy":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Easy")
        for i, item in enumerate(dataset["test"]):
            if max_examples and i >= max_examples:
                break
            labels = item["choices"]["label"]
            answer_idx = labels.index(item["answerKey"]) if item["answerKey"] in labels else 0
            examples.append(MCQExample(
                question_id=f"arc_easy_{i}",
                question=item["question"],
                choices=item["choices"]["text"],
                correct_answer=answer_idx,
                subject="science"
            ))

    elif name == "arc-challenge":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
        for i, item in enumerate(dataset["test"]):
            if max_examples and i >= max_examples:
                break
            labels = item["choices"]["label"]
            answer_idx = labels.index(item["answerKey"]) if item["answerKey"] in labels else 0
            examples.append(MCQExample(
                question_id=f"arc_challenge_{i}",
                question=item["question"],
                choices=item["choices"]["text"],
                correct_answer=answer_idx,
                subject="science",
                difficulty=0.7
            ))

    elif name == "truthfulqa":
        dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
        for i, item in enumerate(dataset["validation"]):
            if max_examples and i >= max_examples:
                break
            choices = item["mc1_targets"]["choices"]
            labels = item["mc1_targets"]["labels"]
            answer_idx = labels.index(1) if 1 in labels else 0
            examples.append(MCQExample(
                question_id=f"truthfulqa_{i}",
                question=item["question"],
                choices=choices,
                correct_answer=answer_idx,
                subject="truthfulness",
                difficulty=0.8
            ))

    elif name == "hellaswag":
        dataset = load_dataset("Rowan/hellaswag")
        for i, item in enumerate(dataset["validation"]):
            if max_examples and i >= max_examples:
                break
            examples.append(MCQExample(
                question_id=f"hellaswag_{i}",
                question=item["ctx"],
                choices=item["endings"],
                correct_answer=int(item["label"]),
                subject="commonsense",
                difficulty=0.7
            ))

    elif name == "commonsenseqa":
        dataset = load_dataset("tau/commonsense_qa", split="validation")
        for i, item in enumerate(dataset):
            if max_examples and i >= max_examples:
                break
            answer_idx = ["A", "B", "C", "D", "E"].index(item["answerKey"]) if item["answerKey"] in ["A", "B", "C", "D", "E"] else 0
            examples.append(MCQExample(
                question_id=f"commonsenseqa_{i}",
                question=item["question"],
                choices=item["choices"]["text"],
                correct_answer=answer_idx,
                subject="commonsense"
            ))

    elif name == "openbookqa":
        dataset = load_dataset("allenai/openbookqa", "main")
        for i, item in enumerate(dataset["test"]):
            if max_examples and i >= max_examples:
                break
            labels = item["choices"]["label"]
            answer_idx = labels.index(item["answerKey"]) if item["answerKey"] in labels else 0
            examples.append(MCQExample(
                question_id=f"openbookqa_{i}",
                question=item["question_stem"],
                choices=item["choices"]["text"],
                correct_answer=answer_idx,
                subject="science"
            ))

    elif name == "sciq":
        dataset = load_dataset("allenai/sciq")
        for i, item in enumerate(dataset["test"]):
            if max_examples and i >= max_examples:
                break
            choices = [item["correct_answer"], item["distractor1"],
                       item["distractor2"], item["distractor3"]]
            seed = hash(item["question"]) & 0xFFFFFFFF
            rng = np.random.RandomState(seed)
            perm = rng.permutation(4)
            shuffled = [choices[j] for j in perm]
            correct_idx = int(np.where(perm == 0)[0][0])
            examples.append(MCQExample(
                question_id=f"sciq_{i}",
                question=item["question"],
                choices=shuffled,
                correct_answer=correct_idx,
                subject="science"
            ))

    elif name == "winogrande":
        dataset = load_dataset("allenai/winogrande", "winogrande_xl")
        for i, item in enumerate(dataset["validation"]):
            if max_examples and i >= max_examples:
                break
            answer_idx = int(item["answer"]) - 1
            examples.append(MCQExample(
                question_id=f"winogrande_{i}",
                question=item["sentence"],
                choices=[item["option1"], item["option2"]],
                correct_answer=answer_idx,
                subject="commonsense"
            ))

    elif name == "medmcqa":
        dataset = load_dataset("openlifescienceai/medmcqa")
        for i, item in enumerate(dataset["validation"]):
            if max_examples and i >= max_examples:
                break
            examples.append(MCQExample(
                question_id=f"medmcqa_{i}",
                question=item["question"],
                choices=[item["opa"], item["opb"], item["opc"], item["opd"]],
                correct_answer=int(item["cop"]),
                subject=item.get("subject_name", "medical")
            ))

    elif name == "boolq":
        dataset = load_dataset("google/boolq")
        for i, item in enumerate(dataset["validation"]):
            if max_examples and i >= max_examples:
                break
            correct_answer = 0 if item["answer"] else 1
            examples.append(MCQExample(
                question_id=f"boolq_{i}",
                question=f"{item['passage'][:300]}\n\nQuestion: {item['question']}",
                choices=["True", "False"],
                correct_answer=correct_answer,
                subject="reading_comprehension"
            ))

    elif name == "mmlu-pro":
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")
        for i, item in enumerate(dataset["test"]):
            if max_examples and i >= max_examples:
                break
            choices = item["options"]
            answer_idx = ord(item["answer"]) - ord("A")
            examples.append(MCQExample(
                question_id=f"mmlu_pro_{i}",
                question=item["question"],
                choices=choices,
                correct_answer=answer_idx,
                subject=item.get("category", "general"),
                difficulty=0.8
            ))

    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: mmlu, arc-easy, "
                         "arc-challenge, truthfulqa, hellaswag, commonsenseqa, "
                         "openbookqa, sciq, winogrande, medmcqa, boolq, mmlu-pro")

    print(f"Loaded {len(examples)} examples from {name}")
    return examples


def list_datasets() -> list:
    """Return list of supported dataset names."""
    return [
        "mmlu", "arc-easy", "arc-challenge", "truthfulqa", "hellaswag",
        "commonsenseqa", "openbookqa", "sciq", "winogrande", "medmcqa",
        "boolq", "mmlu-pro",
    ]


if __name__ == "__main__":
    print("Supported datasets:")
    for ds in list_datasets():
        print(f"  {ds}")
