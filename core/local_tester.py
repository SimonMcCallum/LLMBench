"""
Local LLM Tester

Tests locally-loaded HuggingFace models on MCQ questions
with confidence extraction via logit analysis.

Adapted from NNCONFIDENCE/hlcc_mcq_system.py MCQEvaluator.
"""

import json
import re
from datetime import datetime
from typing import List, Optional

import numpy as np

from core.datasets import MCQExample
from core.scoring import HLCCScorer, BenchmarkResult


def _format_mcq_prompt(example: MCQExample, include_confidence: bool = False) -> str:
    """Format MCQ as a prompt for local models."""
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][:len(example.choices)]
    choices_text = "\n".join(f"{l}. {c}" for l, c in zip(letters, example.choices))

    if include_confidence:
        return f"""Question: {example.question}

{choices_text}

Please respond with your answer and confidence in JSON format:
{{"answer": "A", "confidence": 0.85, "reasoning": "brief explanation"}}

Where confidence is between 0.0 (guessing) and 1.0 (certain).
Response:"""
    else:
        return f"""Question: {example.question}

{choices_text}

Answer with just the letter (A, B, C, D, or E):"""


def evaluate_sequential(model, tokenizer, example: MCQExample, device: str = "cuda") -> dict:
    """
    Sequential evaluation: get answer from logits, estimate confidence from entropy.

    Returns dict with selected_answer, confidence, reasoning.
    """
    import torch
    import torch.nn.functional as F

    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][:len(example.choices)]
    prompt = _format_mcq_prompt(example, include_confidence=False)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=False)
        logits = outputs.logits[0, -1, :]

    # Get logits for each letter token
    choice_logits = []
    for letter in letters:
        for token_str in [letter, f" {letter}", letter.lower(), f" {letter.lower()}"]:
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            if token_ids:
                choice_logits.append(logits[token_ids[0]].item())
                break
        else:
            choice_logits.append(-float("inf"))

    choice_logits = torch.tensor(choice_logits)
    probs = F.softmax(choice_logits, dim=0)
    selected_answer = probs.argmax().item()

    # Entropy-based confidence
    entropy = -(probs * (probs + 1e-10).log()).sum()
    max_entropy = np.log(len(example.choices))
    confidence = float(1.0 - (entropy.item() / max_entropy))
    confidence = max(0.0, min(1.0, confidence))

    return {
        "selected_answer": selected_answer,
        "confidence": confidence,
        "reasoning": "Sequential logit-based evaluation",
    }


def evaluate_unified(model, tokenizer, example: MCQExample, device: str = "cuda") -> dict:
    """
    Unified evaluation: generate answer + confidence in single pass.

    Returns dict with selected_answer, confidence, reasoning.
    """
    import torch

    prompt = _format_mcq_prompt(example, include_confidence=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][:len(example.choices)]
    selected_answer = 0
    confidence = 0.5
    reasoning = response_text

    try:
        json_match = re.search(r"\{[^}]+\}", response_text)
        if json_match:
            data = json.loads(json_match.group())
            letter = data.get("answer", "A").upper()
            if letter in letters:
                selected_answer = letters.index(letter)
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            reasoning = data.get("reasoning", response_text)
    except (json.JSONDecodeError, ValueError):
        for i, letter in enumerate(letters):
            if letter in response_text.upper()[:50]:
                selected_answer = i
                break

    return {
        "selected_answer": selected_answer,
        "confidence": confidence,
        "reasoning": reasoning,
    }


def evaluate_local_model(
    model,
    tokenizer,
    model_key: str,
    examples: List[MCQExample],
    dataset_name: str,
    method: str = "sequential",
    device: str = "cuda",
) -> List[BenchmarkResult]:
    """
    Evaluate a local model on MCQ examples.

    Args:
        model: Loaded HuggingFace model
        tokenizer: Loaded tokenizer
        model_key: Model identifier for results
        examples: MCQ examples to evaluate
        dataset_name: Name of dataset
        method: "sequential" or "unified"
        device: torch device

    Returns:
        List of BenchmarkResult objects
    """
    scorer = HLCCScorer()
    results = []

    for i, example in enumerate(examples):
        if (i + 1) % 20 == 0:
            correct_so_far = sum(1 for r in results if r.is_correct)
            print(f"  Progress: {i+1}/{len(examples)} ({correct_so_far}/{len(results)} correct)")

        start = datetime.now()

        if method == "sequential":
            parsed = evaluate_sequential(model, tokenizer, example, device)
        else:
            parsed = evaluate_unified(model, tokenizer, example, device)

        is_correct = parsed["selected_answer"] == example.correct_answer
        processing_time = (datetime.now() - start).total_seconds()

        results.append(BenchmarkResult(
            question_id=example.question_id,
            model_name=model_key,
            model_type="local",
            vendor="local",
            dataset=dataset_name,
            selected_answer=parsed["selected_answer"],
            correct_answer=example.correct_answer,
            is_correct=is_correct,
            confidence=parsed["confidence"],
            hlcc_score=scorer.hlcc_score(parsed["confidence"], is_correct),
            cbm_score=scorer.cbm_score(parsed["confidence"], is_correct),
            temperature=0.0,
            iteration=1,
            processing_time=processing_time,
            timestamp=start.isoformat(),
            method=method,
            reasoning=parsed.get("reasoning", ""),
        ))

    correct = sum(1 for r in results if r.is_correct)
    print(f"  [{model_key}] {correct}/{len(results)} correct "
          f"({correct/len(results)*100:.1f}%)")

    return results
