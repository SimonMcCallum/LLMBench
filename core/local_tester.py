"""
Local LLM Tester

Tests locally-loaded HuggingFace models on MCQ questions
with confidence extraction via logit analysis or trained NormShift heads.

When a trained NormShift confidence head is available (from NNCONFIDENCE),
it provides calibrated confidence estimates. Otherwise falls back to
entropy-based confidence.

Adapted from NNCONFIDENCE/hlcc_mcq_system.py MCQEvaluator.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

from core.datasets import MCQExample
from core.scoring import HLCCScorer, BenchmarkResult


def _get_checkpoint_dir() -> str:
    """Get checkpoint directory from machine.yaml config."""
    config_path = Path(__file__).parent.parent / "config" / "machine.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config.get("checkpoint_dir", "")
    return ""


def _extract_norm_shift_signals(hidden_states_tuple: tuple):
    """
    Extract normalization-shift signals from hidden states.

    Mirrors NNCONFIDENCE/hlcc_loss.py extract_norm_shift_signals().
    signal_i = 1 - std(hidden_states[i], dim=-1)
    """
    import torch
    layer_states = hidden_states_tuple[1:]  # skip embedding layer
    signals = []
    for h in layer_states:
        std = h.float().std(dim=-1)
        shift = 1.0 - std
        signals.append(shift)
    return torch.stack(signals, dim=-1)


def load_confidence_head(model_key: str, device: str = "cuda"):
    """
    Load a trained NormShift confidence head for the given model.

    Looks in checkpoint_dir (from machine.yaml) for:
      {model_key}_norm_shift/best_norm_shift_combined.pt
      {model_key}_norm_shift/best_norm_shift_norm_shift_only.pt

    Returns (head, head_type) or (None, None) if no checkpoint found.
    """
    import torch
    import torch.nn as nn

    checkpoint_dir = _get_checkpoint_dir()
    if not checkpoint_dir:
        return None, None

    cp_dir = os.path.join(checkpoint_dir, f"{model_key}_norm_shift")
    cp_path = None
    for variant in ["best_norm_shift_combined.pt", "best_norm_shift_norm_shift_only.pt"]:
        path = os.path.join(cp_dir, variant)
        if os.path.exists(path):
            cp_path = path
            break

    if cp_path is None:
        return None, None

    checkpoint = torch.load(cp_path, map_location=device, weights_only=False)
    head_type = checkpoint["head_type"]
    n_layers = checkpoint["n_layers"]

    if head_type == "combined":
        hidden_size = checkpoint["hidden_size"]
        head = _NormShiftConfidenceHead(hidden_size, n_layers)
    elif head_type == "norm_shift_only":
        head = _NormShiftOnlyConfidenceHead(n_layers)
    else:
        print(f"  Unknown head_type '{head_type}' in {cp_path}")
        return None, None

    head.load_state_dict(checkpoint["state_dict"])
    head.to(device)
    head.eval()

    metrics = checkpoint.get("metrics", {})
    hlcc = metrics.get("hlcc_score", "?")
    ece = metrics.get("ece", "?")
    print(f"  Loaded {head_type} confidence head from {os.path.basename(cp_path)}")
    if isinstance(hlcc, (int, float)):
        print(f"    Trained HLCC={hlcc:.3f}, ECE={ece:.3f}")

    return head, head_type


class _NormShiftConfidenceHead:
    """
    Dual-path confidence head combining norm-shift signals with hidden state.

    Mirrors NNCONFIDENCE/hlcc_loss.py NormShiftConfidenceHead.
    Defined inline to avoid importing from NNCONFIDENCE at runtime.
    """
    def __new__(cls, hidden_size: int, n_layers: int, dropout: float = 0.1):
        import torch.nn as nn
        import torch

        class NormShiftConfidenceHead(nn.Module):
            def __init__(self, hidden_size, n_layers, dropout=0.1):
                super().__init__()
                self.hidden_size = hidden_size
                self.n_layers = n_layers
                intermediate = max(64, n_layers * 2)

                self.norm_shift_proj = nn.Sequential(
                    nn.Linear(n_layers, intermediate),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.hidden_proj = nn.Sequential(
                    nn.Linear(hidden_size, intermediate),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.combine = nn.Sequential(
                    nn.Linear(2 * intermediate, intermediate),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(intermediate, 1),
                )
                self.temperature = nn.Parameter(torch.ones(1))

            def forward(self, hidden_states_tuple):
                norm_signals = _extract_norm_shift_signals(hidden_states_tuple)
                ns_last = norm_signals[:, -1, :]
                h_last = hidden_states_tuple[-1][:, -1, :].float()

                ns_proj = self.norm_shift_proj(ns_last)
                h_proj = self.hidden_proj(h_last)

                combined = torch.cat([ns_proj, h_proj], dim=-1)
                logit = self.combine(combined).squeeze(-1)
                return torch.sigmoid(logit / self.temperature)

        return NormShiftConfidenceHead(hidden_size, n_layers, dropout)


class _NormShiftOnlyConfidenceHead:
    """
    Norm-shift-only confidence head (ablation variant).

    Mirrors NNCONFIDENCE/hlcc_loss.py NormShiftOnlyConfidenceHead.
    """
    def __new__(cls, n_layers: int, dropout: float = 0.1):
        import torch.nn as nn
        import torch

        class NormShiftOnlyConfidenceHead(nn.Module):
            def __init__(self, n_layers, dropout=0.1):
                super().__init__()
                self.n_layers = n_layers
                intermediate = max(64, n_layers * 2)

                self.net = nn.Sequential(
                    nn.Linear(n_layers, intermediate),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(intermediate, intermediate),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(intermediate, 1),
                )
                self.temperature = nn.Parameter(torch.ones(1))

            def forward(self, hidden_states_tuple):
                norm_signals = _extract_norm_shift_signals(hidden_states_tuple)
                ns_last = norm_signals[:, -1, :]
                logit = self.net(ns_last).squeeze(-1)
                return torch.sigmoid(logit / self.temperature)

        return NormShiftOnlyConfidenceHead(n_layers, dropout)


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


def evaluate_sequential(
    model, tokenizer, example: MCQExample,
    device: str = "cuda",
    confidence_head=None,
) -> dict:
    """
    Sequential evaluation: get answer from logits, estimate confidence.

    Confidence priority: trained NormShift head > entropy-based fallback.
    When a confidence_head is provided, model is called with output_hidden_states=True
    to feed the head's dual-path architecture.

    Returns dict with selected_answer, confidence, reasoning, confidence_method.
    """
    import torch
    import torch.nn.functional as F

    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][:len(example.choices)]
    prompt = _format_mcq_prompt(example, include_confidence=False)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    need_hidden = confidence_head is not None

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=need_hidden)
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

    # Confidence: use trained head if available, otherwise entropy
    if confidence_head is not None and hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        with torch.no_grad():
            confidence = confidence_head(outputs.hidden_states).item()
        confidence = max(0.0, min(1.0, confidence))
        confidence_method = "norm_shift_head"
    else:
        entropy = -(probs * (probs + 1e-10).log()).sum()
        max_entropy = np.log(len(example.choices))
        confidence = float(1.0 - (entropy.item() / max_entropy))
        confidence = max(0.0, min(1.0, confidence))
        confidence_method = "entropy"

    return {
        "selected_answer": selected_answer,
        "confidence": confidence,
        "reasoning": f"Sequential evaluation ({confidence_method} confidence)",
        "confidence_method": confidence_method,
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
    confidence_head=None,
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
        confidence_head: Optional trained NormShift head for calibrated confidence.
                         Load via load_confidence_head(model_key, device).

    Returns:
        List of BenchmarkResult objects
    """
    scorer = HLCCScorer()
    results = []
    head_used = False

    for i, example in enumerate(examples):
        if (i + 1) % 20 == 0:
            correct_so_far = sum(1 for r in results if r.is_correct)
            print(f"  Progress: {i+1}/{len(examples)} ({correct_so_far}/{len(results)} correct)")

        start = datetime.now()

        if method == "sequential":
            parsed = evaluate_sequential(
                model, tokenizer, example, device,
                confidence_head=confidence_head,
            )
            if parsed.get("confidence_method") == "norm_shift_head":
                head_used = True
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
    conf_source = "NormShift head" if head_used else "entropy"
    print(f"  [{model_key}] {correct}/{len(results)} correct "
          f"({correct/len(results)*100:.1f}%), confidence: {conf_source}")

    return results
