"""
Calibration Metrics

Implements standard calibration metrics for evaluating confidence quality:
- ECE (Expected Calibration Error)
- Brier Score
- AUPRC (Area Under Precision-Recall Curve)
- PRR (Prediction Rejection Ratio)
- Selective Accuracy

Extracted from NNCONFIDENCE/hlcc_loss.py.
"""

from typing import Dict, List, Optional
import numpy as np


def compute_ece(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Expected Calibration Error.

    Measures average gap between confidence and accuracy across bins.
    Lower is better. < 0.05 is excellent, < 0.10 is acceptable.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = correct[in_bin].mean()
            ece += prop_in_bin * abs(avg_confidence - avg_accuracy)

    return float(ece)


def compute_brier_score(
    confidences: np.ndarray,
    correct: np.ndarray
) -> float:
    """
    Brier Score: mean squared error between confidence and correctness.
    Lower is better.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    return float(((confidences - correct) ** 2).mean())


def compute_auprc(
    confidences: np.ndarray,
    correct: np.ndarray
) -> float:
    """
    Area Under Precision-Recall Curve.

    Treats confidence as a predictor of correctness.
    Higher is better (1.0 = perfect).
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)

    n = len(confidences)
    n_pos = correct.sum()

    if n_pos == 0 or n_pos == n:
        return 0.5

    sorted_indices = np.argsort(-confidences)  # Descending
    sorted_correct = correct[sorted_indices]

    tp_cumsum = np.cumsum(sorted_correct)
    total = np.arange(1, n + 1, dtype=float)

    precision = tp_cumsum / total
    recall = tp_cumsum / n_pos

    # Prepend (recall=0, precision=1) for integration
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])

    # Trapezoidal integration
    dr = np.diff(recall)
    auprc = np.sum(dr * (precision[1:] + precision[:-1]) / 2)

    return float(auprc)


def compute_prr(
    confidences: np.ndarray,
    correct: np.ndarray
) -> float:
    """
    Prediction Rejection Ratio (LM-Polygraph standard metric).

    PRR = (AULC_random - AULC_model) / (AULC_random - AULC_oracle)

    PRR = 1.0 means oracle-level rejection quality.
    PRR = 0.0 means random rejection quality.
    Higher is better.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)

    n = len(confidences)
    if n == 0:
        return 0.0

    overall_acc = correct.mean()

    # Model AULC
    sorted_indices = np.argsort(-confidences)
    sorted_correct = correct[sorted_indices]
    cum_acc = np.cumsum(sorted_correct) / np.arange(1, n + 1, dtype=float)
    aulc_model = cum_acc.mean()

    # Random AULC
    aulc_random = overall_acc

    # Oracle AULC
    n_correct = int(correct.sum())
    oracle_correct = np.concatenate([
        np.ones(n_correct),
        np.zeros(n - n_correct),
    ])
    cum_acc_oracle = np.cumsum(oracle_correct) / np.arange(1, n + 1, dtype=float)
    aulc_oracle = cum_acc_oracle.mean()

    denom = aulc_random - aulc_oracle
    if abs(denom) < 1e-10:
        return 0.0

    return float((aulc_random - aulc_model) / denom)


def compute_selective_accuracy(
    confidences: np.ndarray,
    correct: np.ndarray,
    coverage_thresholds: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Selective prediction accuracy at fixed coverage thresholds.

    At coverage=0.7, keeps only the 70% most confident predictions.
    Lower coverage should yield higher accuracy if confidence is well-calibrated.
    """
    if coverage_thresholds is None:
        coverage_thresholds = [0.7, 0.8, 0.9]

    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)

    n = len(confidences)
    sorted_indices = np.argsort(-confidences)
    sorted_correct = correct[sorted_indices]

    result = {}
    for cov in coverage_thresholds:
        k = max(1, int(n * cov))
        kept = sorted_correct[:k]
        pct = int(cov * 100)
        result[f"acc@{pct}%"] = float(kept.mean())

    return result


def compute_all_metrics(
    confidences: np.ndarray,
    correct: np.ndarray
) -> Dict[str, float]:
    """Compute all calibration metrics at once."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)

    metrics = {
        "ece": compute_ece(confidences, correct),
        "brier": compute_brier_score(confidences, correct),
        "auprc": compute_auprc(confidences, correct),
        "prr": compute_prr(confidences, correct),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidences.mean()),
        "calibration_gap": float(abs(confidences.mean() - correct.mean())),
    }

    selective = compute_selective_accuracy(confidences, correct)
    metrics.update(selective)

    return metrics
