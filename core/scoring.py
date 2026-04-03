"""
HLCC and CBM Scoring

Implements both scoring models:
- HLCC (Hybrid Linear-Convex Confidence): continuous, asymmetric
- CBM (Confidence-Based Marking): discrete levels

Merged from CBM-paper and NNCONFIDENCE implementations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def _load_scoring_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "scoring.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class HLCCScorer:
    """
    HLCC Scoring: asymmetric reward/penalty based on confidence.

    Correct:   R(c) = 1 + c     (linear reward, max 2.0)
    Incorrect: P(c) = -2 * c^2  (quadratic penalty, max -2.0)
    """

    def __init__(self):
        config = _load_scoring_config()
        hlcc = config["hlcc"]
        self.correct_base = hlcc["correct_base"]
        self.correct_coeff = hlcc["correct_coeff"]
        self.incorrect_coeff = hlcc["incorrect_coeff"]
        self.incorrect_power = hlcc["incorrect_power"]

        cbm = config["cbm"]
        self.cbm_matrix = {}
        for level_str, scores in cbm["levels"].items():
            level = float(level_str)
            self.cbm_matrix[level] = {
                "correct": scores["correct"],
                "incorrect": scores["incorrect"],
            }

        ranking = config["ranking"]
        self.ranking_weights = {
            "hlcc": ranking["hlcc_weight"],
            "accuracy": ranking["accuracy_weight"],
            "calibration": ranking["calibration_weight"],
        }

    def hlcc_score(self, confidence: float, is_correct: bool) -> float:
        """Compute HLCC score for a single prediction."""
        if is_correct:
            return self.correct_base + self.correct_coeff * confidence
        else:
            return self.incorrect_coeff * (confidence ** self.incorrect_power)

    def cbm_score(self, confidence: float, is_correct: bool) -> float:
        """Compute CBM score using discrete confidence levels."""
        closest_level = min(self.cbm_matrix.keys(), key=lambda x: abs(x - confidence))
        key = "correct" if is_correct else "incorrect"
        return self.cbm_matrix[closest_level][key]

    def expected_score(self, confidence: float, accuracy: float) -> float:
        """Expected HLCC score given confidence and true accuracy."""
        correct_part = accuracy * (self.correct_base + self.correct_coeff * confidence)
        incorrect_part = (1.0 - accuracy) * (self.incorrect_coeff * confidence ** self.incorrect_power)
        return correct_part + incorrect_part

    def optimal_confidence(self, accuracy: float, bounded: bool = True) -> float:
        """Compute optimal confidence bet for a given accuracy level.

        When bounded=True (default): clamps to [0, 1] for standard HLCC.
        When bounded=False: returns unbounded c = p/(4(1-p)) for rational betting.
        """
        if accuracy <= 0.0:
            return 0.0
        if accuracy >= 1.0:
            return 1.0 if bounded else float("inf")
        optimal = accuracy / (4.0 * (1.0 - accuracy))
        if bounded:
            return min(1.0, max(0.0, optimal))
        return max(0.0, optimal)


@dataclass
class BenchmarkResult:
    """Result for a single model-question evaluation."""
    question_id: str
    model_name: str
    model_type: str  # "local" or "api"
    vendor: str      # "local", "openai", "anthropic", etc.
    dataset: str
    selected_answer: int
    correct_answer: int
    is_correct: bool
    confidence: float
    hlcc_score: float
    cbm_score: float
    temperature: float
    iteration: int
    processing_time: float
    timestamp: str
    method: str  # "sequential", "unified", "api"
    reasoning: str = ""


@dataclass
class RationalBetResult(BenchmarkResult):
    """Result for rational betting mode — includes probability and bet decomposition."""
    probability_p: float = 0.5     # Model's stated probability of being correct
    optimal_c: float = 0.25        # What c should be given p: p/(4(1-p))
    stated_c: float = 0.25         # What c the model actually stated
    betting_error: float = 0.0     # |stated_c - optimal_c|
    computation_correct: bool = True  # Did the model compute c from p correctly?


@dataclass
class ModelSummary:
    """Summary statistics for a model's benchmark run."""
    model_name: str
    model_type: str
    vendor: str
    dataset: str
    total_examples: int
    accuracy: float
    mean_confidence: float
    calibration_gap: float
    mean_hlcc_score: float
    mean_cbm_score: float
    correct_count: int
    incorrect_count: int

    def rank_score(self, weights: Optional[Dict] = None) -> float:
        """Composite ranking score. Higher is better."""
        if weights is None:
            weights = {"hlcc": 0.4, "accuracy": 0.3, "calibration": 0.3}
        cal_score = 1.0 - min(self.calibration_gap, 1.0)
        return (
            weights["hlcc"] * self.mean_hlcc_score
            + weights["accuracy"] * self.accuracy
            + weights["calibration"] * cal_score
        )


def compute_summary(results: List[BenchmarkResult]) -> Optional[ModelSummary]:
    """Compute summary statistics from a list of results."""
    if not results:
        return None

    r0 = results[0]
    accuracy = sum(1 for r in results if r.is_correct) / len(results)
    mean_conf = sum(r.confidence for r in results) / len(results)

    return ModelSummary(
        model_name=r0.model_name,
        model_type=r0.model_type,
        vendor=r0.vendor,
        dataset=r0.dataset,
        total_examples=len(results),
        accuracy=accuracy,
        mean_confidence=mean_conf,
        calibration_gap=abs(mean_conf - accuracy),
        mean_hlcc_score=sum(r.hlcc_score for r in results) / len(results),
        mean_cbm_score=sum(r.cbm_score for r in results) / len(results),
        correct_count=sum(1 for r in results if r.is_correct),
        incorrect_count=sum(1 for r in results if not r.is_correct),
    )
