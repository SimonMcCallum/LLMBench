"""
Generate a hard question set from local model benchmark results.

Scans all benchmark results in data/results/, scores each question by how many
models got it wrong, and produces a curated "hard" subset. This hard set is then
used for cost-effective testing of expensive frontier API models.

Difficulty scoring:
  - Each model that got the question wrong adds 1 point
  - Larger models getting it wrong adds extra weight (a question that stumps
    a 14B model is harder than one that only stumps TinyLlama)
  - Questions where ALL models got it right are excluded
  - Questions where confident models got it wrong score higher (overconfidence trap)

Output:
  results/questions/hard.json       — Hard question set for API testing
  results/questions/hard_meta.json  — Difficulty metadata per question

Usage:
    python generate_hard_set.py                              # Generate from all results
    python generate_hard_set.py --min-difficulty 3           # Only very hard questions
    python generate_hard_set.py --max-questions 200          # Limit total size
    python generate_hard_set.py --datasets truthfulqa mmlu   # Filter datasets
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Load .env if present (needed for API pilot mode)
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and value:
                os.environ[key] = value  # .env always wins (may be updated)

sys.path.insert(0, str(Path(__file__).parent))

from core.model_loader import get_model_params_b


RESULTS_DIR = Path("data/results")
QUESTIONS_DIR = Path("results/questions")

# Weight multiplier based on model size — if a big model gets it wrong, it's harder
def _size_weight(model_name: str) -> float:
    """Larger models getting a question wrong = higher difficulty."""
    try:
        params = get_model_params_b(model_name)
    except Exception:
        params = 7.0  # default for unknown
    if params >= 24:
        return 3.0
    elif params >= 12:
        return 2.0
    elif params >= 6:
        return 1.5
    elif params >= 3:
        return 1.0
    else:
        return 0.5


def _confidence_penalty(confidence: float, is_correct: bool) -> float:
    """Extra difficulty points when a model was confidently wrong."""
    if is_correct:
        return 0.0
    # High confidence + wrong = this question is a trap
    return confidence * 2.0


def load_all_results() -> list:
    """Load all benchmark result files."""
    all_results = []
    if not RESULTS_DIR.exists():
        return all_results
    for f in sorted(RESULTS_DIR.glob("benchmark_*.json")):
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
            all_results.extend(data.get("results", []))
        except Exception as e:
            print(f"  Warning: Could not load {f.name}: {e}")
    return all_results


def compute_difficulty(results: list, dataset_filter: list = None) -> dict:
    """
    Compute difficulty score for each question across all model evaluations.

    Returns dict: question_id -> {
        difficulty: float,
        dataset: str,
        models_tested: int,
        models_wrong: int,
        models_correct: int,
        wrong_by: [model names],
        correct_by: [model names],
        avg_wrong_confidence: float,
        details: [{model, correct, confidence, size_weight}, ...]
    }
    """
    # Group results by question_id
    by_question = defaultdict(list)
    for r in results:
        ds = r.get("dataset", "")
        if dataset_filter and ds not in dataset_filter:
            continue
        qid = r["question_id"]
        by_question[qid].append(r)

    difficulties = {}
    for qid, evals in by_question.items():
        # Deduplicate: keep one result per model (latest)
        by_model = {}
        for e in evals:
            model = e["model_name"]
            by_model[model] = e  # last one wins

        models_tested = len(by_model)
        wrong_models = []
        correct_models = []
        difficulty = 0.0
        details = []
        wrong_confidences = []

        for model, e in by_model.items():
            is_correct = e["is_correct"]
            confidence = e.get("confidence", 0.5)
            sw = _size_weight(model)

            details.append({
                "model": model,
                "correct": is_correct,
                "confidence": round(confidence, 3),
                "size_weight": sw,
            })

            if not is_correct:
                wrong_models.append(model)
                difficulty += sw
                difficulty += _confidence_penalty(confidence, is_correct)
                wrong_confidences.append(confidence)
            else:
                correct_models.append(model)

        if not wrong_models:
            continue  # Skip questions everyone got right

        difficulties[qid] = {
            "question_id": qid,
            "dataset": evals[0].get("dataset", "unknown"),
            "difficulty": round(difficulty, 2),
            "models_tested": models_tested,
            "models_wrong": len(wrong_models),
            "models_correct": len(correct_models),
            "wrong_by": wrong_models,
            "correct_by": correct_models,
            "avg_wrong_confidence": round(
                sum(wrong_confidences) / len(wrong_confidences), 3
            ) if wrong_confidences else 0,
            "details": details,
        }

    return difficulties


def build_hard_set(
    difficulties: dict,
    min_difficulty: float = 1.0,
    max_questions: int = 500,
) -> list:
    """
    Select hard questions sorted by difficulty.

    Returns list of question metadata dicts, sorted hardest-first.
    """
    hard = [
        v for v in difficulties.values()
        if v["difficulty"] >= min_difficulty
    ]
    hard.sort(key=lambda x: x["difficulty"], reverse=True)
    return hard[:max_questions]


def export_hard_questions(hard_meta: list) -> int:
    """
    Create hard.json by pulling the actual question content from exported datasets.

    Returns number of questions exported.
    """
    # Load all exported question files
    question_lookup = {}
    for f in QUESTIONS_DIR.glob("*.json"):
        if f.name in ("manifest.json", "hard.json", "hard_meta.json"):
            continue
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for q in data.get("questions", []):
                question_lookup[q["question_id"]] = q
        except Exception:
            pass

    # Build hard question set
    hard_questions = []
    for meta in hard_meta:
        qid = meta["question_id"]
        if qid in question_lookup:
            q = dict(question_lookup[qid])
            q["difficulty"] = meta["difficulty"]
            q["models_wrong"] = meta["models_wrong"]
            q["models_tested"] = meta["models_tested"]
            q["wrong_by"] = meta["wrong_by"]
            hard_questions.append(q)

    if not hard_questions:
        print("No matching questions found in exported datasets.")
        return 0

    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Save the hard question set (same format as other dataset exports)
    hard_data = {
        "dataset": "hard",
        "description": "Questions that local LLMs struggle with — curated for frontier model testing",
        "exported": datetime.now().isoformat(),
        "total": len(hard_questions),
        "source_datasets": sorted(set(m["dataset"] for m in hard_meta)),
        "difficulty_range": {
            "min": min(q["difficulty"] for q in hard_questions),
            "max": max(q["difficulty"] for q in hard_questions),
            "mean": round(sum(q["difficulty"] for q in hard_questions) / len(hard_questions), 2),
        },
        "questions": hard_questions,
    }

    with open(QUESTIONS_DIR / "hard.json", "w", encoding="utf-8") as f:
        json.dump(hard_data, f, indent=2, ensure_ascii=False)

    # Save difficulty metadata (for analysis/paper)
    with open(QUESTIONS_DIR / "hard_meta.json", "w") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "total_questions_analyzed": len(hard_meta),
            "questions": hard_meta,
        }, f, indent=2)

    return len(hard_questions)


def print_summary(hard_meta: list):
    """Print a summary of the hard question set."""
    if not hard_meta:
        print("No hard questions found.")
        return

    by_dataset = defaultdict(list)
    for m in hard_meta:
        by_dataset[m["dataset"]].append(m)

    print(f"\n{'='*70}")
    print(f"HARD QUESTION SET: {len(hard_meta)} questions")
    print(f"{'='*70}")

    print(f"\nBy dataset:")
    for ds in sorted(by_dataset):
        items = by_dataset[ds]
        avg_diff = sum(i["difficulty"] for i in items) / len(items)
        avg_wrong = sum(i["models_wrong"] for i in items) / len(items)
        print(f"  {ds:20s}  {len(items):4d} questions  "
              f"avg difficulty: {avg_diff:.1f}  avg models wrong: {avg_wrong:.1f}")

    print(f"\nDifficulty distribution:")
    brackets = [(1, 2, "moderate"), (2, 4, "hard"), (4, 8, "very hard"), (8, 999, "extreme")]
    for lo, hi, label in brackets:
        count = sum(1 for m in hard_meta if lo <= m["difficulty"] < hi)
        if count:
            bar = "#" * min(count, 50)
            print(f"  {label:12s} ({lo:.0f}-{hi:.0f}): {count:4d}  {bar}")

    print(f"\nMost-failed questions (top 10):")
    for m in hard_meta[:10]:
        wrong_str = ", ".join(m["wrong_by"][:4])
        if len(m["wrong_by"]) > 4:
            wrong_str += f" +{len(m['wrong_by'])-4} more"
        print(f"  {m['question_id']:30s}  diff={m['difficulty']:5.1f}  "
              f"wrong: {wrong_str}")

    # Flag overconfidence traps
    traps = [m for m in hard_meta if m["avg_wrong_confidence"] > 0.6]
    if traps:
        print(f"\nOverconfidence traps ({len(traps)} questions where wrong models were >60% confident):")
        for m in sorted(traps, key=lambda x: x["avg_wrong_confidence"], reverse=True)[:5]:
            print(f"  {m['question_id']:30s}  avg wrong conf: {m['avg_wrong_confidence']:.1%}  "
                  f"wrong by: {', '.join(m['wrong_by'][:3])}")


def run_pilot_and_regenerate(
    pilot_model: str,
    datasets: list,
    max_examples: int = 200,
    min_accuracy: float = 0.3,
    max_accuracy: float = 0.8,
):
    """
    Run a cheap API model as pilot, then build hard set from questions
    where the pilot model's accuracy is in the calibration-useful range.

    The goal: find questions where frontier models get 40-70% accuracy,
    because that's where confidence calibration is actually measurable.
    If they get 95%+ right, HLCC just becomes 1+c for everything.

    Args:
        pilot_model: Cheap fast model (e.g. "gemini-2.5-flash-lite")
        datasets: Datasets to test on
        max_examples: Questions per dataset
        min_accuracy: Minimum per-dataset accuracy to include (too easy = no wrong answers)
        max_accuracy: Maximum per-dataset accuracy to include (too hard = random guessing)
    """
    from service.runner import run_benchmark

    print(f"{'='*70}")
    print(f"PILOT RUN: {pilot_model}")
    print(f"  Testing {len(datasets)} datasets x {max_examples} questions")
    print(f"  Looking for datasets with {min_accuracy:.0%}-{max_accuracy:.0%} accuracy")
    print(f"{'='*70}")

    result = run_benchmark({
        "models": [pilot_model],
        "datasets": datasets,
        "max_examples": max_examples,
        "temperatures": [0.0],
        "num_repetitions": 1,
    })

    # Analyze per-dataset accuracy from pilot
    all_results = load_all_results()
    pilot_results = [r for r in all_results if r["model_name"] == pilot_model]

    by_dataset = defaultdict(list)
    for r in pilot_results:
        by_dataset[r["dataset"]].append(r)

    print(f"\nPilot results ({pilot_model}):")
    print(f"  {'Dataset':25s} {'Accuracy':>8s} {'Confidence':>10s} {'Status':>12s}")
    print(f"  {'-'*60}")

    useful_datasets = []
    for ds in sorted(by_dataset):
        results_ds = by_dataset[ds]
        acc = sum(1 for r in results_ds if r["is_correct"]) / len(results_ds)
        conf = sum(r["confidence"] for r in results_ds) / len(results_ds)
        if min_accuracy <= acc <= max_accuracy:
            status = "USEFUL"
            useful_datasets.append(ds)
        elif acc > max_accuracy:
            status = "too easy"
        else:
            status = "too hard"
        print(f"  {ds:25s} {acc:7.1%} {conf:10.3f} {status:>12s}")

    # Now find the individual questions the pilot got wrong
    pilot_wrong = {r["question_id"] for r in pilot_results if not r["is_correct"]}
    # And questions it was uncertain on (confidence < 0.7 even if correct)
    pilot_uncertain = {r["question_id"] for r in pilot_results if r["confidence"] < 0.7}

    # Build hard set from: questions the pilot got wrong OR was uncertain about
    frontier_hard_ids = pilot_wrong | pilot_uncertain

    print(f"\nPilot analysis:")
    print(f"  Questions wrong:      {len(pilot_wrong)}")
    print(f"  Questions uncertain:  {len(pilot_uncertain)}")
    print(f"  Union (hard set):     {len(frontier_hard_ids)}")
    print(f"  Useful datasets:      {', '.join(useful_datasets) or 'none'}")

    # Regenerate hard set combining local model difficulty + pilot results
    difficulties = compute_difficulty(all_results)

    # Boost difficulty for questions the pilot also got wrong
    for qid in frontier_hard_ids:
        if qid in difficulties:
            difficulties[qid]["difficulty"] += 5.0  # Strong boost
        elif qid in {r["question_id"] for r in pilot_results}:
            # Question that only pilot got wrong — still interesting
            r = next(r for r in pilot_results if r["question_id"] == qid)
            difficulties[qid] = {
                "question_id": qid,
                "dataset": r["dataset"],
                "difficulty": 5.0,
                "models_tested": 1,
                "models_wrong": 1,
                "models_correct": 0,
                "wrong_by": [pilot_model],
                "correct_by": [],
                "avg_wrong_confidence": r["confidence"],
                "details": [{"model": pilot_model, "correct": False,
                            "confidence": r["confidence"], "size_weight": 5.0}],
            }

    hard_meta = build_hard_set(difficulties, min_difficulty=1.0, max_questions=500)
    print_summary(hard_meta)

    n_exported = export_hard_questions(hard_meta)
    if n_exported:
        print(f"\nExported {n_exported} hard questions to results/questions/hard.json")
        print(f"\nRecommended next step — run expensive frontier models on hard set:")
        print(f"  python run_all.py --api --datasets hard")

    return useful_datasets, frontier_hard_ids


def main():
    parser = argparse.ArgumentParser(
        description="Generate hard question set from benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_hard_set.py                           # From existing local results
  python generate_hard_set.py --tier extreme            # Only hardest questions
  python generate_hard_set.py --pilot gemini-2.5-flash-lite  # Run cheap API pilot first
  python generate_hard_set.py --pilot gemini-2.5-flash-lite --datasets mmlu-pro-math mmlu-pro-physics
        """
    )
    parser.add_argument("--min-difficulty", type=float, default=1.0,
                        help="Minimum difficulty score to include (default: 1.0)")
    parser.add_argument("--max-questions", type=int, default=500,
                        help="Maximum questions in hard set (default: 500)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Filter to specific datasets")
    parser.add_argument("--tier", choices=["moderate", "hard", "very-hard", "extreme"],
                        help="Only include questions at this difficulty tier or above")
    parser.add_argument("--pilot", type=str, default=None,
                        help="Run a cheap API model as pilot to find frontier-hard questions "
                             "(e.g. gemini-2.5-flash-lite)")
    parser.add_argument("--pilot-max", type=int, default=200,
                        help="Max questions per dataset for pilot run (default: 200)")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON to stdout")
    args = parser.parse_args()

    # Tier to min-difficulty mapping
    tier_thresholds = {
        "moderate": 1.0, "hard": 2.0, "very-hard": 4.0, "extreme": 8.0,
    }
    if args.tier:
        args.min_difficulty = max(args.min_difficulty, tier_thresholds[args.tier])

    # Pilot mode: run cheap API model first
    if args.pilot:
        pilot_datasets = args.datasets or [
            "mmlu-pro", "mmlu-pro-math", "mmlu-pro-physics", "truthfulqa", "medmcqa",
        ]
        run_pilot_and_regenerate(
            args.pilot, pilot_datasets, args.pilot_max,
        )
        return

    # Standard mode: generate from existing results
    results = load_all_results()
    if not results:
        print("No benchmark results found in data/results/")
        print("Run benchmarks first: python run_all.py --local")
        return

    n_models = len(set(r["model_name"] for r in results))
    n_questions = len(set(r["question_id"] for r in results))
    print(f"Loaded {len(results)} evaluations across {n_models} models, {n_questions} unique questions")

    # Compute difficulty
    difficulties = compute_difficulty(results, dataset_filter=args.datasets)
    print(f"Questions with at least 1 model wrong: {len(difficulties)}")

    # Build hard set
    hard_meta = build_hard_set(difficulties, args.min_difficulty, args.max_questions)

    if args.json:
        print(json.dumps(hard_meta, indent=2))
        return

    print_summary(hard_meta)

    # Export
    n_exported = export_hard_questions(hard_meta)
    if n_exported:
        print(f"\nExported {n_exported} hard questions to results/questions/hard.json")
        print(f"Difficulty metadata saved to results/questions/hard_meta.json")
        print(f"\nTo test frontier models on hard questions:")
        print(f"  python run_all.py --api --datasets hard --max-examples {n_exported}")
    else:
        print("\nCould not match questions to exported datasets.")
        print("Run export first: python run_all.py --export")


if __name__ == "__main__":
    main()
