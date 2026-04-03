"""
Centralized benchmark runner — runs all configured models across datasets.

Three modes:
  1. Local models:  Tests all downloaded local models (uses NormShift confidence heads)
  2. API models:    Tests all API models with configured keys (costs tokens)
  3. Adaptive:      Runs easy models first, then selects hard questions for frontier models

Usage:
    python run_all.py --local                          # All local models, all datasets
    python run_all.py --api                            # All API models (needs keys)
    python run_all.py --api --models gpt-4.1 claude-sonnet-4-20250514
    python run_all.py --adaptive                       # Smart: easy models first, hard Qs for frontier
    python run_all.py --local --datasets truthfulqa arc-challenge --max-examples 50
    python run_all.py --export                         # Export all datasets for web server
    python run_all.py --cost-estimate                  # Estimate API costs without running

Environment:
    Set API keys in .env or environment variables (see .env.example)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Load .env if present
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

from core.model_loader import list_available_models, get_model_params_b, get_api_client
from core.datasets import list_datasets, load_mcq_dataset
from service.runner import run_benchmark


# ============================================================
# Cost estimation
# ============================================================

# Approximate cost per 1K input tokens (USD), as of early 2026
API_COSTS_PER_1K_INPUT = {
    "gpt-4.1": 0.002, "gpt-4.1-mini": 0.0004, "gpt-4.1-nano": 0.0001,
    "gpt-4o": 0.0025, "gpt-4o-mini": 0.00015,
    "o3": 0.01, "o4-mini": 0.0011,
    "claude-opus-4-20250514": 0.015, "claude-sonnet-4-20250514": 0.003,
    "claude-haiku-4-20250414": 0.0008,
    "gemini-2.5-pro": 0.00125, "gemini-2.5-flash": 0.00015,
    "gemini-2.0-flash": 0.0001,
    "deepseek-chat": 0.00014, "deepseek-reasoner": 0.00055,
}

AVG_TOKENS_PER_QUESTION = 350  # prompt + response


def estimate_costs(models, datasets, max_examples, temperatures, num_repetitions):
    """Estimate API costs for a benchmark run."""
    n_datasets = len(datasets)
    n_questions = max_examples * n_datasets
    total_cost = 0.0
    lines = []

    for model in models:
        cost_per_1k = API_COSTS_PER_1K_INPUT.get(model, 0.003)  # default guess
        n_calls = n_questions * len(temperatures) * num_repetitions
        tokens = n_calls * AVG_TOKENS_PER_QUESTION
        cost = (tokens / 1000) * cost_per_1k * 2  # input + output (rough 2x)
        total_cost += cost
        lines.append(f"  {model:40s}  {n_calls:5d} calls  ~${cost:.2f}")

    return total_cost, lines


# ============================================================
# Rational betting benchmark
# ============================================================

def run_rational_benchmark(config: dict) -> dict:
    """
    Run rational betting benchmark — separate from standard HLCC.

    Results are saved to separate files (rational_*) to avoid mixing experiments.
    """
    import asyncio
    from dataclasses import asdict
    from core.api_tester import evaluate_api_model_rational

    models = config.get("models", [])
    datasets = config.get("datasets", ["gamedesign"])
    max_examples = config.get("max_examples", 10)

    available = list_available_models()
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    RESULTS_DIR = Path("data/results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} [RATIONAL BETTING]")
        print(f"{'='*60}")

        examples = load_mcq_dataset(dataset_name, max_examples=max_examples)
        if not examples:
            print(f"  No examples loaded, skipping")
            continue

        for model_key in models:
            print(f"\nModel: {model_key}")

            # Find vendor
            vendor = None
            for v, v_models in available.get("api", {}).items():
                if model_key in v_models:
                    vendor = v
                    break

            if vendor is None:
                print(f"  Model {model_key} not found in API config. "
                      f"Rational mode is API-only.")
                continue

            client = get_api_client(vendor)
            if client is None:
                print(f"  No API key for {vendor}")
                continue

            try:
                results = asyncio.run(evaluate_api_model_rational(
                    vendor=vendor,
                    model=model_key,
                    examples=examples,
                    dataset_name=dataset_name,
                    api_key=client["api_key"],
                    endpoint=client["endpoint"],
                ))
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            if results:
                all_results.extend(results)

    # Save to SEPARATE files — rational_* prefix to avoid mixing with standard results
    if all_results:
        results_file = RESULTS_DIR / f"rational_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "mode": "rational_betting",
                "config": config,
                "results": [asdict(r) for r in all_results],
            }, f, indent=2)
        print(f"\nRational betting results saved to: {results_file}")

        # Print per-model summary
        by_model = {}
        for r in all_results:
            by_model.setdefault(r.model_name, []).append(r)

        print(f"\n{'='*70}")
        print(f"RATIONAL BETTING SUMMARY")
        print(f"{'='*70}")
        print(f"{'Model':30s} {'Acc':>5s} {'Mean p':>7s} {'Mean c':>7s} "
              f"{'Math OK':>7s} {'HLCC':>8s}")
        print(f"{'-'*70}")

        for model_name, mrs in by_model.items():
            acc = sum(1 for r in mrs if r.is_correct) / len(mrs)
            mean_p = sum(r.probability_p for r in mrs) / len(mrs)
            mean_c = sum(r.stated_c for r in mrs) / len(mrs)
            n_math = sum(1 for r in mrs if r.computation_correct)
            mean_hlcc = sum(r.hlcc_score for r in mrs) / len(mrs)
            print(f"{model_name:30s} {acc:5.0%} {mean_p:7.3f} {mean_c:7.3f} "
                  f"{n_math:3d}/{len(mrs):3d} {mean_hlcc:+8.2f}")

    return {
        "timestamp": timestamp,
        "mode": "rational_betting",
        "total_results": len(all_results),
    }


# ============================================================
# Adaptive difficulty
# ============================================================

def run_adaptive(datasets, max_examples, frontier_models, method="sequential"):
    """
    Adaptive difficulty: run small models first, then select questions they
    got wrong to create a "hard" subset for frontier models.

    This ensures frontier models are tested on genuinely difficult questions,
    making the benchmark more discriminating and cost-effective.
    """
    # Phase 1: Run cheap local models to find hard questions
    easy_models = ["tinyllama", "phi-2", "qwen2-1.5b"]
    available = list_available_models()
    easy_models = [m for m in easy_models if m in available["local"]]

    if not easy_models:
        print("No small local models available for difficulty calibration.")
        print("Falling back to full benchmark on all datasets.")
        return run_benchmark({
            "models": frontier_models,
            "datasets": datasets,
            "max_examples": max_examples,
            "temperatures": [0.0],
            "num_repetitions": 1,
            "method": method,
        })

    print("=" * 60)
    print("PHASE 1: Running easy models to calibrate difficulty")
    print(f"  Models: {', '.join(easy_models)}")
    print("=" * 60)

    phase1_result = run_benchmark({
        "models": easy_models,
        "datasets": datasets,
        "max_examples": max_examples,
        "temperatures": [0.0],
        "num_repetitions": 1,
        "method": method,
    })

    # Phase 2: Identify hard questions (ones that easy models got wrong)
    results_dir = Path("data/results")
    latest = sorted(results_dir.glob("benchmark_*.json"))[-1] if results_dir.exists() else None

    hard_question_ids = set()
    if latest:
        with open(latest) as f:
            data = json.load(f)
        # Count how many easy models got each question wrong
        question_errors = {}
        for r in data.get("results", []):
            qid = r["question_id"]
            if not r["is_correct"]:
                question_errors[qid] = question_errors.get(qid, 0) + 1
        # Questions wrong by at least half the easy models
        threshold = max(1, len(easy_models) // 2)
        hard_question_ids = {qid for qid, count in question_errors.items() if count >= threshold}

    n_hard = len(hard_question_ids)
    print(f"\n{'=' * 60}")
    print(f"PHASE 2: Found {n_hard} hard questions (wrong by >={threshold} easy models)")
    print(f"  Running frontier models on {'hard subset' if n_hard > 10 else 'full set'}")
    print(f"  Frontier models: {', '.join(frontier_models)}")
    print(f"{'=' * 60}")

    # Save hard question IDs for reference
    hard_file = results_dir / "hard_questions.json"
    if hard_question_ids:
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(hard_file, "w") as f:
            json.dump({
                "generated": datetime.now().isoformat(),
                "easy_models": easy_models,
                "threshold": threshold,
                "total_hard": n_hard,
                "question_ids": sorted(hard_question_ids),
            }, f, indent=2)

    # Phase 3: Run frontier models (on full set — hard_questions.json is for analysis)
    phase2_result = run_benchmark({
        "models": frontier_models,
        "datasets": datasets,
        "max_examples": max_examples,
        "temperatures": [0.0, 0.7],
        "num_repetitions": 2,
        "method": method,
    })

    return {
        "phase1": phase1_result,
        "phase2": phase2_result,
        "hard_questions": n_hard,
    }


# ============================================================
# Export helper
# ============================================================

def export_all_datasets(max_examples=200):
    """Export all configured datasets for the web server."""
    from web.export_questions import export_dataset, export_manifest
    datasets = list_datasets()
    total = 0
    for ds in datasets:
        total += export_dataset(ds, max_examples=max_examples)
    export_manifest()
    print(f"\nExported {total} questions across {len(datasets)} datasets")
    print("Commit: git add results/questions/ && git commit -m 'Export all datasets' && git push")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLM-Bench: Centralized confidence calibration benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--local", action="store_true", help="Run all local models")
    mode.add_argument("--api", action="store_true", help="Run all API models (costs tokens)")
    mode.add_argument("--adaptive", action="store_true",
                      help="Smart: easy models first, hard questions for frontier")
    mode.add_argument("--all", action="store_true", help="Run everything (local + API)")
    mode.add_argument("--export", action="store_true", help="Export all datasets for web server")
    mode.add_argument("--cost-estimate", action="store_true", help="Estimate API costs")

    parser.add_argument("--models", nargs="+", help="Specific model keys to run")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets (default: all)")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.0])
    parser.add_argument("--num-repetitions", type=int, default=1)
    parser.add_argument("--method", choices=["sequential", "unified"], default="sequential")
    parser.add_argument("--budget", type=float, default=None,
                        help="Maximum API spend in USD (e.g. --budget 5.00). "
                             "Saves partial results when exceeded.")
    parser.add_argument("--delay", type=float, default=0,
                        help="Seconds to wait between API calls (e.g. --delay 30). "
                             "Helps avoid rate limits on slow/free tiers.")
    parser.add_argument("--rational", action="store_true",
                        help="Rational betting mode: unbounded confidence, model must "
                             "output probability p and bet c=p/(4(1-p)). Tests both "
                             "calibration and rational computation.")

    args = parser.parse_args()
    datasets = args.datasets or list_datasets()

    # Set budget if specified
    if args.budget is not None:
        from core.api_tester import set_budget, reset_spend
        reset_spend()
        set_budget(args.budget)

    # Set delay between API calls
    if args.delay > 0:
        from core.api_tester import set_delay
        set_delay(args.delay)

    if args.export:
        export_all_datasets(args.max_examples or 200)
        return

    available = list_available_models()

    if args.cost_estimate or args.api:
        api_models = args.models or []
        if not api_models:
            for vendor, models in available.get("api", {}).items():
                api_models.extend(models)
        if not api_models:
            print("No API keys configured. Set keys in .env (see .env.example)")
            return

        if args.cost_estimate:
            total, lines = estimate_costs(
                api_models, datasets, args.max_examples,
                args.temperatures, args.num_repetitions,
            )
            print("API Cost Estimate:")
            print("-" * 60)
            for line in lines:
                print(line)
            print("-" * 60)
            print(f"  TOTAL: ~${total:.2f}")
            print(f"\n  ({len(datasets)} datasets x {args.max_examples} examples "
                  f"x {len(args.temperatures)} temps x {args.num_repetitions} reps)")
            return

    if args.adaptive:
        # Frontier = largest API models + largest local models
        frontier = args.models or []
        if not frontier:
            # Pick the best from each vendor
            for vendor, models in available.get("api", {}).items():
                if models:
                    frontier.append(models[0])  # First model is usually best
            # Add large local models
            for m in available["local"]:
                if get_model_params_b(m) >= 14:
                    frontier.append(m)
        result = run_adaptive(datasets, args.max_examples, frontier, args.method)
        print(f"\nAdaptive run complete. {result['hard_questions']} hard questions identified.")
        return

    # Standard run
    models = args.models or []
    if not models:
        # No explicit --models: use all available for the selected mode
        if args.local or args.all:
            models.extend(available["local"])
        if args.api or args.all:
            for vendor, vendor_models in available.get("api", {}).items():
                models.extend(vendor_models)
    elif args.local:
        # --local --models X Y: filter to only those local models
        models = [m for m in models if m in available["local"]]
    elif args.api:
        pass  # --api --models X Y: keep as-is (runner resolves vendor)

    if not models:
        print("No models selected. Use --local, --api, --all, --adaptive, or --models")
        parser.print_help()
        return

    # Deduplicate preserving order
    seen = set()
    unique_models = []
    for m in models:
        if m not in seen:
            seen.add(m)
            unique_models.append(m)

    mode_str = " [RATIONAL BETTING]" if args.rational else ""
    print(f"LLM-Bench: Running {len(unique_models)} models on {len(datasets)} datasets{mode_str}")
    print(f"  Models: {', '.join(unique_models)}")
    print(f"  Datasets: {', '.join(datasets)}")
    print(f"  Max examples: {args.max_examples}")

    if args.rational:
        # Rational betting mode — separate evaluation path
        result = run_rational_benchmark({
            "models": unique_models,
            "datasets": datasets,
            "max_examples": args.max_examples,
        })
    else:
        result = run_benchmark({
            "models": unique_models,
            "datasets": datasets,
            "max_examples": args.max_examples,
            "temperatures": args.temperatures,
            "num_repetitions": args.num_repetitions,
            "method": args.method,
        })

    print(f"\nDone. {result['total_results']} total evaluations.")
    if result.get("stopped") == "budget_exceeded":
        print(f"  (Stopped early — budget limit reached. Partial results saved.)")

    # Report spend
    try:
        from core.api_tester import get_spend
        spend = get_spend()
        if spend > 0:
            print(f"  Estimated API spend: ~${spend:.2f}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
