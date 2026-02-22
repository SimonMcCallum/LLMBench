"""
Benchmark Runner

Unified entry point for executing benchmark jobs. Handles:
- Parsing job configuration (YAML frontmatter or dict)
- Loading models (local or API)
- Running evaluations
- Computing metrics
- Saving results and updating leaderboard

Can be invoked directly or by the daemon.
"""

import asyncio
import json
import os
import sys
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.datasets import MCQExample, load_mcq_dataset
from core.scoring import HLCCScorer, BenchmarkResult, compute_summary
from core.metrics import compute_all_metrics
from core.model_loader import (
    load_local_model, unload_model, get_api_client,
    list_available_models, get_cache_dir,
)


RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
HISTORY_DIR = Path(__file__).parent.parent / "results" / "history"
LEADERBOARD_FILE = Path(__file__).parent.parent / "results" / "leaderboard.json"


def parse_job_frontmatter(content: str) -> Optional[Dict]:
    """
    Parse YAML frontmatter from a task file.

    Returns the parsed config dict, or None if no frontmatter found.
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return None

    try:
        return yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None


def run_benchmark(config: Dict) -> Dict:
    """
    Execute a benchmark job from a configuration dict.

    Config keys:
        models: list of model keys
        datasets: list of dataset names
        max_examples: int (default 100)
        temperatures: list of floats (default [0.0])
        num_repetitions: int (default 1)
        method: "sequential" | "unified" (for local models, default "sequential")

    Returns:
        Summary dict with results per model/dataset
    """
    models = config.get("models", [])
    datasets = config.get("datasets", ["truthfulqa"])
    max_examples = config.get("max_examples", 100)
    temperatures = config.get("temperatures", [0.0])
    num_repetitions = config.get("num_repetitions", 1)
    method = config.get("method", "sequential")

    if isinstance(datasets, str):
        datasets = [datasets]
    if isinstance(models, str):
        models = [models]

    available = list_available_models()
    all_results = []
    summaries = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        examples = load_mcq_dataset(dataset_name, max_examples=max_examples)
        if not examples:
            print(f"  No examples loaded, skipping")
            continue

        for model_key in models:
            print(f"\nModel: {model_key}")

            # Determine if local or API model
            if model_key in available["local"]:
                results = _run_local_benchmark(
                    model_key, examples, dataset_name, method
                )
            else:
                # Check API models
                results = asyncio.run(_run_api_benchmark(
                    model_key, examples, dataset_name, temperatures, num_repetitions
                ))

            if results:
                all_results.extend(results)
                summary = compute_summary(results)
                if summary:
                    summaries.append(summary)
                    _print_summary(summary)

    # Save results
    _save_results(all_results, summaries, timestamp, config)

    return {
        "timestamp": timestamp,
        "total_results": len(all_results),
        "summaries": [asdict(s) for s in summaries],
    }


def _run_local_benchmark(
    model_key: str,
    examples: List[MCQExample],
    dataset_name: str,
    method: str,
) -> List[BenchmarkResult]:
    """Run benchmark on a local model."""
    from core.local_tester import evaluate_local_model

    model, tokenizer, info = load_local_model(model_key)
    device = "cuda" if info.get("vram_gb", 0) > 0 else "cpu"

    try:
        results = evaluate_local_model(
            model, tokenizer, model_key, examples, dataset_name,
            method=method, device=device,
        )
    finally:
        unload_model(model)

    return results


async def _run_api_benchmark(
    model_key: str,
    examples: List[MCQExample],
    dataset_name: str,
    temperatures: List[float],
    num_repetitions: int,
) -> List[BenchmarkResult]:
    """Run benchmark on an API model."""
    from core.api_tester import evaluate_api_model

    # Find which vendor this model belongs to
    available = list_available_models()
    vendor = None
    for v, models in available.get("api", {}).items():
        if model_key in models:
            vendor = v
            break

    if vendor is None:
        print(f"  Model {model_key} not found in any API vendor config")
        return []

    client = get_api_client(vendor)
    if client is None:
        print(f"  No API key for {vendor}")
        return []

    return await evaluate_api_model(
        vendor=vendor,
        model=model_key,
        examples=examples,
        dataset_name=dataset_name,
        temperatures=temperatures,
        num_repetitions=num_repetitions,
        api_key=client["api_key"],
        endpoint=client["endpoint"],
    )


def _print_summary(summary):
    """Print a formatted summary."""
    print(f"  Accuracy:   {summary.accuracy:.1%}")
    print(f"  Confidence: {summary.mean_confidence:.3f}")
    print(f"  HLCC Score: {summary.mean_hlcc_score:+.3f}")
    print(f"  CBM Score:  {summary.mean_cbm_score:+.3f}")
    print(f"  Cal. Gap:   {summary.calibration_gap:.3f}")


def _save_results(
    results: List[BenchmarkResult],
    summaries,
    timestamp: str,
    config: Dict,
):
    """Save results to disk and update leaderboard."""
    if not results:
        return

    # Save detailed results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / f"benchmark_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "config": config,
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")

    # Save summary to history (committed to git)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = HISTORY_DIR / f"summary_{timestamp}.json"

    # Compute calibration metrics
    confidences = np.array([r.confidence for r in results])
    correct = np.array([float(r.is_correct) for r in results])
    metrics = compute_all_metrics(confidences, correct)

    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "config": config,
            "overall_metrics": metrics,
            "summaries": [asdict(s) for s in summaries],
        }, f, indent=2)
    print(f"Summary saved to: {summary_file}")

    # Update leaderboard
    _update_leaderboard(summaries)


def _update_leaderboard(summaries):
    """Update the leaderboard file with new results."""
    leaderboard = {}
    if LEADERBOARD_FILE.exists():
        try:
            with open(LEADERBOARD_FILE, "r") as f:
                leaderboard = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            leaderboard = {}

    entries = leaderboard.get("entries", {})

    for summary in summaries:
        key = f"{summary.model_name}_{summary.dataset}"
        entry = asdict(summary)
        entry["rank_score"] = summary.rank_score()
        entry["updated"] = datetime.now().isoformat()
        entries[key] = entry

    # Sort by rank score
    sorted_entries = dict(
        sorted(entries.items(), key=lambda x: x[1].get("rank_score", 0), reverse=True)
    )

    LEADERBOARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump({
            "updated": datetime.now().isoformat(),
            "entries": sorted_entries,
        }, f, indent=2)

    print(f"Leaderboard updated: {LEADERBOARD_FILE}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-Bench Runner")
    parser.add_argument("--models", nargs="+", required=True, help="Model keys to evaluate")
    parser.add_argument("--datasets", nargs="+", default=["truthfulqa"], help="Datasets to use")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.0])
    parser.add_argument("--num-repetitions", type=int, default=1)
    parser.add_argument("--method", choices=["sequential", "unified"], default="sequential")
    parser.add_argument("--job-file", help="YAML/MD file with job configuration")

    args = parser.parse_args()

    if args.job_file:
        with open(args.job_file, "r") as f:
            content = f.read()
        config = parse_job_frontmatter(content)
        if config is None:
            config = yaml.safe_load(content)
    else:
        config = {
            "models": args.models,
            "datasets": args.datasets,
            "max_examples": args.max_examples,
            "temperatures": args.temperatures,
            "num_repetitions": args.num_repetitions,
            "method": args.method,
        }

    result = run_benchmark(config)
    print(f"\nDone. {result['total_results']} total evaluations.")
