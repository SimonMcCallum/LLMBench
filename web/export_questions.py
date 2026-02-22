"""
Export MCQ questions to JSON files for the web server.

Run this on aroma (the machine with HuggingFace access) to pre-generate
question files that the Ubuntu web server can serve without any ML dependencies.

Usage:
    python web/export_questions.py                           # Export all datasets
    python web/export_questions.py --datasets truthfulqa arc-challenge
    python web/export_questions.py --max-examples 200        # More questions per dataset

Output goes to results/questions/ (committed to git, pulled by Ubuntu server).
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.datasets import load_mcq_dataset, list_datasets

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "questions"


def export_dataset(name: str, max_examples: int = 200):
    """Export a single dataset to JSON."""
    print(f"Exporting {name}...")
    try:
        examples = load_mcq_dataset(name, max_examples=max_examples)
    except Exception as e:
        print(f"  FAILED: {e}")
        return 0

    data = {
        "dataset": name,
        "exported": datetime.now().isoformat(),
        "total": len(examples),
        "questions": [ex.to_dict() for ex in examples],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outfile = OUTPUT_DIR / f"{name}.json"
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  {len(examples)} questions -> {outfile}")
    return len(examples)


def export_manifest():
    """Write a manifest listing all exported datasets."""
    datasets = []
    for f in sorted(OUTPUT_DIR.glob("*.json")):
        if f.name == "manifest.json":
            continue
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
            datasets.append({
                "name": data["dataset"],
                "total": data["total"],
                "exported": data["exported"],
                "file": f.name,
            })
        except Exception:
            pass

    manifest = {
        "updated": datetime.now().isoformat(),
        "datasets": datasets,
    }

    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written with {len(datasets)} datasets")


def main():
    parser = argparse.ArgumentParser(description="Export MCQ questions for web server")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets to export (default: all)")
    parser.add_argument("--max-examples", type=int, default=200,
                        help="Max questions per dataset")
    args = parser.parse_args()

    datasets = args.datasets or list_datasets()
    total = 0

    for ds in datasets:
        total += export_dataset(ds, args.max_examples)

    export_manifest()
    print(f"\nDone. {total} total questions exported to {OUTPUT_DIR}")
    print("Commit and push to make available on the web server:")
    print(f"  git add results/questions/ && git commit -m 'Update question exports' && git push")


if __name__ == "__main__":
    main()
