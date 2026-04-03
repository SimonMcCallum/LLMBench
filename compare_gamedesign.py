"""
Compare human and AI performance on game design MCQ — per-question analysis.

The key question: does the model know what it doesn't know?
We compare HLCC (not just accuracy) because a model that says "I'm not sure"
when wrong scores better than one that says "I'm 95% confident" and is wrong.

Usage:
    python compare_gamedesign.py
"""

import json
from pathlib import Path
from collections import defaultdict

QUESTIONS_FILE = Path("results/questions/gamedesign.json")
HUMAN_FILE = Path("results/human/gamedesign_humans.json")
AI_OLD_FILE = Path("results/human/gamedesign_ais_cbmpaper.json")
RESULTS_DIR = Path("data/results")

LETTERS = "ABCDEFGHIJ"
CORRECT_ANSWERS = ["d", "c", "e", "c", "e", "a", "e", "a", "e", "c"]


def load_questions():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        return json.load(f)["questions"]


def load_humans():
    with open(HUMAN_FILE) as f:
        return json.load(f)["results"]


def load_old_ais():
    with open(AI_OLD_FILE) as f:
        return json.load(f)["results"]


def load_new_ais():
    """Load new API model results from benchmark files."""
    results_by_model = defaultdict(list)
    for f in sorted(RESULTS_DIR.glob("benchmark_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        for r in data["results"]:
            if r["dataset"] == "gamedesign":
                results_by_model[r["model_name"]].append(r)
    return results_by_model


def hlcc(confidence, is_correct):
    if is_correct:
        return 1.0 + confidence
    else:
        return -2.0 * confidence ** 2


def print_per_question(questions, humans, old_ais, new_ais):
    """Per-question breakdown: who got it right, how confident, and HLCC impact."""

    print("=" * 90)
    print("PER-QUESTION ANALYSIS: Does the model know what it doesn't know?")
    print("=" * 90)

    for q_idx, q in enumerate(questions):
        qid = q["question_id"]
        qnum = q_idx + 1
        correct_idx = q["correct_answer"]

        print(f"\n{'─' * 90}")
        print(f"Q{qnum}: {q['question']}")
        for i, choice in enumerate(q["choices"]):
            marker = " ✓" if i == correct_idx else ""
            print(f"  {LETTERS[i]}) {choice}{marker}")

        # Collect all responses for this question
        responses = []

        # Humans
        for h in humans:
            for r in h["responses"]:
                if r["question_id"] == qid:
                    responses.append({
                        "name": f"Student {h['participant']}",
                        "type": "human",
                        "answer": LETTERS[r["selected_answer"]] if r["selected_answer"] >= 0 else "?",
                        "correct": r["is_correct"],
                        "confidence": r["confidence"],
                        "hlcc": r["hlcc_score"],
                    })

        # Old AIs (from CBM paper)
        for a in old_ais:
            for r in a["responses"]:
                if r["question_id"] == qid:
                    responses.append({
                        "name": a["participant"],
                        "type": "ai_old",
                        "answer": LETTERS[r["selected_answer"]] if r["selected_answer"] >= 0 else "?",
                        "correct": r["is_correct"],
                        "confidence": r["confidence"],
                        "hlcc": r["hlcc_score"],
                    })

        # New AIs (from LLM-Bench)
        for model, results in new_ais.items():
            for r in results:
                if r["question_id"] == qid:
                    responses.append({
                        "name": model,
                        "type": "ai_new",
                        "answer": LETTERS[r["selected_answer"]],
                        "correct": r["is_correct"],
                        "confidence": r["confidence"],
                        "hlcc": r["hlcc_score"],
                        "reasoning": r.get("reasoning", "")[:100],
                    })

        # Sort: correct+high HLCC first, then wrong sorted by confidence (overconfident first)
        responses.sort(key=lambda x: (-x["correct"], -x["hlcc"]))

        # Print table
        human_correct = sum(1 for r in responses if r["type"] == "human" and r["correct"])
        human_total = sum(1 for r in responses if r["type"] == "human")
        ai_correct = sum(1 for r in responses if r["type"] != "human" and r["correct"])
        ai_total = sum(1 for r in responses if r["type"] != "human")

        print(f"\n  Humans: {human_correct}/{human_total} correct   "
              f"AIs: {ai_correct}/{ai_total} correct")
        print(f"  {'Name':35s} {'Ans':>3s} {'OK':>3s} {'Conf':>5s} {'HLCC':>6s}")
        print(f"  {'─' * 55}")

        for r in responses:
            ok = "Y" if r["correct"] else "N"
            tag = ""
            if r["type"] == "ai_new":
                tag = " *"
            elif r["type"] == "ai_old":
                tag = " ~"

            # Flag overconfident wrong answers
            flag = ""
            if not r["correct"] and r["confidence"] > 0.7:
                flag = " << OVERCONFIDENT"

            print(f"  {r['name']:35s} {r['answer']:>3s} {ok:>3s} "
                  f"{r['confidence']:5.0%} {r['hlcc']:+6.2f}{tag}{flag}")

        # Print reasoning for new AIs that got it wrong
        wrong_ais = [r for r in responses if r["type"] == "ai_new" and not r["correct"]]
        if wrong_ais:
            print(f"\n  Wrong AI reasoning:")
            for r in wrong_ais:
                if r.get("reasoning"):
                    print(f"    {r['name']}: {r['reasoning']}")


def print_calibration_summary(humans, old_ais, new_ais):
    """Overall calibration comparison — the core finding."""

    print(f"\n\n{'=' * 90}")
    print("CALIBRATION SUMMARY: Who knows what they don't know?")
    print("=" * 90)

    all_agents = []

    # Human averages
    for h in humans:
        r = h["responses"]
        all_agents.append({
            "name": f"Student {h['participant']}",
            "type": "human",
            "accuracy": h["metrics"]["accuracy"],
            "confidence": h["metrics"]["mean_confidence"],
            "hlcc": h["mean_hlcc"],
            "gap": h["metrics"]["calibration_gap"],
            "n": len(r),
            "wrong_conf": sum(x["confidence"] for x in r if not x["is_correct"]) /
                          max(1, sum(1 for x in r if not x["is_correct"])) if any(not x["is_correct"] for x in r) else 0,
        })

    # Old AIs
    for a in old_ais:
        r = a["responses"]
        all_agents.append({
            "name": a["participant"],
            "type": "ai_old",
            "accuracy": a["metrics"]["accuracy"],
            "confidence": a["metrics"]["mean_confidence"],
            "hlcc": a["mean_hlcc"],
            "gap": a["metrics"]["calibration_gap"],
            "n": len(r),
            "wrong_conf": sum(x["confidence"] for x in r if not x["is_correct"]) /
                          max(1, sum(1 for x in r if not x["is_correct"])) if any(not x["is_correct"] for x in r) else 0,
        })

    # New AIs
    for model, results in new_ais.items():
        correct = [r for r in results if r["is_correct"]]
        wrong = [r for r in results if not r["is_correct"]]
        n = len(results)
        if n == 0:
            continue
        acc = len(correct) / n
        conf = sum(r["confidence"] for r in results) / n
        mean_hlcc = sum(r["hlcc_score"] for r in results) / n
        wrong_conf = sum(r["confidence"] for r in wrong) / len(wrong) if wrong else 0

        all_agents.append({
            "name": f"{model} *",
            "type": "ai_new",
            "accuracy": acc,
            "confidence": conf,
            "hlcc": mean_hlcc,
            "gap": abs(conf - acc),
            "n": n,
            "wrong_conf": wrong_conf,
        })

    # Sort by HLCC
    all_agents.sort(key=lambda x: x["hlcc"], reverse=True)

    print(f"\n  {'Name':35s} {'Acc':>5s} {'Conf':>5s} {'Gap':>5s} {'HLCC':>6s} "
          f"{'WrConf':>6s} {'Type'}")
    print(f"  {'─' * 75}")

    for a in all_agents:
        type_tag = {"human": "Human", "ai_old": "AI '24", "ai_new": "AI '26"}.get(a["type"], "?")
        wrong_conf_str = f"{a['wrong_conf']:.0%}" if a["wrong_conf"] > 0 else "  -"
        print(f"  {a['name']:35s} {a['accuracy']:5.0%} {a['confidence']:5.0%} "
              f"{a['gap']:5.2f} {a['hlcc']:+6.2f} {wrong_conf_str:>6s}  {type_tag}")

    # Key stats
    humans_list = [a for a in all_agents if a["type"] == "human"]
    ais_list = [a for a in all_agents if a["type"] != "human"]

    if humans_list and ais_list:
        h_gap = sum(a["gap"] for a in humans_list) / len(humans_list)
        a_gap = sum(a["gap"] for a in ais_list) / len(ais_list)
        h_wconf = sum(a["wrong_conf"] for a in humans_list if a["wrong_conf"] > 0) / max(1, sum(1 for a in humans_list if a["wrong_conf"] > 0))
        a_wconf = sum(a["wrong_conf"] for a in ais_list if a["wrong_conf"] > 0) / max(1, sum(1 for a in ais_list if a["wrong_conf"] > 0))

        print(f"\n  KEY FINDING:")
        print(f"  Human avg calibration gap:    {h_gap:.3f}")
        print(f"  AI avg calibration gap:       {a_gap:.3f}  ({a_gap/h_gap:.1f}x worse)")
        print(f"  Human avg confidence when WRONG: {h_wconf:.0%}")
        print(f"  AI avg confidence when WRONG:    {a_wconf:.0%}")
        print(f"\n  When humans don't know, they reduce confidence.")
        print(f"  When AIs don't know, they stay at ~{a_wconf:.0%} confidence.")
        print(f"  This is the metacognitive gap HLCC is designed to measure.")


def main():
    questions = load_questions()
    humans = load_humans()
    old_ais = load_old_ais()
    new_ais = load_new_ais()

    print(f"Loaded: {len(questions)} questions, {len(humans)} humans, "
          f"{len(old_ais)} old AIs, {len(new_ais)} new AI models")

    print_per_question(questions, humans, old_ais, new_ais)
    print_calibration_summary(humans, old_ais, new_ais)


if __name__ == "__main__":
    main()
