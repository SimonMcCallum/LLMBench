"""
Import game design MCQ questions and human+AI CBM results from CBM-paper.

Sources:
  - CBM-paper/Code/mcq.json: 10 game design MCQ questions (5-choice)
  - CBM-paper/CBM_Assessment.csv: Human (students 60-77) and AI model results

Outputs:
  - results/questions/gamedesign.json: Questions in LLM-Bench format
  - results/human/gamedesign_humans.json: Human results for leaderboard
  - results/human/gamedesign_ais_cbmpaper.json: AI results from original CBM study
"""

import csv
import json
from datetime import datetime
from pathlib import Path

CBM_PAPER = Path("D:/git/CBM-paper")
MCQ_FILE = CBM_PAPER / "Code" / "mcq.json"
CSV_FILE = CBM_PAPER / "CBM_Assessment.csv"
QUESTIONS_DIR = Path("results/questions")
HUMAN_DIR = Path("results/human")


def import_questions():
    """Convert mcq.json to LLM-Bench format."""
    with open(MCQ_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    for q in data["questions"]:
        correct_idx = ord(q["correctAnswer"]) - ord("a")
        questions.append({
            "question_id": f"gamedesign_{q['id']}",
            "question": q["question"],
            "choices": [opt["text"] for opt in q["options"]],
            "correct_answer": correct_idx,
            "subject": "game_design",
            "difficulty": 0.7,
        })

    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    export = {
        "dataset": "gamedesign",
        "description": "Game design MCQ from university course — with human CBM data (zero contamination risk)",
        "exported": datetime.now().isoformat(),
        "total": len(questions),
        "questions": questions,
    }
    with open(QUESTIONS_DIR / "gamedesign.json", "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(questions)} game design questions")
    return questions


def parse_cbm_csv():
    """
    Parse CBM_Assessment.csv for human and AI results.

    CSV structure per participant (columns after the name):
      For each of 10 questions: answer_letter, confidence_level(1-3), cbm_score
      Triplets at columns: 16+(q-1)*3, 17+(q-1)*3, 18+(q-1)*3

    Confidence levels: 1=low(+1/0), 2=medium(+1.5/-0.5), 3=high(+2/-2)
    """
    correct_answers = ["d", "c", "e", "c", "e", "a", "e", "a", "e", "c"]
    # CBM scoring matrix (level -> correct, incorrect)
    cbm_matrix = {
        1: (1.0, 0.0),
        2: (1.5, -0.5),
        3: (2.0, -2.0),
    }

    humans = []
    ais = []

    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    for row in rows[1:]:  # Skip header
        if len(row) < 50:
            continue
        name = row[0].strip()
        if not name or name.startswith("Average") or name.startswith("Number"):
            break

        # Skip empty/absent students
        mc_score = row[1].strip()
        if mc_score == "-" or mc_score == "":
            continue

        # Parse 10 question triplets starting at column index 16
        # Format: answer, confidence, score (repeating)
        responses = []
        for q_idx in range(10):
            base = 16 + q_idx * 3
            if base + 2 >= len(row):
                break

            answer_raw = row[base].strip().lower()
            conf_raw = row[base + 1].strip()
            score_raw = row[base + 2].strip()

            if not answer_raw or not conf_raw:
                continue

            try:
                conf_level = int(conf_raw)
            except ValueError:
                continue

            correct = correct_answers[q_idx]
            is_correct = answer_raw == correct

            # Map 3-level CBM to 0-1 confidence for HLCC comparison
            # Level 1 = guessing (0), Level 2 = somewhat sure (0.5), Level 3 = confident (1.0)
            confidence_01 = {1: 0.0, 2: 0.5, 3: 1.0}.get(conf_level, 0.5)

            # HLCC score
            if is_correct:
                hlcc = 1.0 + confidence_01
            else:
                hlcc = -2.0 * (confidence_01 ** 2)

            cbm_correct_val, cbm_incorrect_val = cbm_matrix.get(conf_level, (1.0, 0.0))
            cbm = cbm_correct_val if is_correct else cbm_incorrect_val

            responses.append({
                "question_id": f"gamedesign_{q_idx + 1}",
                "selected_answer": ord(answer_raw) - ord("a") if answer_raw in "abcde" else -1,
                "correct_answer": ord(correct) - ord("a"),
                "is_correct": is_correct,
                "confidence": confidence_01,
                "confidence_level_cbm": conf_level,
                "hlcc_score": round(hlcc, 3),
                "cbm_score": cbm,
            })

        if not responses:
            continue

        n_correct = sum(1 for r in responses if r["is_correct"])
        accuracy = n_correct / len(responses)
        mean_conf = sum(r["confidence"] for r in responses) / len(responses)
        mean_hlcc = sum(r["hlcc_score"] for r in responses) / len(responses)
        mean_cbm = sum(r["cbm_score"] for r in responses) / len(responses)

        result = {
            "participant": name,
            "dataset": "gamedesign",
            "total_questions": len(responses),
            "metrics": {
                "accuracy": round(accuracy, 3),
                "mean_confidence": round(mean_conf, 3),
                "calibration_gap": round(abs(mean_conf - accuracy), 3),
            },
            "mean_hlcc": round(mean_hlcc, 3),
            "mean_cbm": round(mean_cbm, 3),
            "total_hlcc": round(sum(r["hlcc_score"] for r in responses), 3),
            "total_cbm": round(sum(r["cbm_score"] for r in responses), 3),
            "responses": responses,
        }

        # Determine if human or AI
        is_ai = any(kw in name.lower() for kw in
                     ["claude", "chatgpt", "gpt", "gemini", "deepseek"])
        if is_ai:
            ais.append(result)
        else:
            # Student numbered 60-77
            humans.append(result)

    return humans, ais


def main():
    # Import questions
    questions = import_questions()

    # Parse human + AI data
    humans, ais = parse_cbm_csv()

    HUMAN_DIR.mkdir(parents=True, exist_ok=True)

    # Save human results
    human_export = {
        "source": "CBM-paper/CBM_Assessment.csv",
        "dataset": "gamedesign",
        "description": "University game design students (numbered 60-77)",
        "total_participants": len(humans),
        "imported": datetime.now().isoformat(),
        "results": humans,
    }
    with open(HUMAN_DIR / "gamedesign_humans.json", "w") as f:
        json.dump(human_export, f, indent=2)

    # Save AI results from original CBM study
    ai_export = {
        "source": "CBM-paper/CBM_Assessment.csv",
        "dataset": "gamedesign",
        "description": "AI models tested on same questions as students (original CBM paper data)",
        "total_models": len(ais),
        "imported": datetime.now().isoformat(),
        "results": ais,
    }
    with open(HUMAN_DIR / "gamedesign_ais_cbmpaper.json", "w") as f:
        json.dump(ai_export, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"GAME DESIGN CBM DATA IMPORTED")
    print(f"{'='*70}")
    print(f"\nQuestions: {len(questions)} (5-choice, game design domain)")
    print(f"Correct answers: d, c, e, c, e, a, e, a, e, c")

    print(f"\nHumans ({len(humans)} students):")
    print(f"  {'Name':>6s}  {'Acc':>5s}  {'Conf':>5s}  {'HLCC':>6s}  {'CBM':>6s}  {'Gap':>5s}")
    for h in sorted(humans, key=lambda x: x["mean_hlcc"], reverse=True):
        m = h["metrics"]
        print(f"  {h['participant']:>6s}  {m['accuracy']:5.0%}  {m['mean_confidence']:5.2f}  "
              f"{h['mean_hlcc']:+6.2f}  {h['mean_cbm']:+6.2f}  {m['calibration_gap']:5.2f}")

    avg_acc = sum(h["metrics"]["accuracy"] for h in humans) / len(humans)
    avg_hlcc = sum(h["mean_hlcc"] for h in humans) / len(humans)
    avg_conf = sum(h["metrics"]["mean_confidence"] for h in humans) / len(humans)
    print(f"  {'AVG':>6s}  {avg_acc:5.0%}  {avg_conf:5.2f}  {avg_hlcc:+6.2f}")

    print(f"\nAI Models ({len(ais)} from original CBM study):")
    print(f"  {'Model':>35s}  {'Acc':>5s}  {'Conf':>5s}  {'HLCC':>6s}  {'CBM':>6s}")
    for a in sorted(ais, key=lambda x: x["mean_hlcc"], reverse=True):
        m = a["metrics"]
        print(f"  {a['participant']:>35s}  {m['accuracy']:5.0%}  {m['mean_confidence']:5.2f}  "
              f"{a['mean_hlcc']:+6.2f}  {a['mean_cbm']:+6.2f}")


if __name__ == "__main__":
    main()
