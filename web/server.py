"""
LLM-Bench Human Quiz Server (standalone)

Serves MCQ questions from pre-exported JSON files. No ML dependencies needed.
Designed to run on the Ubuntu web server behind nginx.

Questions are exported by running `python web/export_questions.py` on aroma,
then committed to results/questions/ and pulled by this server.

LLM benchmark results are read from results/leaderboard.json (also committed
from aroma). Human quiz results are saved to results/human/ and committed.

Dependencies: flask (that's it — no torch, no transformers, no datasets)

Usage:
    python web/server.py                     # Start on port 8080
    python web/server.py --port 3000
    gunicorn web.server:app -b 0.0.0.0:8080  # Production
"""

import argparse
import json
import os
import random
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(32))
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True

# Register admin panel blueprint
from web.admin import admin_bp  # noqa: E402

app.register_blueprint(admin_bp)

# ============================================================
# PATHS — relative to repo root
# ============================================================

REPO_ROOT = Path(__file__).parent.parent
QUESTIONS_DIR = REPO_ROOT / "results" / "questions"
LEADERBOARD_FILE = REPO_ROOT / "results" / "leaderboard.json"
HUMAN_RESULTS_DIR = REPO_ROOT / "results" / "human"

# ============================================================
# INLINE HLCC/CBM SCORING (no core/ dependency)
# ============================================================

def hlcc_score(confidence: float, is_correct: bool) -> float:
    """HLCC: Correct = 1 + c, Incorrect = -2c^2"""
    if is_correct:
        return 1.0 + confidence
    else:
        return -2.0 * (confidence ** 2)


def cbm_score(confidence: float, is_correct: bool) -> float:
    """CBM with discrete levels."""
    levels = {1.0: (2.0, -2.0), 0.8: (1.5, -1.5), 0.6: (1.0, -1.0),
              0.4: (0.5, -0.5), 0.2: (0.0, 0.0)}
    closest = min(levels.keys(), key=lambda x: abs(x - confidence))
    correct_val, incorrect_val = levels[closest]
    return correct_val if is_correct else incorrect_val


def compute_ece(confidences, correct, n_bins=10):
    """Expected Calibration Error."""
    if not confidences:
        return 0.0
    n = len(confidences)
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        in_bin = [(c, a) for c, a in zip(confidences, correct) if lo < c <= hi]
        if in_bin:
            prop = len(in_bin) / n
            avg_conf = sum(c for c, _ in in_bin) / len(in_bin)
            avg_acc = sum(a for _, a in in_bin) / len(in_bin)
            ece += prop * abs(avg_conf - avg_acc)
    return ece


def compute_brier(confidences, correct):
    """Brier score: mean squared error."""
    if not confidences:
        return 0.0
    return sum((c - a) ** 2 for c, a in zip(confidences, correct)) / len(confidences)


def compute_metrics(responses):
    """Compute all metrics from a list of response dicts."""
    if not responses:
        return {}
    confidences = [r["confidence"] for r in responses]
    correct = [1.0 if r["is_correct"] else 0.0 for r in responses]
    n = len(responses)
    accuracy = sum(correct) / n
    mean_conf = sum(confidences) / n

    return {
        "accuracy": accuracy,
        "mean_confidence": mean_conf,
        "calibration_gap": abs(mean_conf - accuracy),
        "ece": compute_ece(confidences, correct),
        "brier": compute_brier(confidences, correct),
    }


# ============================================================
# QUESTION LOADING (from pre-exported JSON)
# ============================================================

_question_cache = {}


def _load_questions(dataset_name: str) -> list:
    """Load questions from exported JSON file."""
    if dataset_name in _question_cache:
        return _question_cache[dataset_name]

    filepath = QUESTIONS_DIR / f"{dataset_name}.json"
    if not filepath.exists():
        return []

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    _question_cache[dataset_name] = questions
    return questions


def _list_available_datasets() -> list:
    """List datasets that have exported question files."""
    manifest_path = QUESTIONS_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        return [d["name"] for d in manifest.get("datasets", [])]

    # Fallback: scan directory
    if not QUESTIONS_DIR.exists():
        return []
    return [f.stem for f in sorted(QUESTIONS_DIR.glob("*.json")) if f.stem != "manifest"]


# ============================================================
# SESSION STORE
# ============================================================

sessions = {}


# ============================================================
# ROUTES — STATIC
# ============================================================

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/explore")
def explore():
    return send_from_directory("static", "explore.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# ============================================================
# ROUTES — API
# ============================================================

@app.route("/api/hard-questions")
def api_hard_questions():
    """Serve the hard/discriminating question analysis."""
    analysis_path = QUESTIONS_DIR / "hard_analysis.json"
    if not analysis_path.exists():
        return jsonify({"error": "No hard question analysis found. "
                        "Run the analysis script first."}), 404
    with open(analysis_path, encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/api/datasets")
def api_datasets():
    """List available datasets (those with exported question files)."""
    datasets = _list_available_datasets()
    return jsonify({"datasets": datasets})


@app.route("/api/start", methods=["POST"])
def api_start():
    """
    Start a new quiz session.

    Body: { "dataset": "truthfulqa", "max_questions": 20, "participant": "name" }
    """
    data = request.json or {}
    dataset_name = data.get("dataset", "truthfulqa")
    max_questions = min(data.get("max_questions", 20), 200)
    participant = data.get("participant", "anonymous")

    questions = _load_questions(dataset_name)
    if not questions:
        return jsonify({"error": f"No questions available for '{dataset_name}'. "
                        "Run export_questions.py on aroma first."}), 400

    # Shuffle and limit
    session_id = uuid.uuid4().hex[:12]
    rng = random.Random(session_id)
    shuffled = list(questions)
    rng.shuffle(shuffled)
    shuffled = shuffled[:max_questions]

    sessions[session_id] = {
        "id": session_id,
        "dataset": dataset_name,
        "participant": participant,
        "questions": shuffled,
        "responses": [],
        "current": 0,
        "started": datetime.now().isoformat(),
    }

    return jsonify({
        "session_id": session_id,
        "dataset": dataset_name,
        "total_questions": len(shuffled),
    })


@app.route("/api/question/<session_id>")
def api_question(session_id):
    """Get the current question for a session."""
    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    idx = session["current"]
    if idx >= len(session["questions"]):
        return jsonify({"done": True, "message": "Quiz complete"})

    q = session["questions"][idx]
    return jsonify({
        "question_number": idx + 1,
        "total_questions": len(session["questions"]),
        "question_id": q["question_id"],
        "question": q["question"],
        "choices": q["choices"],
        "subject": q.get("subject", ""),
        "dataset": session["dataset"],
    })


@app.route("/api/answer", methods=["POST"])
def api_answer():
    """
    Submit an answer with confidence.

    Body: { "session_id": "...", "selected_answer": 0, "confidence": 0.8 }
    """
    data = request.json or {}
    session_id = data.get("session_id")
    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    idx = session["current"]
    if idx >= len(session["questions"]):
        return jsonify({"error": "Quiz already complete"}), 400

    q = session["questions"][idx]
    selected = int(data.get("selected_answer", 0))
    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    is_correct = selected == q["correct_answer"]
    h_score = hlcc_score(confidence, is_correct)
    c_score = cbm_score(confidence, is_correct)

    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    response = {
        "question_id": q["question_id"],
        "selected_answer": selected,
        "selected_letter": letters[selected] if selected < len(letters) else "?",
        "correct_answer": q["correct_answer"],
        "correct_letter": letters[q["correct_answer"]] if q["correct_answer"] < len(letters) else "?",
        "correct_text": q["choices"][q["correct_answer"]] if q["correct_answer"] < len(q["choices"]) else "",
        "is_correct": is_correct,
        "confidence": confidence,
        "hlcc_score": h_score,
        "cbm_score": c_score,
        "timestamp": datetime.now().isoformat(),
    }

    session["responses"].append(response)
    session["current"] = idx + 1

    # Running totals
    responses = session["responses"]
    total_correct = sum(1 for r in responses if r["is_correct"])
    total_hlcc = sum(r["hlcc_score"] for r in responses)

    response["running_accuracy"] = total_correct / len(responses)
    response["running_hlcc"] = total_hlcc / len(responses)
    response["running_cbm"] = sum(r["cbm_score"] for r in responses) / len(responses)
    response["questions_remaining"] = len(session["questions"]) - session["current"]

    return jsonify(response)


@app.route("/api/results/<session_id>")
def api_results(session_id):
    """Get full results for a completed session."""
    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    responses = session["responses"]
    if not responses:
        return jsonify({"error": "No responses yet"}), 400

    metrics = compute_metrics(responses)
    hlcc_scores = [r["hlcc_score"] for r in responses]
    cbm_scores = [r["cbm_score"] for r in responses]

    result = {
        "session_id": session_id,
        "participant": session["participant"],
        "dataset": session["dataset"],
        "total_questions": len(responses),
        "completed": session["current"] >= len(session["questions"]),
        "started": session["started"],
        "finished": datetime.now().isoformat(),
        "metrics": metrics,
        "mean_hlcc": sum(hlcc_scores) / len(hlcc_scores),
        "mean_cbm": sum(cbm_scores) / len(cbm_scores),
        "total_hlcc": sum(hlcc_scores),
        "total_cbm": sum(cbm_scores),
        "responses": responses,
    }

    # Save to disk (committed to repo, visible to aroma too)
    _save_human_result(result)

    return jsonify(result)


@app.route("/api/leaderboard")
def api_leaderboard():
    """Combined leaderboard: LLM results (from aroma) + human results."""
    entries = []

    # LLM results — committed by aroma daemon
    if LEADERBOARD_FILE.exists():
        try:
            with open(LEADERBOARD_FILE, "r") as f:
                llm_data = json.load(f)
            for key, entry in llm_data.get("entries", {}).items():
                entry["type"] = "llm"
                entry["entry_key"] = key
                entries.append(entry)
        except Exception:
            pass

    # Human results — saved locally on this server
    if HUMAN_RESULTS_DIR.exists():
        for f in sorted(HUMAN_RESULTS_DIR.glob("*.json")):
            try:
                with open(f, "r") as fh:
                    data = json.load(fh)
                entries.append({
                    "type": "human",
                    "entry_key": f"human_{data.get('participant', 'anon')}_{data.get('dataset', '')}",
                    "model_name": f"Human: {data.get('participant', 'Anonymous')}",
                    "dataset": data.get("dataset", ""),
                    "accuracy": data.get("metrics", {}).get("accuracy", 0),
                    "mean_confidence": data.get("metrics", {}).get("mean_confidence", 0),
                    "mean_hlcc_score": data.get("mean_hlcc", 0),
                    "calibration_gap": data.get("metrics", {}).get("calibration_gap", 0),
                    "ece": data.get("metrics", {}).get("ece", 0),
                    "total_examples": data.get("total_questions", 0),
                    "updated": data.get("finished", ""),
                })
            except Exception:
                pass

    entries.sort(key=lambda x: x.get("mean_hlcc_score", 0), reverse=True)
    return jsonify({"entries": entries})


def _save_human_result(result):
    """Save human quiz result to results/human/ (committed to git)."""
    HUMAN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    participant = result.get("participant", "anonymous").replace(" ", "_")[:30]
    dataset = result.get("dataset", "unknown")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"human_{participant}_{dataset}_{ts}.json"

    with open(HUMAN_RESULTS_DIR / filename, "w") as f:
        json.dump(result, f, indent=2)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-Bench Human Quiz Server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"LLM-Bench Quiz Server")
    print(f"  Questions dir: {QUESTIONS_DIR}")
    print(f"  Leaderboard:   {LEADERBOARD_FILE}")
    print(f"  Human results: {HUMAN_RESULTS_DIR}")
    print(f"  Datasets:      {_list_available_datasets()}")
    print(f"  URL: http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
