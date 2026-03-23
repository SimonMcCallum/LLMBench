"""
LLM-Bench Human Quiz Server

Serves MCQ questions from pre-exported JSON files with persistent sessions
stored in SQLite. Supports user accounts, score sharing, and performance
tracking over time.

Dependencies: flask, gunicorn, pyyaml, bcrypt

Usage:
    python web/server.py                     # Start on port 8080
    gunicorn web.server:app -b 0.0.0.0:8080  # Production
"""

import argparse
import json
import os
import random
import secrets
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, session

app = Flask(__name__, static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True

# Support serving under a URL prefix (e.g. /bench on simonmccallum.org.nz/bench/)
# Set URL_PREFIX env var or --prefix CLI arg. Nginx strips the prefix before proxying,
# so Flask routes stay clean — but we also support direct proxying where the prefix
# is passed through, using APPLICATION_ROOT + DispatcherMiddleware.
_url_prefix = os.environ.get("URL_PREFIX", "")  # e.g. "/bench"

# Register admin panel blueprint
from web.admin import admin_bp  # noqa: E402
from web.user_auth import auth_bp  # noqa: E402

app.register_blueprint(admin_bp)
app.register_blueprint(auth_bp)

# Initialize database
from web import database as db  # noqa: E402

db.init_app(app)

# ============================================================
# PATHS — relative to repo root
# ============================================================

REPO_ROOT = Path(__file__).parent.parent
QUESTIONS_DIR = REPO_ROOT / "results" / "questions"
LEADERBOARD_FILE = REPO_ROOT / "results" / "leaderboard.json"
HUMAN_RESULTS_DIR = REPO_ROOT / "results" / "human"
BENCHMARK_RESULTS_DIR = REPO_ROOT / "data" / "results"

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


def _get_question_by_id(dataset_name, question_id):
    """Look up a single question by ID from the cached dataset."""
    questions = _load_questions(dataset_name)
    for q in questions:
        if q["question_id"] == question_id:
            return q
    return None


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


def _dataset_question_counts():
    """Get question counts per dataset."""
    result = {}
    for ds in _list_available_datasets():
        questions = _load_questions(ds)
        result[ds] = len(questions)
    return result


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
# ROUTES — QUIZ API
# ============================================================

_benchmark_cache = None
_benchmark_cache_mtime = 0


def _load_all_benchmarks():
    """Load and consolidate all benchmark results. Cached until files change."""
    global _benchmark_cache, _benchmark_cache_mtime
    if not BENCHMARK_RESULTS_DIR.exists():
        return {"results": [], "models": [], "datasets": []}

    # Check if any file is newer than cache
    latest_mtime = max(
        (f.stat().st_mtime for f in BENCHMARK_RESULTS_DIR.glob("*.json")),
        default=0,
    )
    if _benchmark_cache and latest_mtime <= _benchmark_cache_mtime:
        return _benchmark_cache

    all_results = []
    for fpath in sorted(BENCHMARK_RESULTS_DIR.glob("*.json")):
        if fpath.name == ".gitkeep":
            continue
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            for r in data.get("results", []):
                all_results.append(r)
        except Exception:
            pass

    # Deduplicate: keep latest result per (model, dataset, question_id, method)
    seen = {}
    for r in all_results:
        key = (r.get("model_name"), r.get("dataset"), r.get("question_id"),
               r.get("method", "sequential"))
        seen[key] = r  # later files overwrite earlier
    deduped = list(seen.values())

    models = sorted(set(r.get("model_name", "?") for r in deduped))
    datasets = sorted(set(r.get("dataset", "?") for r in deduped))

    _benchmark_cache = {
        "total_results": len(deduped),
        "models": models,
        "datasets": datasets,
        "results": deduped,
    }
    _benchmark_cache_mtime = latest_mtime
    return _benchmark_cache


@app.route("/api/benchmark-results")
def api_benchmark_results():
    """Consolidated benchmark results with optional filters.

    Query params:
        model: filter by model name
        dataset: filter by dataset
        question_id: filter by question ID
        method: filter by method (sequential, api, rational)
        summary: if "true", return per-model-dataset summaries instead of raw results
    """
    data = _load_all_benchmarks()
    results = data["results"]

    # Apply filters
    model = request.args.get("model")
    dataset = request.args.get("dataset")
    qid = request.args.get("question_id")
    method = request.args.get("method")

    if model:
        results = [r for r in results if r.get("model_name") == model]
    if dataset:
        results = [r for r in results if r.get("dataset") == dataset]
    if qid:
        results = [r for r in results if r.get("question_id") == qid]
    if method:
        results = [r for r in results if r.get("method") == method]

    # Summary mode: aggregate per model-dataset
    if request.args.get("summary") == "true":
        from collections import defaultdict
        groups = defaultdict(list)
        for r in results:
            groups[(r.get("model_name"), r.get("dataset"), r.get("method"))].append(r)

        summaries = []
        for (m, d, meth), rlist in sorted(groups.items()):
            n = len(rlist)
            correct = sum(1 for r in rlist if r.get("is_correct"))
            confs = [r.get("confidence", 0) for r in rlist]
            hlccs = [r.get("hlcc_score", 0) for r in rlist]
            summaries.append({
                "model_name": m, "dataset": d, "method": meth,
                "total": n, "correct": correct,
                "accuracy": correct / n if n else 0,
                "mean_confidence": sum(confs) / n if n else 0,
                "mean_hlcc": sum(hlccs) / n if n else 0,
                "calibration_gap": abs(sum(confs) / n - correct / n) if n else 0,
            })
        return jsonify({
            "models": data["models"], "datasets": data["datasets"],
            "summaries": summaries,
        })

    return jsonify({
        "total": len(results),
        "models": data["models"], "datasets": data["datasets"],
        "results": results,
    })


@app.route("/api/question-detail/<question_id>")
def api_question_detail(question_id):
    """All model answers for a specific question, plus the question text."""
    data = _load_all_benchmarks()
    answers = [r for r in data["results"] if r.get("question_id") == question_id]

    # Try to get question text
    question_text = None
    for ds_file in QUESTIONS_DIR.glob("*.json"):
        if ds_file.name in ("manifest.json", "hard_analysis.json"):
            continue
        try:
            with open(ds_file, encoding="utf-8") as f:
                qs = json.load(f).get("questions", [])
            for q in qs:
                if q.get("question_id") == question_id:
                    question_text = q
                    break
            if question_text:
                break
        except Exception:
            pass

    return jsonify({
        "question_id": question_id,
        "question": question_text,
        "answers": answers,
    })


@app.route("/api/comparison")
def api_comparison():
    """Model comparison data for charts.

    Returns per-model-dataset summaries from benchmark results,
    human results, and CBM-paper results — all in one payload
    with model metadata (params_b, type).
    """
    # Model size metadata
    model_sizes = {}
    try:
        models_yaml = REPO_ROOT / "config" / "models.yaml"
        if models_yaml.exists():
            import yaml
            with open(models_yaml) as f:
                cfg = yaml.safe_load(f)
            for k, v in cfg.get("local_models", {}).items():
                model_sizes[k] = v.get("params_b", 0)
    except Exception:
        pass

    # API model approximate sizes (for chart positioning)
    api_sizes = {
        "gpt-4.1-nano": 5, "gpt-4.1-mini": 30, "gpt-4.1": 200,
        "gpt-4o-mini": 8, "gpt-4o": 200, "o3": 200, "o4-mini": 30,
        "claude-haiku-4-5-20251001": 8, "claude-sonnet-4-6": 70,
        "claude-sonnet-4-20250514": 70, "claude-opus-4-6": 200,
        "gemini-2.5-flash": 30, "gemini-2.5-flash-lite": 10,
        "gemini-2.5-pro": 200, "deepseek-chat": 70,
        "deepseek-reasoner": 70,
    }
    model_sizes.update(api_sizes)

    entries = []

    # Benchmark results (LLM)
    data = _load_all_benchmarks()
    from collections import defaultdict
    groups = defaultdict(list)
    for r in data["results"]:
        groups[(r.get("model_name"), r.get("dataset"), r.get("method"))].append(r)

    for (m, d, meth), rlist in groups.items():
        n = len(rlist)
        correct = sum(1 for r in rlist if r.get("is_correct"))
        confs = [r.get("confidence", 0) for r in rlist]
        hlccs = [r.get("hlcc_score", 0) for r in rlist]
        entries.append({
            "name": m, "dataset": d, "method": meth, "type": "llm",
            "params_b": model_sizes.get(m, 0),
            "n": n, "correct": correct,
            "accuracy": correct / n if n else 0,
            "mean_confidence": sum(confs) / n if n else 0,
            "mean_hlcc": sum(hlccs) / n if n else 0,
        })

    # Human results
    if HUMAN_RESULTS_DIR.exists():
        for fpath in HUMAN_RESULTS_DIR.glob("*.json"):
            try:
                with open(fpath, encoding="utf-8") as f:
                    hdata = json.load(f)
                for result in hdata.get("results", []):
                    met = result.get("metrics", {})
                    entries.append({
                        "name": result.get("participant", "?"),
                        "dataset": result.get("dataset", ""),
                        "method": "human",
                        "type": "human",
                        "params_b": 0,
                        "n": result.get("total_questions", 0),
                        "correct": int(met.get("accuracy", 0) *
                                       result.get("total_questions", 0)),
                        "accuracy": met.get("accuracy", 0),
                        "mean_confidence": met.get("mean_confidence", 0),
                        "mean_hlcc": result.get("mean_hlcc", 0),
                    })
            except Exception:
                pass

    return jsonify({"entries": entries})


@app.route("/api/abstention-bench")
def api_abstention_bench():
    """Serve abstention benchmark results."""
    path = QUESTIONS_DIR / "abstention_bench.json"
    if not path.exists():
        return jsonify({"error": "No abstention bench data found."}), 404
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


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
    """List available datasets with question counts and user progress."""
    datasets = _list_available_datasets()
    counts = _dataset_question_counts()

    user_id = session.get("user_id")
    progress = {}
    if user_id:
        for dp in db.get_user_dataset_progress(user_id):
            progress[dp["dataset"]] = dp

    result = []
    for ds in datasets:
        entry = {"name": ds, "total_questions": counts.get(ds, 0)}
        if ds in progress:
            entry["sessions_completed"] = progress[ds]["sessions"]
            entry["avg_accuracy"] = progress[ds]["avg_accuracy"]
            entry["avg_hlcc"] = progress[ds]["avg_hlcc"]
        result.append(entry)

    return jsonify({"datasets": result})


@app.route("/api/start", methods=["POST"])
def api_start():
    """Start a new quiz session.

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

    # Get or create user
    user_id = session.get("user_id")
    if not user_id:
        user_id = db.create_anonymous_user(participant)

    question_ids = [q["question_id"] for q in shuffled]
    db.create_session(session_id, user_id, dataset_name, max_questions, question_ids)

    return jsonify({
        "session_id": session_id,
        "dataset": dataset_name,
        "total_questions": len(shuffled),
    })


@app.route("/api/question/<session_id>")
def api_question(session_id):
    """Get the current question for a session."""
    sess = db.get_session(session_id)
    if not sess:
        return jsonify({"error": "Session not found"}), 404

    idx = sess["current_index"]
    q_ids = sess["question_ids"]

    if idx >= len(q_ids):
        return jsonify({"done": True, "message": "Quiz complete"})

    question_id = q_ids[idx]
    q = _get_question_by_id(sess["dataset"], question_id)
    if not q:
        return jsonify({"error": f"Question '{question_id}' not found in dataset"}), 500

    return jsonify({
        "question_number": idx + 1,
        "total_questions": len(q_ids),
        "question_id": q["question_id"],
        "question": q["question"],
        "choices": q["choices"],
        "subject": q.get("subject", ""),
        "dataset": sess["dataset"],
    })


@app.route("/api/answer", methods=["POST"])
def api_answer():
    """Submit an answer with confidence.

    Body: { "session_id": "...", "selected_answer": 0, "confidence": 0.8 }
    """
    data = request.json or {}
    session_id = data.get("session_id")
    sess = db.get_session(session_id)
    if not sess:
        return jsonify({"error": "Session not found"}), 404

    idx = sess["current_index"]
    q_ids = sess["question_ids"]

    if idx >= len(q_ids):
        return jsonify({"error": "Quiz already complete"}), 400

    question_id = q_ids[idx]
    q = _get_question_by_id(sess["dataset"], question_id)
    if not q:
        return jsonify({"error": f"Question '{question_id}' not found"}), 500

    selected = int(data.get("selected_answer", 0))
    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    # Check for human-specific correct answers (e.g. TruthfulQA consciousness questions)
    human_correct = q.get("correct_answer_human")
    if human_correct is not None:
        # correct_answer_human can be a single int or a list of ints
        if isinstance(human_correct, list):
            is_correct = selected in human_correct
        else:
            is_correct = selected == human_correct
        # For feedback, use the first human-correct answer as the "canonical" correct one
        effective_correct = human_correct[0] if isinstance(human_correct, list) else human_correct
    else:
        is_correct = selected == q["correct_answer"]
        effective_correct = q["correct_answer"]

    h_score = hlcc_score(confidence, is_correct)
    c_score = cbm_score(confidence, is_correct)

    # Save answer to database (store effective correct answer for human context)
    db.save_answer(
        session_id=session_id,
        question_id=question_id,
        question_index=idx,
        dataset=sess["dataset"],
        selected_answer=selected,
        correct_answer=effective_correct,
        is_correct=is_correct,
        confidence=confidence,
        hlcc_score=h_score,
        cbm_score=c_score,
    )
    db.advance_session(session_id)

    # Get running stats from DB
    stats = db.get_running_stats(session_id)

    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    response = {
        "question_id": question_id,
        "selected_answer": selected,
        "selected_letter": letters[selected] if selected < len(letters) else "?",
        "correct_answer": effective_correct,
        "correct_letter": letters[effective_correct] if effective_correct < len(letters) else "?",
        "correct_text": q["choices"][effective_correct] if effective_correct < len(q["choices"]) else "",
        "is_correct": is_correct,
        "confidence": confidence,
        "hlcc_score": h_score,
        "cbm_score": c_score,
        "timestamp": datetime.now().isoformat(),
        "running_accuracy": stats.get("accuracy", 0),
        "running_hlcc": stats.get("mean_hlcc", 0),
        "running_cbm": stats.get("mean_cbm", 0),
        "questions_remaining": len(q_ids) - (idx + 1),
    }

    return jsonify(response)


@app.route("/api/results/<session_id>")
def api_results(session_id):
    """Get full results for a completed session."""
    sess = db.get_session(session_id)
    if not sess:
        return jsonify({"error": "Session not found"}), 404

    answers = db.get_session_answers(session_id)
    if not answers:
        return jsonify({"error": "No responses yet"}), 400

    # Build response list with question text
    responses = []
    for a in answers:
        q = _get_question_by_id(sess["dataset"], a["question_id"])
        letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        responses.append({
            "question_id": a["question_id"],
            "selected_answer": a["selected_answer"],
            "selected_letter": letters[a["selected_answer"]] if a["selected_answer"] < len(letters) else "?",
            "correct_answer": a["correct_answer"],
            "correct_letter": letters[a["correct_answer"]] if a["correct_answer"] < len(letters) else "?",
            "correct_text": q["choices"][a["correct_answer"]] if q and a["correct_answer"] < len(q.get("choices", [])) else "",
            "is_correct": bool(a["is_correct"]),
            "confidence": a["confidence"],
            "hlcc_score": a["hlcc_score"],
            "cbm_score": a["cbm_score"],
        })

    metrics = compute_metrics(responses)
    hlcc_scores = [r["hlcc_score"] for r in responses]
    cbm_scores = [r["cbm_score"] for r in responses]

    # Generate share token and save result summary
    share_token = secrets.token_urlsafe(8)
    if not sess["is_complete"]:
        db.complete_session(session_id)

    db.save_session_result(
        session_id=session_id,
        user_id=sess["user_id"],
        dataset=sess["dataset"],
        total_questions=len(responses),
        accuracy=metrics["accuracy"],
        mean_confidence=metrics["mean_confidence"],
        mean_hlcc=sum(hlcc_scores) / len(hlcc_scores),
        mean_cbm=sum(cbm_scores) / len(cbm_scores),
        total_hlcc=sum(hlcc_scores),
        total_cbm=sum(cbm_scores),
        ece=metrics["ece"],
        brier=metrics["brier"],
        calibration_gap=metrics["calibration_gap"],
        share_token=share_token,
    )

    result = {
        "session_id": session_id,
        "dataset": sess["dataset"],
        "total_questions": len(responses),
        "completed": True,
        "started": sess["started_at"],
        "finished": datetime.now().isoformat(),
        "metrics": metrics,
        "mean_hlcc": sum(hlcc_scores) / len(hlcc_scores),
        "mean_cbm": sum(cbm_scores) / len(cbm_scores),
        "total_hlcc": sum(hlcc_scores),
        "total_cbm": sum(cbm_scores),
        "share_token": share_token,
        "responses": responses,
    }

    # Also save JSON for backward compatibility with git workflow
    _save_human_result(result)

    return jsonify(result)


# ============================================================
# ROUTES — SHARING
# ============================================================

@app.route("/api/shared/<share_token>")
def api_shared_result(share_token):
    """View shared quiz results (public, no auth required)."""
    sr = db.get_result_by_share_token(share_token)
    if not sr:
        return jsonify({"error": "Shared result not found"}), 404

    # Get the user's display name
    user = db.get_user_by_id(sr["user_id"]) if sr["user_id"] else None
    display_name = user["display_name"] if user else "Anonymous"

    # Get individual answers
    answers = db.get_session_answers(sr["session_id"])
    sess = db.get_session(sr["session_id"])

    responses = []
    for a in answers:
        q = _get_question_by_id(sr["dataset"], a["question_id"]) if sess else None
        letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        responses.append({
            "question_id": a["question_id"],
            "is_correct": bool(a["is_correct"]),
            "confidence": a["confidence"],
            "hlcc_score": a["hlcc_score"],
            "selected_letter": letters[a["selected_answer"]] if a["selected_answer"] < len(letters) else "?",
            "correct_letter": letters[a["correct_answer"]] if a["correct_answer"] < len(letters) else "?",
        })

    return jsonify({
        "display_name": display_name,
        "dataset": sr["dataset"],
        "total_questions": sr["total_questions"],
        "accuracy": sr["accuracy"],
        "mean_confidence": sr["mean_confidence"],
        "mean_hlcc": sr["mean_hlcc"],
        "ece": sr["ece"],
        "brier": sr["brier"],
        "calibration_gap": sr["calibration_gap"],
        "finished_at": sr["finished_at"],
        "responses": responses,
    })


# ============================================================
# ROUTES — LEADERBOARD
# ============================================================

@app.route("/api/leaderboard")
def api_leaderboard():
    """Combined leaderboard: LLM results (from aroma) + human results (from DB)."""
    dataset_filter = request.args.get("dataset")
    entries = []

    # LLM results — committed by aroma daemon
    if LEADERBOARD_FILE.exists():
        try:
            with open(LEADERBOARD_FILE, "r") as f:
                llm_data = json.load(f)
            for key, entry in llm_data.get("entries", {}).items():
                if dataset_filter and entry.get("dataset") != dataset_filter:
                    continue
                entry["type"] = "llm"
                entry["entry_key"] = key
                entries.append(entry)
        except Exception:
            pass

    # Human results from database
    human_entries = db.get_human_leaderboard(dataset=dataset_filter)
    for h in human_entries:
        entries.append({
            "type": "human",
            "entry_key": f"human_{h['session_id']}",
            "model_name": f"Human: {h.get('display_name', 'Anonymous')}",
            "dataset": h["dataset"],
            "accuracy": h["accuracy"],
            "mean_confidence": h["mean_confidence"],
            "mean_hlcc_score": h["mean_hlcc"],
            "calibration_gap": h["calibration_gap"],
            "ece": h["ece"],
            "total_examples": h["total_questions"],
            "updated": h["finished_at"],
            "share_token": h.get("share_token"),
        })

    entries.sort(key=lambda x: x.get("mean_hlcc_score", 0), reverse=True)
    return jsonify({"entries": entries})


# ============================================================
# ROUTES — USER PROFILE & HISTORY
# ============================================================

@app.route("/api/user/history")
def api_user_history():
    """Get quiz history for the logged-in user."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    history = db.get_user_history(user_id)
    return jsonify({"history": history})


@app.route("/api/user/stats")
def api_user_stats():
    """Get performance stats over time for the logged-in user."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    history = db.get_user_history(user_id)
    progress = db.get_user_dataset_progress(user_id)

    # Build time series
    time_series = []
    for h in reversed(history):  # oldest first
        time_series.append({
            "dataset": h["dataset"],
            "accuracy": h["accuracy"],
            "mean_hlcc": h["mean_hlcc"],
            "ece": h["ece"],
            "calibration_gap": h["calibration_gap"],
            "finished_at": h["finished_at"],
            "total_questions": h["total_questions"],
        })

    return jsonify({
        "datasets": progress,
        "time_series": time_series,
        "total_sessions": len(history),
        "total_questions": sum(h.get("total_questions", 0) for h in history),
    })


# ============================================================
# ROUTES — MODEL BENCHMARK UPLOAD
# ============================================================

@app.route("/api/upload_results", methods=["POST"])
def api_upload_results():
    """Upload AI model benchmark results (from desktop runner).

    Authenticated via ADMIN_PASSWORD or UPLOAD_API_KEY header.
    Body: { "run_id": "...", "model_name": "...", "dataset": "...",
            "summary": {...}, "results": [{...}, ...] }
    """
    # Auth check: header token or admin session
    api_key = request.headers.get("X-API-Key", "")
    admin_pw = os.environ.get("ADMIN_PASSWORD", "")
    upload_key = os.environ.get("UPLOAD_API_KEY", admin_pw)

    is_admin = session.get("admin_authenticated")
    if not is_admin and not (upload_key and api_key == upload_key):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json or {}
    run_id = data.get("run_id")
    model_name = data.get("model_name")
    dataset = data.get("dataset")
    summary = data.get("summary", {})
    results = data.get("results", [])

    if not all([run_id, model_name, dataset]):
        return jsonify({"error": "run_id, model_name, and dataset are required"}), 400

    # Save run summary
    db.save_model_run(
        run_id=run_id,
        model_name=model_name,
        dataset=dataset,
        total_questions=summary.get("total_examples", len(results)),
        accuracy=summary.get("accuracy", 0),
        mean_confidence=summary.get("mean_confidence", 0),
        mean_hlcc=summary.get("mean_hlcc_score", 0),
        mean_cbm=summary.get("mean_cbm_score", 0),
        total_hlcc=summary.get("total_hlcc", 0),
        total_cbm=summary.get("total_cbm", 0),
        ece=summary.get("ece", 0),
        brier=summary.get("brier", 0),
        calibration_gap=summary.get("calibration_gap", 0),
        method=data.get("method"),
        temperature=data.get("temperature"),
        run_timestamp=data.get("timestamp", datetime.now().isoformat()),
    )

    # Save per-question results
    if results:
        answers = []
        for r in results:
            answers.append({
                "run_id": run_id,
                "question_id": r.get("question_id", ""),
                "dataset": dataset,
                "model_name": model_name,
                "selected_answer": r.get("selected_answer", 0),
                "correct_answer": r.get("correct_answer", 0),
                "is_correct": r.get("is_correct", False),
                "confidence": r.get("confidence", 0.5),
                "hlcc_score": r.get("hlcc_score", 0),
                "cbm_score": r.get("cbm_score", 0),
                "processing_time": r.get("processing_time"),
                "temperature": r.get("temperature"),
            })
        db.save_model_answers_batch(answers)

    return jsonify({
        "success": True,
        "run_id": run_id,
        "questions_saved": len(results),
    })


# ============================================================
# ROUTES — QUESTION ANALYTICS
# ============================================================

@app.route("/api/questions/analytics")
def api_question_analytics():
    """Get question-level analytics across all models and humans."""
    dataset_filter = request.args.get("dataset")
    dynamics = db.get_question_dynamics(dataset=dataset_filter)
    return jsonify({"questions": dynamics})


@app.route("/api/questions/<question_id>/stats")
def api_question_detail(question_id):
    """Get detailed stats for a single question."""
    # Model stats
    model_stats = db.get_question_model_stats(question_id)
    # Human stats
    human_stats = db.get_question_stats(question_id)
    # Get question text
    q_text = None
    for ds in _list_available_datasets():
        q = _get_question_by_id(ds, question_id)
        if q:
            q_text = q
            break

    return jsonify({
        "question_id": question_id,
        "question": q_text,
        "model_stats": model_stats,
        "human_stats": human_stats,
    })


@app.route("/api/models/runs")
def api_model_runs():
    """Get all model benchmark runs."""
    dataset = request.args.get("dataset")
    model = request.args.get("model")
    runs = db.get_model_runs(dataset=dataset, model_name=model)
    return jsonify({"runs": runs})


@app.route("/api/models/runs/<run_id>/answers")
def api_model_run_answers(run_id):
    """Get per-question answers for a specific model run."""
    answers = db.get_model_answers_for_run(run_id)
    return jsonify({"answers": answers})


@app.route("/api/questions/hardest")
def api_hardest_questions():
    """Get the hardest questions (lowest accuracy across models)."""
    dataset = request.args.get("dataset")
    limit = int(request.args.get("limit", 50))
    questions = db.get_hardest_questions(dataset=dataset, limit=limit)

    # Enrich with question text
    for q in questions:
        q_data = _get_question_by_id(q["dataset"], q["question_id"])
        if q_data:
            q["question_text"] = q_data["question"][:120]
            q["choices"] = q_data.get("choices", [])
    return jsonify({"questions": questions})


# ============================================================
# BACKWARD COMPAT — JSON file saves
# ============================================================

def _save_human_result(result):
    """Save human quiz result to results/human/ (for git workflow compat)."""
    HUMAN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = result.get("dataset", "unknown")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sid = result.get("session_id", "unknown")
    filename = f"human_{sid}_{dataset}_{ts}.json"

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
    print(f"  Database:      {db.DB_PATH}")
    print(f"  Datasets:      {_list_available_datasets()}")
    print(f"  URL: http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
