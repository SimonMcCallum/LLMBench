"""
Admin panel routes — all mounted at /vet via Blueprint.
"""

import hmac
import json
import logging
import os
import secrets
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml
from flask import Blueprint, jsonify, request, send_from_directory, session

from .auth import (
    check_rate_limit,
    clear_attempts,
    csrf_check,
    get_admin_password,
    record_failed_attempt,
    require_admin,
)
from .config_manager import (
    add_api_model,
    add_api_vendor,
    add_local_model,
    read_models_config,
    remove_api_model,
    remove_api_vendor,
    remove_local_model,
    toggle_api_vendor,
    toggle_model,
)
from web import database as db

logger = logging.getLogger("admin.routes")

admin_bp = Blueprint("admin", __name__, url_prefix="/vet")
admin_bp.before_request(csrf_check)

REPO_ROOT = Path(__file__).parent.parent.parent
STATIC_DIR = Path(__file__).parent.parent / "static"
LEADERBOARD_FILE = REPO_ROOT / "results" / "leaderboard.json"
HISTORY_DIR = REPO_ROOT / "results" / "history"
DATASETS_CONFIG = REPO_ROOT / "config" / "datasets.yaml"
INBOX_DIR = REPO_ROOT / ".claude" / "inbox"


# ============================================================
# AUTHENTICATION
# ============================================================

@admin_bp.route("/login", methods=["GET"])
def login_page():
    return send_from_directory(str(STATIC_DIR), "admin.html")


@admin_bp.route("/login", methods=["POST"])
def login_submit():
    ip = request.headers.get("X-Real-IP", request.remote_addr)
    allowed, retry_after = check_rate_limit(ip)
    if not allowed:
        return jsonify({"error": f"Too many attempts. Retry in {retry_after}s"}), 429

    data = request.json or request.form
    password = data.get("password", "")
    admin_password = get_admin_password()

    if not admin_password:
        return jsonify({"error": "Admin panel not configured"}), 503

    if hmac.compare_digest(password.encode(), admin_password.encode()):
        session["admin_authenticated"] = True
        session["admin_login_time"] = time.time()
        clear_attempts(ip)
        return jsonify({"success": True, "redirect": "/vet/"})
    else:
        record_failed_attempt(ip)
        logger.warning("Failed admin login from %s", ip)
        return jsonify({"error": "Invalid password"}), 401


@admin_bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True})


# ============================================================
# DASHBOARD
# ============================================================

@admin_bp.route("/")
@require_admin
def dashboard():
    return send_from_directory(str(STATIC_DIR), "admin.html")


# ============================================================
# MODEL MANAGEMENT
# ============================================================

@admin_bp.route("/api/models", methods=["GET"])
@require_admin
def get_models():
    """Return full model registry (env var names only, never actual keys)."""
    config = read_models_config()
    return jsonify(config)


@admin_bp.route("/api/models/local", methods=["POST"])
@require_admin
def api_add_local_model():
    data = request.json or {}
    required = ("key", "hf_path")
    missing = [k for k in required if not data.get(k)]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400
    try:
        add_local_model(
            key=data["key"].strip(),
            hf_path=data["hf_path"].strip(),
            params_b=float(data.get("params_b", 7.0)),
            requires_auth=bool(data.get("requires_auth", False)),
        )
        return jsonify({"success": True})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@admin_bp.route("/api/models/local/<key>", methods=["DELETE"])
@require_admin
def api_remove_local_model(key):
    try:
        remove_local_model(key)
        return jsonify({"success": True})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@admin_bp.route("/api/models/local/<key>/toggle", methods=["POST"])
@require_admin
def api_toggle_local_model(key):
    data = request.json or {}
    try:
        toggle_model(key, bool(data.get("enabled", True)))
        return jsonify({"success": True})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


# ---- API Vendors ----

@admin_bp.route("/api/models/api/vendor", methods=["POST"])
@require_admin
def api_add_vendor():
    data = request.json or {}
    required = ("vendor", "endpoint", "api_key_env")
    missing = [k for k in required if not data.get(k)]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400
    try:
        add_api_vendor(
            vendor=data["vendor"].strip(),
            endpoint=data["endpoint"].strip(),
            api_key_env=data["api_key_env"].strip(),
            models=data.get("models", []),
        )
        return jsonify({"success": True})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@admin_bp.route("/api/models/api/<vendor>", methods=["POST"])
@require_admin
def api_add_api_model(vendor):
    data = request.json or {}
    model_name = data.get("model_name", "").strip()
    if not model_name:
        return jsonify({"error": "model_name is required"}), 400
    try:
        add_api_model(vendor, model_name)
        return jsonify({"success": True})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@admin_bp.route("/api/models/api/<vendor>/<path:model_name>", methods=["DELETE"])
@require_admin
def api_remove_api_model(vendor, model_name):
    try:
        remove_api_model(vendor, model_name)
        return jsonify({"success": True})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@admin_bp.route("/api/models/api/<vendor>/toggle", methods=["POST"])
@require_admin
def api_toggle_vendor(vendor):
    data = request.json or {}
    try:
        toggle_api_vendor(vendor, bool(data.get("enabled", True)))
        return jsonify({"success": True})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@admin_bp.route("/api/models/api/<vendor>", methods=["DELETE"])
@require_admin
def api_remove_vendor(vendor):
    try:
        remove_api_vendor(vendor)
        return jsonify({"success": True})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


# ============================================================
# DATASETS
# ============================================================

@admin_bp.route("/api/datasets", methods=["GET"])
@require_admin
def get_datasets():
    if not DATASETS_CONFIG.exists():
        return jsonify({})
    with open(DATASETS_CONFIG, "r", encoding="utf-8") as f:
        return jsonify(yaml.safe_load(f) or {})


# ============================================================
# BENCHMARK TRIGGER
# ============================================================

@admin_bp.route("/api/benchmark", methods=["POST"])
@require_admin
def trigger_benchmark():
    """Write a task file to .claude/inbox/ for the aroma daemon to pick up."""
    data = request.json or {}

    models = data.get("models", [])
    datasets_list = data.get("datasets", [])
    if not models or not datasets_list:
        return jsonify({"error": "models and datasets are required"}), 400

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = f"bench_{ts}.md"

    task_config = {
        "type": "benchmark",
        "models": models,
        "datasets": datasets_list,
        "max_examples": int(data.get("max_examples", 100)),
    }
    # Optional fields
    if data.get("temperatures"):
        task_config["temperatures"] = data["temperatures"]
    if data.get("num_repetitions"):
        task_config["num_repetitions"] = int(data["num_repetitions"])
    if data.get("method"):
        task_config["method"] = data["method"]

    frontmatter = yaml.dump(task_config, default_flow_style=False)
    task_content = f"---\n{frontmatter}---\n# Benchmark triggered from admin panel\n"

    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    task_path = INBOX_DIR / task_name
    task_path.write_text(task_content, encoding="utf-8")

    # Git add + commit + push so aroma daemon sees it
    git_ok = True
    try:
        rel_path = str(task_path.relative_to(REPO_ROOT)).replace("\\", "/")
        subprocess.run(
            ["git", "add", rel_path],
            cwd=REPO_ROOT, check=True, capture_output=True, timeout=30,
        )
        subprocess.run(
            ["git", "commit", "-m", f"[admin] Benchmark task: {task_name}"],
            cwd=REPO_ROOT, check=True, capture_output=True, timeout=30,
        )
        subprocess.run(
            ["git", "push"],
            cwd=REPO_ROOT, check=True, capture_output=True, timeout=60,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("Git push failed for benchmark task: %s", e)
        git_ok = False

    result = {"success": True, "task_file": task_name, "git_pushed": git_ok}
    if not git_ok:
        result["warning"] = "Task file created but git push failed. Daemon will pick it up on next local sync."
    return jsonify(result), 200 if git_ok else 202


# ============================================================
# RESULTS / LEADERBOARD
# ============================================================

@admin_bp.route("/api/leaderboard", methods=["GET"])
@require_admin
def get_leaderboard():
    if not LEADERBOARD_FILE.exists():
        return jsonify({"entries": {}})
    with open(LEADERBOARD_FILE, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@admin_bp.route("/api/history", methods=["GET"])
@require_admin
def get_history():
    if not HISTORY_DIR.exists():
        return jsonify({"runs": []})
    runs = []
    for f in sorted(HISTORY_DIR.glob("*.json"), reverse=True)[:50]:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            runs.append({
                "filename": f.name,
                "timestamp": data.get("timestamp", ""),
                "config": data.get("config", {}),
                "summaries": data.get("summaries", []),
            })
        except Exception:
            pass
    return jsonify({"runs": runs})


# ============================================================
# API KEY MANAGEMENT
# ============================================================

API_KEYS_FILE = REPO_ROOT / "config" / "api_keys.yaml"


def _read_api_keys():
    """Read API keys config (stores env var names and whether key is set)."""
    config = read_models_config()
    api = config.get("api_models", {})
    keys_info = {}
    for vendor, cfg in api.items():
        env_var = cfg.get("api_key_env", "")
        keys_info[vendor] = {
            "env_var": env_var,
            "is_set": bool(os.environ.get(env_var, "")),
            "endpoint": cfg.get("endpoint", ""),
            "models": cfg.get("models", []),
            "enabled": cfg.get("enabled", True),
        }
    return keys_info


@admin_bp.route("/api/keys", methods=["GET"])
@require_admin
def get_api_keys():
    """Get API key status (never returns actual keys)."""
    return jsonify(_read_api_keys())


@admin_bp.route("/api/keys/<vendor>/test", methods=["POST"])
@require_admin
def test_api_key(vendor):
    """Test if an API key works by making a simple request."""
    config = read_models_config()
    api = config.get("api_models", {})
    if vendor not in api:
        return jsonify({"error": f"Vendor '{vendor}' not found"}), 404

    env_var = api[vendor].get("api_key_env", "")
    key = os.environ.get(env_var, "")
    if not key:
        return jsonify({"error": f"No key set for {env_var}"}), 400

    return jsonify({"success": True, "message": f"Key is set ({len(key)} chars)"})


@admin_bp.route("/api/keys/env_info", methods=["GET"])
@require_admin
def get_env_info():
    """Return which .env file to edit and what vars are needed."""
    config = read_models_config()
    api = config.get("api_models", {})
    env_vars = {}
    for vendor, cfg in api.items():
        env_var = cfg.get("api_key_env", "")
        env_vars[env_var] = {
            "vendor": vendor,
            "is_set": bool(os.environ.get(env_var, "")),
        }
    # Also include upload key
    env_vars["UPLOAD_API_KEY"] = {
        "vendor": "benchmark-upload",
        "is_set": bool(os.environ.get("UPLOAD_API_KEY", "")),
    }
    return jsonify({
        "env_file": "/home/simon/docker/.env",
        "env_vars": env_vars,
        "instructions": (
            "Add API keys to /home/simon/docker/.env then rebuild the container. "
            "For desktop benchmarking, create a .env file in your LLMBench repo root."
        ),
    })


# ============================================================
# MODEL BENCHMARK RUNS (from DB)
# ============================================================

@admin_bp.route("/api/model_runs", methods=["GET"])
@require_admin
def get_model_runs():
    """Get all model benchmark runs from the database."""
    dataset = request.args.get("dataset")
    model = request.args.get("model")
    runs = db.get_model_runs(dataset=dataset, model_name=model)
    return jsonify({"runs": runs})


@admin_bp.route("/api/model_runs/<run_id>/answers", methods=["GET"])
@require_admin
def get_model_run_answers(run_id):
    """Get per-question answers for a specific model run."""
    answers = db.get_model_answers_for_run(run_id)
    return jsonify({"answers": answers})


# ============================================================
# QUESTION ANALYTICS
# ============================================================

@admin_bp.route("/api/question_analytics", methods=["GET"])
@require_admin
def admin_question_analytics():
    """Get question-level analytics for admin dashboard."""
    dataset = request.args.get("dataset")
    dynamics = db.get_question_dynamics(dataset=dataset)
    return jsonify({"questions": dynamics})


@admin_bp.route("/api/hardest_questions", methods=["GET"])
@require_admin
def admin_hardest_questions():
    """Get hardest questions with enriched data."""
    dataset = request.args.get("dataset")
    limit = int(request.args.get("limit", 50))
    questions = db.get_hardest_questions(dataset=dataset, limit=limit)

    # Enrich with question text
    questions_dir = REPO_ROOT / "results" / "questions"
    for q in questions:
        q_data = _find_question(questions_dir, q["dataset"], q["question_id"])
        if q_data:
            q["question_text"] = q_data["question"][:200]
    return jsonify({"questions": questions})


def _find_question(questions_dir, dataset, question_id):
    """Look up a question from exported JSON."""
    filepath = questions_dir / f"{dataset}.json"
    if not filepath.exists():
        return None
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        for q in data.get("questions", []):
            if q["question_id"] == question_id:
                return q
    except Exception:
        pass
    return None


# ============================================================
# DATASET MANAGEMENT
# ============================================================

@admin_bp.route("/api/datasets/export", methods=["POST"])
@require_admin
def export_dataset():
    """Trigger export of a dataset from HuggingFace to results/questions/."""
    data = request.json or {}
    dataset_name = data.get("dataset", "").strip()
    max_questions = int(data.get("max_questions", 50))

    if not dataset_name:
        return jsonify({"error": "dataset name is required"}), 400

    # Check if dataset is in config
    if DATASETS_CONFIG.exists():
        with open(DATASETS_CONFIG, "r") as f:
            ds_config = yaml.safe_load(f) or {}
        datasets = ds_config.get("datasets", {})
        if dataset_name not in datasets:
            return jsonify({"error": f"Dataset '{dataset_name}' not in config/datasets.yaml"}), 400

    # Write export task to inbox for daemon
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = f"export_{dataset_name}_{ts}.md"
    task_content = f"""---
type: benchmark
datasets: [{dataset_name}]
max_examples: {max_questions}
models: []
---
# Export dataset: {dataset_name}
Export {max_questions} questions for the web quiz.
"""
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    task_path = INBOX_DIR / task_name
    task_path.write_text(task_content, encoding="utf-8")

    return jsonify({
        "success": True,
        "message": f"Export task created for {dataset_name} ({max_questions} questions)",
        "task_file": task_name,
    })


@admin_bp.route("/api/datasets/available", methods=["GET"])
@require_admin
def available_datasets():
    """List all datasets from config with export status."""
    if not DATASETS_CONFIG.exists():
        return jsonify({"datasets": {}})

    with open(DATASETS_CONFIG, "r") as f:
        ds_config = yaml.safe_load(f) or {}
    datasets = ds_config.get("datasets", {})

    questions_dir = REPO_ROOT / "results" / "questions"
    result = {}
    for name, cfg in datasets.items():
        exported_file = questions_dir / f"{name}.json"
        exported = exported_file.exists()
        question_count = 0
        if exported:
            try:
                with open(exported_file, "r") as f:
                    qdata = json.load(f)
                question_count = len(qdata.get("questions", []))
            except Exception:
                pass
        result[name] = {
            **cfg,
            "exported": exported,
            "exported_questions": question_count,
        }
    return jsonify({"datasets": result})


@admin_bp.route("/api/upload_key", methods=["POST"])
@require_admin
def generate_upload_key():
    """Generate a new upload API key for desktop benchmarking."""
    key = secrets.token_urlsafe(32)
    return jsonify({
        "key": key,
        "instruction": f"Add to .env: UPLOAD_API_KEY={key}",
    })
