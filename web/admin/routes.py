"""
Admin panel routes — all mounted at /vet via Blueprint.
"""

import hmac
import json
import logging
import os
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
