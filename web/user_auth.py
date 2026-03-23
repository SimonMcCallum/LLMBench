"""
Public user authentication — registration, login, profile.

Separate from web/admin/auth.py which handles the admin panel.
Uses bcrypt for password hashing and Flask sessions for state.
"""

import logging
import re
import time

import bcrypt
from flask import Blueprint, jsonify, request, session

from .database import (
    create_user,
    get_user_by_id,
    get_user_by_username,
    get_user_dataset_progress,
    get_user_history,
    update_last_seen,
)

logger = logging.getLogger("llmbench.auth")

auth_bp = Blueprint("user_auth", __name__, url_prefix="/api/auth")

# Rate limiting for login/register
_attempts = {}  # ip -> (count, first_attempt_time)
MAX_ATTEMPTS = 10
WINDOW_SECONDS = 300


def _rate_ok(ip):
    now = time.time()
    if ip in _attempts:
        count, first = _attempts[ip]
        if now - first > WINDOW_SECONDS:
            _attempts[ip] = (1, now)
            return True
        if count >= MAX_ATTEMPTS:
            return False
        _attempts[ip] = (count + 1, first)
    else:
        _attempts[ip] = (1, now)
    return True


def _validate_username(username):
    if not username or len(username) < 3 or len(username) > 30:
        return "Username must be 3-30 characters"
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        return "Username may only contain letters, numbers, hyphens, underscores"
    return None


def _validate_password(password):
    if not password or len(password) < 6:
        return "Password must be at least 6 characters"
    if len(password) > 128:
        return "Password too long"
    return None


def get_current_user_id():
    """Get user_id from Flask session, or None."""
    return session.get("user_id")


@auth_bp.route("/register", methods=["POST"])
def register():
    ip = request.headers.get("X-Real-IP", request.remote_addr)
    if not _rate_ok(ip):
        return jsonify({"error": "Too many attempts. Try again later."}), 429

    data = request.json or {}
    username = (data.get("username") or "").strip().lower()
    display_name = (data.get("display_name") or "").strip() or username
    password = data.get("password", "")

    err = _validate_username(username)
    if err:
        return jsonify({"error": err}), 400

    err = _validate_password(password)
    if err:
        return jsonify({"error": err}), 400

    email = (data.get("email") or "").strip()
    if not email or "@" not in email:
        return jsonify({"error": "A valid email address is required"}), 400

    if get_user_by_username(username):
        return jsonify({"error": "Username already taken"}), 409

    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    user_id = create_user(username, display_name, pw_hash, email=email)

    session["user_id"] = user_id
    session["username"] = username
    logger.info("User registered: %s (id=%d)", username, user_id)

    return jsonify({
        "success": True,
        "user_id": user_id,
        "username": username,
        "display_name": display_name,
    })


@auth_bp.route("/login", methods=["POST"])
def login():
    ip = request.headers.get("X-Real-IP", request.remote_addr)
    if not _rate_ok(ip):
        return jsonify({"error": "Too many attempts. Try again later."}), 429

    data = request.json or {}
    username = (data.get("username") or "").strip().lower()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    user = get_user_by_username(username)
    if not user or not user.get("password_hash"):
        return jsonify({"error": "Invalid username or password"}), 401

    if not bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
        return jsonify({"error": "Invalid username or password"}), 401

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    update_last_seen(user["id"])
    logger.info("User logged in: %s", username)

    return jsonify({
        "success": True,
        "user_id": user["id"],
        "username": user["username"],
        "display_name": user["display_name"],
    })


@auth_bp.route("/logout", methods=["POST"])
def logout():
    session.pop("user_id", None)
    session.pop("username", None)
    return jsonify({"success": True})


@auth_bp.route("/me", methods=["GET"])
def me():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"anonymous": True})

    user = get_user_by_id(user_id)
    if not user:
        session.clear()
        return jsonify({"anonymous": True})

    return jsonify({
        "anonymous": False,
        "user_id": user["id"],
        "username": user["username"],
        "display_name": user["display_name"],
        "created_at": user["created_at"],
    })


@auth_bp.route("/profile", methods=["GET"])
def profile():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    user = get_user_by_id(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    history = get_user_history(user_id)
    progress = get_user_dataset_progress(user_id)

    return jsonify({
        "user": {
            "user_id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
            "created_at": user["created_at"],
        },
        "history": history,
        "datasets": progress,
        "total_sessions": len(history),
        "total_questions": sum(h.get("total_questions", 0) for h in history),
    })
