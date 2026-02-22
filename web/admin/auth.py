"""
Admin authentication — single password from ADMIN_PASSWORD env var.

Provides:
  - @require_admin decorator (session check + timeout)
  - Login rate limiting (per-IP, in-memory)
  - CSRF check via X-Requested-With header
"""

import functools
import hmac
import logging
import os
import time

from flask import jsonify, redirect, request, session

logger = logging.getLogger("admin.auth")

SESSION_TIMEOUT = 3600  # 1 hour
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION = 300  # 5 minutes

# In-memory rate limiter: { ip: { "count": int, "locked_until": float } }
_login_attempts: dict = {}


def get_admin_password() -> str:
    return os.environ.get("ADMIN_PASSWORD", "")


def require_admin(f):
    """Decorator: enforce admin authentication with session timeout."""

    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not get_admin_password():
            return jsonify({"error": "Admin panel not configured"}), 503

        if not session.get("admin_authenticated"):
            if request.path.startswith("/vet/api/"):
                return jsonify({"error": "Not authenticated"}), 401
            return redirect("/vet/login")

        login_time = session.get("admin_login_time", 0)
        if time.time() - login_time > SESSION_TIMEOUT:
            session.clear()
            if request.path.startswith("/vet/api/"):
                return jsonify({"error": "Session expired"}), 401
            return redirect("/vet/login")

        return f(*args, **kwargs)

    return decorated


def check_rate_limit(ip: str) -> tuple[bool, int]:
    """Returns (allowed, retry_after_seconds)."""
    info = _login_attempts.get(ip)
    if not info:
        return True, 0
    locked_until = info.get("locked_until", 0)
    if time.time() < locked_until:
        return False, int(locked_until - time.time())
    return True, 0


def record_failed_attempt(ip: str):
    """Record a failed login; lock out after MAX_LOGIN_ATTEMPTS."""
    info = _login_attempts.setdefault(ip, {"count": 0, "locked_until": 0})
    info["count"] += 1
    if info["count"] >= MAX_LOGIN_ATTEMPTS:
        info["locked_until"] = time.time() + LOCKOUT_DURATION
        info["count"] = 0
        logger.warning("IP %s locked out for %ds", ip, LOCKOUT_DURATION)


def clear_attempts(ip: str):
    _login_attempts.pop(ip, None)


def csrf_check():
    """Blueprint before_request hook: verify X-Requested-With on mutations."""
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        # Login endpoint is exempt (form submission)
        if request.path == "/vet/login":
            return None
        if request.headers.get("X-Requested-With") != "XMLHttpRequest":
            return jsonify({"error": "CSRF check failed"}), 403
    return None
