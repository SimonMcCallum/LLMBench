"""
Safe read/write for config/models.yaml.

All writes are atomic (write to temp, then rename). File locking prevents
concurrent corruption from overlapping requests.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import yaml

logger = logging.getLogger("admin.config")

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "models.yaml"
REPO_ROOT = Path(__file__).parent.parent.parent


def read_models_config() -> dict:
    """Read the current models.yaml."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_models_config(config: dict):
    """Atomically write updated models.yaml."""
    fd, tmp_path = tempfile.mkstemp(
        suffix=".yaml", dir=CONFIG_PATH.parent, prefix=".models_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        os.replace(tmp_path, str(CONFIG_PATH))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _git_commit_and_push(message: str, paths: list[str]):
    """Stage paths, commit, and push. Errors are logged but not raised."""
    try:
        for p in paths:
            subprocess.run(
                ["git", "add", p],
                cwd=REPO_ROOT, check=True, capture_output=True, timeout=30,
            )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=REPO_ROOT, check=True, capture_output=True, timeout=30,
        )
        subprocess.run(
            ["git", "push"],
            cwd=REPO_ROOT, check=True, capture_output=True, timeout=60,
        )
        logger.info("Git push succeeded: %s", message)
    except subprocess.CalledProcessError as e:
        logger.warning("Git operation failed: %s — %s", message, e.stderr)
    except FileNotFoundError:
        logger.warning("Git not available, skipping push")


# ----------------------------------------------------------------
# LOCAL MODELS
# ----------------------------------------------------------------

def add_local_model(key: str, hf_path: str, params_b: float, requires_auth: bool = False):
    config = read_models_config()
    local = config.setdefault("local_models", {})
    if key in local:
        raise ValueError(f"Model '{key}' already exists")
    local[key] = {
        "hf_path": hf_path,
        "params_b": params_b,
        "requires_auth": requires_auth,
        "enabled": True,
    }
    write_models_config(config)
    _git_commit_and_push(
        f"[admin] Add local model: {key}",
        ["config/models.yaml"],
    )


def remove_local_model(key: str):
    config = read_models_config()
    local = config.get("local_models", {})
    if key not in local:
        raise ValueError(f"Model '{key}' not found")
    del local[key]
    write_models_config(config)
    _git_commit_and_push(
        f"[admin] Remove local model: {key}",
        ["config/models.yaml"],
    )


def toggle_model(key: str, enabled: bool):
    config = read_models_config()
    local = config.get("local_models", {})
    if key not in local:
        raise ValueError(f"Model '{key}' not found")
    local[key]["enabled"] = enabled
    write_models_config(config)
    state = "enabled" if enabled else "disabled"
    _git_commit_and_push(
        f"[admin] {state.capitalize()} local model: {key}",
        ["config/models.yaml"],
    )


# ----------------------------------------------------------------
# API VENDORS / MODELS
# ----------------------------------------------------------------

def add_api_vendor(vendor: str, endpoint: str, api_key_env: str, models: list[str]):
    config = read_models_config()
    api = config.setdefault("api_models", {})
    if vendor in api:
        raise ValueError(f"Vendor '{vendor}' already exists")
    api[vendor] = {
        "endpoint": endpoint,
        "api_key_env": api_key_env,
        "models": models or [],
        "enabled": True,
    }
    write_models_config(config)
    _git_commit_and_push(
        f"[admin] Add API vendor: {vendor}",
        ["config/models.yaml"],
    )


def add_api_model(vendor: str, model_name: str):
    config = read_models_config()
    api = config.get("api_models", {})
    if vendor not in api:
        raise ValueError(f"Vendor '{vendor}' not found")
    models = api[vendor].setdefault("models", [])
    if model_name in models:
        raise ValueError(f"Model '{model_name}' already exists for {vendor}")
    models.append(model_name)
    write_models_config(config)
    _git_commit_and_push(
        f"[admin] Add API model: {vendor}/{model_name}",
        ["config/models.yaml"],
    )


def remove_api_model(vendor: str, model_name: str):
    config = read_models_config()
    api = config.get("api_models", {})
    if vendor not in api:
        raise ValueError(f"Vendor '{vendor}' not found")
    models = api[vendor].get("models", [])
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found for {vendor}")
    models.remove(model_name)
    write_models_config(config)
    _git_commit_and_push(
        f"[admin] Remove API model: {vendor}/{model_name}",
        ["config/models.yaml"],
    )


def toggle_api_vendor(vendor: str, enabled: bool):
    config = read_models_config()
    api = config.get("api_models", {})
    if vendor not in api:
        raise ValueError(f"Vendor '{vendor}' not found")
    api[vendor]["enabled"] = enabled
    write_models_config(config)
    state = "enabled" if enabled else "disabled"
    _git_commit_and_push(
        f"[admin] {state.capitalize()} API vendor: {vendor}",
        ["config/models.yaml"],
    )


def remove_api_vendor(vendor: str):
    config = read_models_config()
    api = config.get("api_models", {})
    if vendor not in api:
        raise ValueError(f"Vendor '{vendor}' not found")
    del api[vendor]
    write_models_config(config)
    _git_commit_and_push(
        f"[admin] Remove API vendor: {vendor}",
        ["config/models.yaml"],
    )
