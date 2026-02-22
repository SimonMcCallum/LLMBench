"""
Unified Model Loader — Local HuggingFace models and Cloud API clients

Provides:
  - load_local_model(): Load a local HuggingFace model with auto-precision
  - get_api_client(): Get an API client for cloud models
  - list_available_models(): List all configured models
  - unload_model(): Free GPU memory

Adapted from NNCONFIDENCE/model_loader.py with added API model support.
"""

import gc
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import yaml

# ============================================================
# CONFIG LOADING
# ============================================================

_CONFIG_DIR = Path(__file__).parent.parent / "config"


def _load_config(name: str) -> dict:
    path = _CONFIG_DIR / f"{name}.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_models_config() -> dict:
    return _load_config("models")


def get_machine_config() -> dict:
    return _load_config("machine")


# ============================================================
# LOCAL MODEL LOADING
# ============================================================

def get_model_hf_path(model_key: str) -> str:
    """Resolve a model key to its HuggingFace path."""
    config = get_models_config()
    local = config.get("local_models", {})
    if model_key in local:
        return local[model_key]["hf_path"]
    return model_key  # Assume it's already a HF path


def get_model_params_b(model_key: str) -> float:
    """Get approximate parameter count in billions."""
    config = get_models_config()
    local = config.get("local_models", {})
    if model_key in local:
        return local[model_key].get("params_b", 14.0)
    return 14.0  # Conservative default


def get_recommended_precision(model_key: str, vram_gb: float = 32.0) -> str:
    """Select precision based on model size and available VRAM."""
    params_b = get_model_params_b(model_key)
    fp16_vram = params_b * 2.0
    usable_vram = vram_gb * 0.75

    if fp16_vram <= usable_vram:
        return "fp16"
    elif params_b <= usable_vram:
        return "8bit"
    else:
        return "4bit"


def get_cache_dir() -> str:
    """Get the model cache directory from machine config."""
    try:
        config = get_machine_config()
        return config.get("model_cache_dir", "data/models")
    except FileNotFoundError:
        return "data/models"


def load_local_model(
    model_key: str,
    precision: str = "auto",
    cache_dir: Optional[str] = None,
    device_map: str = "auto",
) -> Tuple[Any, Any, Dict]:
    """
    Load a local HuggingFace model and tokenizer.

    Args:
        model_key: Short key (e.g. "qwen2.5-7b") or HuggingFace path
        precision: "auto", "fp16", "8bit", or "4bit"
        cache_dir: Model cache directory (default from machine.yaml)
        device_map: Device placement strategy

    Returns:
        (model, tokenizer, info_dict)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if cache_dir is None:
        cache_dir = get_cache_dir()

    model_path = get_model_hf_path(model_key)

    if precision == "auto":
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            vram_gb = 0
        actual_precision = get_recommended_precision(model_key, vram_gb)
    else:
        actual_precision = precision

    print(f"Loading {model_key} ({model_path}) at {actual_precision} precision...")

    quant_config = None
    torch_dtype = None

    if actual_precision == "fp16":
        torch_dtype = torch.float16
    elif actual_precision == "8bit":
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif actual_precision == "4bit":
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir=cache_dir, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "cache_dir": cache_dir,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    vram_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    info = {
        "precision": actual_precision,
        "vram_gb": round(vram_used, 2),
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "model_path": model_path,
        "model_key": model_key,
        "type": "local",
    }

    print(f"  Loaded: {n_layers} layers, hidden_size={hidden_size}")
    print(f"  Precision: {actual_precision}, VRAM: {vram_used:.2f} GB")

    return model, tokenizer, info


def unload_model(model=None):
    """Free GPU memory."""
    import torch
    if model is not None:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ============================================================
# API MODEL CLIENTS
# ============================================================

def get_api_client(vendor: str) -> Optional[Dict]:
    """
    Get API client configuration for a cloud vendor.

    Returns dict with: endpoint, api_key, models list, or None if no key.
    """
    config = get_models_config()
    api_config = config.get("api_models", {}).get(vendor)
    if not api_config:
        return None

    api_key = os.environ.get(api_config["api_key_env"], "")
    if not api_key:
        return None

    return {
        "vendor": vendor,
        "endpoint": api_config["endpoint"],
        "api_key": api_key,
        "models": api_config["models"],
    }


def list_available_models() -> Dict[str, list]:
    """List all available models grouped by type (respects enabled field)."""
    config = get_models_config()

    local = config.get("local_models", {})
    result = {
        "local": [k for k, v in local.items() if v.get("enabled", True)],
        "api": {},
    }

    for vendor, vendor_config in config.get("api_models", {}).items():
        if not vendor_config.get("enabled", True):
            continue
        api_key = os.environ.get(vendor_config["api_key_env"], "")
        if api_key:
            result["api"][vendor] = vendor_config["models"]

    return result


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    available = list_available_models()
    print("Available models:")
    print(f"\nLocal ({len(available['local'])}):")
    for m in available["local"]:
        params = get_model_params_b(m)
        prec = get_recommended_precision(m)
        print(f"  {m:25s} {params:5.1f}B  ({prec})")

    print(f"\nAPI:")
    for vendor, models in available["api"].items():
        print(f"  {vendor}: {', '.join(models)}")
    if not available["api"]:
        print("  (no API keys configured)")
