from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML file and always return a dict (or {}).
    This function exists because some parts of the project (older pipeline versions)
    import `load_yaml` directly.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {p.as_posix()}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


# Backwards-compatible name used in some earlier commits
def load_yaml_file(path: str) -> Dict[str, Any]:
    return load_yaml(path)


def load_vetro_yaml(path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns: (vetro_methodology_dict, full_yaml_dict)
    """
    cfg = load_yaml(path)
    vetro = cfg.get("vetro_methodology") or {}
    if not isinstance(vetro, dict):
        vetro = {}
    return vetro, cfg


def load_prompts_yaml(path: str) -> Dict[str, Any]:
    cfg = load_yaml(path)
    symbols = cfg.get("symbols") or {}
    if not isinstance(symbols, dict):
        cfg["symbols"] = {}
    return cfg
