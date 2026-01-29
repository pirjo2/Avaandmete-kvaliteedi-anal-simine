from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import yaml

def load_vetro_yaml(path: str | Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    p = Path(path)
    y = yaml.safe_load(p.read_text(encoding="utf-8"))
    if "vetro_methodology" not in y:
        raise ValueError("Could not find 'vetro_methodology' in YAML.")
    return y["vetro_methodology"], y.get("meta", {})

def load_prompts_yaml(path: str | Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    p = Path(path)
    y = yaml.safe_load(p.read_text(encoding="utf-8"))
    return y.get("symbols", {}), y.get("meta", {})
