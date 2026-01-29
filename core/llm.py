from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Tuple, Optional
import re

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
INT_RE = re.compile(r"[-+]?\d+")

def format_prompt(prompt_template: str, context: str, N: int) -> str:
    return (
        prompt_template.strip()
        + "\n\n--- CONTEXT START ---\n"
        + context.strip()
        + "\n--- CONTEXT END ---\n"
        + f"\nN={N}\n"
        + "Return only in the requested output format."
    )

@lru_cache(maxsize=2)
def get_hf_pipe(model_name: str):
    # Import lazily so running without LLM deps still works.
    from transformers import pipeline  # type: ignore
    return pipeline("text2text-generation", model=model_name, device_map="auto")

def infer_symbol(
    symbol: str,
    context: str,
    N: int,
    prompt_defs: Dict[str, Any],
    hf_pipe,
) -> Tuple[Optional[float | str], Optional[str]]:
    cfg = prompt_defs.get(symbol)
    if cfg is None or hf_pipe is None:
        return None, None

    prompt = format_prompt(cfg["prompt"], context, N)
    out = hf_pipe(prompt, max_new_tokens=96)[0].get("generated_text", "").strip()

    typ = cfg.get("type", "binary")
    if typ == "date":
        m = DATE_RE.search(out)
        if not m or m.group(1).upper() == "UNKNOWN":
            return None, out
        return m.group(1), out

    m = INT_RE.search(out)
    if not m:
        return None, out

    val = float(m.group(0))

    if typ == "binary":
        val = 1.0 if val >= 1 else 0.0
        return val, out

    if typ == "count_0_to_N":
        val = max(0.0, min(float(N), val))
        return val, out

    return val, out
