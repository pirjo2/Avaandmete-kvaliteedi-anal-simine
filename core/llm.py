from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple
import re
import json

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# Very important: avoid "AgeGroup1" => 1 mistakes.
# This matches ONLY if the whole output is a number.
NUM_ONLY_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*$", re.DOTALL)

# If prompts include "answer: 0" style outputs (your current prompts do),
# parse that safely.
ANSWER_RE = re.compile(r"\banswer\s*[:=]\s*([01])\b", re.IGNORECASE)

def format_prompt(prompt_template: str, context: str, N: int) -> str:
    # Keep it short. Context is injected by caller.
    return (
        prompt_template.strip()
        + "\n\n--- CONTEXT START ---\n"
        + context.strip()
        + "\n--- CONTEXT END ---\n"
        + f"\nN={N}\n"
        + "\nReturn ONLY the answer in the requested format. No extra text."
    )

@lru_cache(maxsize=8)
def get_hf_pipe(model_name: str):
    """
    Model adapter:
    - T5/mT5 style models: text2text-generation
    - Others (instruct/causal): text-generation
    """
    from transformers import pipeline  # type: ignore

    name = model_name.lower()
    if "t5" in name or "mt5" in name:
        task = "text2text-generation"
    else:
        task = "text-generation"

    # Streamlit Cloud is typically CPU
    return pipeline(task, model=model_name, device=-1)

def _extract_generated_text(out: Any, prompt: str) -> str:
    """
    pipelines usually return list[dict] with 'generated_text'.
    For text-generation, the output often includes the prompt; strip it if present.
    """
    if isinstance(out, list) and out and isinstance(out[0], dict):
        txt = str(out[0].get("generated_text", "")).strip()
    else:
        txt = str(out).strip()

    # If model echoes prompt, remove it (common with text-generation)
    if txt.startswith(prompt):
        txt = txt[len(prompt):].strip()

    return txt

def _parse_value_and_confidence(generated_text: str, typ: str) -> Tuple[Optional[float | str], float]:
    """
    Supports:
    1) JSON: {"value":..., "confidence":...}
    2) Old prompt format: "answer: 0" + evidence
    3) Strict numeric-only output (whole text is a number)
    4) Date extraction for typ=='date'
    """
    confidence = 0.0

    if typ == "date":
        m = DATE_RE.search(generated_text)
        if not m or generated_text.strip().upper() == "UNKNOWN":
            return None, 0.0
        return m.group(1), 0.8

    # Try JSON first
    jm = JSON_RE.search(generated_text)
    if jm:
        try:
            obj = json.loads(jm.group(0))
            val = obj.get("value", None)
            conf = obj.get("confidence", 0.0)
            try:
                confidence = float(conf)
            except Exception:
                confidence = 0.0
            if val is None:
                return None, confidence
            # normalize numeric
            try:
                return float(val), confidence
            except Exception:
                return None, confidence
        except Exception:
            pass

    # Try "answer: 0/1" (your current prompt pack uses this)
    am = ANSWER_RE.search(generated_text)
    if am:
        val = float(am.group(1))
        return val, 0.6  # parsed from structured output, medium confidence

    # Strict numeric-only
    nm = NUM_ONLY_RE.match(generated_text)
    if nm:
        val = float(nm.group(1))
        return val, 0.8

    return None, 0.0

def infer_symbol(
    symbol: str,
    context: str,
    N: int,
    prompt_defs: Dict[str, Any],
    hf_pipe,
) -> Tuple[Optional[float | str], str, float]:
    """
    Returns: (value_or_None, raw_text, confidence)
    - IMPORTANT: binary/count symbols fall back to 0.0 so formulas still run.
    """
    cfg = prompt_defs.get(symbol)
    if cfg is None or hf_pipe is None:
        return None, "", 0.0

    typ = cfg.get("type", "binary")
    prompt = format_prompt(cfg["prompt"], context, N)

    out = hf_pipe(
        prompt,
        truncation=True,
        max_new_tokens=32,  # allow JSON, but keep small
        do_sample=False,
        # temperature not needed when do_sample=False (and may be ignored)
    )

    generated_text = _extract_generated_text(out, prompt)

    parsed_val, conf = _parse_value_and_confidence(generated_text, typ)

    # Fallbacks so computations keep working:
    if parsed_val is None:
        if typ in ("binary", "count_0_to_N", "count"):
            return 0.0, generated_text, 0.0
        return None, generated_text, 0.0

    # Clamp by type
    if typ == "binary":
        v = 1.0 if float(parsed_val) >= 1 else 0.0
        return v, generated_text, conf

    if typ in ("count_0_to_N", "count"):
        v = float(parsed_val)
        v = max(0.0, min(float(N), v))
        return v, generated_text, conf

    # default numeric
    return float(parsed_val), generated_text, conf