from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Callable

import json
import re

# --- Regexid vastuse parsimiseks ---
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
ANSWER_LINE_RE = re.compile(r"(?im)^\s*answer\s*[:=]\s*(.+?)\s*$")
CONF_LINE_RE = re.compile(r"(?im)^\s*confidence\s*[:=]\s*([0-9]*\.?[0-9]+)\s*$")
EVID_LINE_RE = re.compile(r"(?im)^\s*evidence\s*[:=]\s*(.*)\s*$")
NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
YES_RE = re.compile(r"\b(yes|true)\b", re.IGNORECASE)
NO_RE = re.compile(r"\b(no|false)\b", re.IGNORECASE)


def _safe_format(template: str, values: Dict[str, Any]) -> str:
    """Format a prompt template safely, leaving unknown {keys} untouched."""
    class _Safe(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    try:
        return template.format_map(_Safe(values or {}))
    except Exception:
        return template


def format_prompt(prompt_template: str, context: str, values: Dict[str, Any]) -> str:
    rendered = _safe_format(prompt_template.strip(), values)
    return (
        rendered
        + "\n\n--- CONTEXT START ---\n"
        + context.strip()
        + "\n--- CONTEXT END ---\n"
        + "\nReturn ONLY in the exact format described (answer/confidence/evidence lines).\n"
    )


@lru_cache(maxsize=4)
def get_hf_runner(model_name: str) -> Callable[[str, int], str]:
    """
    Transformers v5-safe runner:
    - uses AutoTokenizer + AutoModelForSeq2SeqLM / AutoModelForCausalLM
    - no high-level pipeline strings
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    kind = "seq2seq"
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception:
        kind = "causal"
        model = AutoModelForCausalLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    def runner(prompt: str, max_new_tokens: int = 128) -> str:
        with torch.no_grad():
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
            text = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
            if kind == "causal" and text.startswith(prompt):
                text = text[len(prompt):].strip()
            return text

    return runner


# --- Abi-parsers Variant B jaoks ---


def _parse_binary(text: str) -> Tuple[Optional[float], float]:
    m = ANSWER_LINE_RE.search(text)
    if m:
        ans = m.group(1).strip()
        nm = NUM_RE.search(ans)
        if nm:
            return (1.0 if float(nm.group(0)) >= 1.0 else 0.0), 0.8
        if YES_RE.search(ans):
            return 1.0, 0.6
        if NO_RE.search(ans):
            return 0.0, 0.6
        return None, 0.0

    if YES_RE.search(text) and not NO_RE.search(text):
        return 1.0, 0.5
    if NO_RE.search(text) and not YES_RE.search(text):
        return 0.0, 0.5

    nm = NUM_RE.search(text)
    if nm:
        return (1.0 if float(nm.group(0)) >= 1.0 else 0.0), 0.5

    return None, 0.0


def _parse_count(text: str) -> Tuple[Optional[float], float]:
    m = ANSWER_LINE_RE.search(text)
    if m:
        ans = m.group(1).strip()
        nm = NUM_RE.search(ans)
        if nm:
            return float(nm.group(0)), 0.8
        return None, 0.0

    nm = NUM_RE.search(text)
    if nm:
        return float(nm.group(0)), 0.5

    return None, 0.0


def _parse_date(text: str) -> Tuple[Optional[str], float]:
    m = ANSWER_LINE_RE.search(text)
    if m:
        ans = m.group(1).strip()
        if ans.upper() == "UNKNOWN":
            return None, 0.8
        dm = DATE_RE.search(ans)
        if dm:
            return dm.group(1), 0.8

    dm = DATE_RE.search(text)
    if dm:
        return dm.group(1), 0.5

    if "UNKNOWN" in text.upper():
        return None, 0.5

    return None, 0.0


def _parse_json(text: str) -> Tuple[Optional[Any], Optional[float], Optional[str]]:
    jm = JSON_OBJ_RE.search(text)
    if not jm:
        return None, None, None
    try:
        obj = json.loads(jm.group(0))
    except Exception:
        return None, None, None

    ans = obj.get("answer", obj.get("value", None))
    conf = obj.get("confidence", None)
    evid = obj.get("evidence", None)

    try:
        conf_f = float(conf) if conf is not None else None
    except Exception:
        conf_f = None

    evid_s = str(evid).strip() if evid is not None else None
    return ans, conf_f, evid_s


def infer_symbol(
    symbol: str,
    context: str,
    N: int,
    prompt_defs: Dict[str, Any],
    hf_runner: Callable[[str, int], str] | None,
    extra_values: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float | str], str, float, str]:
    """
    LLM-põhine sümboli inferents (Variant B).
    Tagastab: (value_or_none, raw_text, confidence_0_1, evidence)
    """
    cfg = prompt_defs.get(symbol)
    if cfg is None or hf_runner is None:
        return None, "", 0.0, ""

    typ = cfg.get("type", "binary")
    prompt_template = str(cfg.get("prompt", ""))

    values: Dict[str, Any] = {"N": N}
    if extra_values:
        values.update(extra_values)

    full_prompt = format_prompt(prompt_template, context, values)
    raw = hf_runner(full_prompt, max_new_tokens=96)
    raw_str = str(raw or "").strip()

    # Evidence line
    em = EVID_LINE_RE.search(raw_str)
    evidence = (em.group(1).strip() if em else "")

    # JSON fallback
    j_ans, j_conf, j_evid = _parse_json(raw_str)
    if j_evid and not evidence:
        evidence = j_evid

    # Confidence line
    cm = CONF_LINE_RE.search(raw_str)
    conf = float(cm.group(1)) if cm else (j_conf if j_conf is not None else None)

    if typ == "date":
        val, base_conf = _parse_date(raw_str)
        if val is None and isinstance(j_ans, str):
            dm = DATE_RE.search(j_ans)
            val = dm.group(1) if dm else None
        final_conf = float(conf) if conf is not None else base_conf
        return val, raw_str, max(0.0, min(1.0, final_conf)), evidence

    if typ in ("count_0_to_N", "count"):
        val, base_conf = _parse_count(raw_str)
        if val is None and j_ans is not None:
            try:
                val = float(j_ans)
            except Exception:
                val = None
        if val is None:
            return None, raw_str, 0.0, evidence
        final_conf = float(conf) if conf is not None else base_conf
        return float(val), raw_str, max(0.0, min(1.0, final_conf)), evidence

    # binary
    val, base_conf = _parse_binary(raw_str)
    if val is None and j_ans is not None:
        try:
            val = 1.0 if float(j_ans) >= 1.0 else 0.0
        except Exception:
            if isinstance(j_ans, str) and YES_RE.search(j_ans):
                val = 1.0
            elif isinstance(j_ans, str) and NO_RE.search(j_ans):
                val = 0.0
            else:
                val = None

    if val is None:
        return None, raw_str, 0.0, evidence

    final_conf = float(conf) if conf is not None else base_conf
    return float(val), raw_str, max(0.0, min(1.0, final_conf)), evidence
