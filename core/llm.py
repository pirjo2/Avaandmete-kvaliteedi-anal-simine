from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple
import re
import datetime as _dt

DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
ANSWER_LINE_RE = re.compile(r"(?im)^\s*answer\s*[:=]\s*(.+?)\s*$")
CONF_LINE_RE = re.compile(r"(?im)^\s*confidence\s*[:=]\s*([0-9]*\.?[0-9]+)\s*$")
EVID_LINE_RE = re.compile(r"(?im)^\s*evidence\s*[:=]\s*(.*)\s*$")
NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
YES_RE = re.compile(r"\b(yes|true)\b", re.IGNORECASE)
NO_RE = re.compile(r"\b(no|false)\b", re.IGNORECASE)
PLACEHOLDER_DATE_RE = re.compile(r"YYYY-MM-DD", re.IGNORECASE)


def _safe_format(template: str, values: Dict[str, Any]) -> str:
    class _Safe(dict):
        def __missing__(self, key):
            return ""

    try:
        return template.format_map(_Safe(values))
    except Exception:
        return template


def _truncate_text(s: str, max_chars: int) -> str:
    if not s:
        return ""
    s = s.strip()
    if len(s) <= max_chars:
        return s
    head = s[: int(max_chars * 0.75)]
    tail = s[-int(max_chars * 0.25):]
    return head.rstrip() + "\n...\n" + tail.lstrip()


def format_prompt(prompt_template: str, context: str, values: Dict[str, Any], max_context_chars: int = 3500) -> str:
    # Put instructions BEFORE context so tokenizer truncation keeps the format constraints.
    rendered = _safe_format(str(prompt_template or "").strip(), values)
    ctx = _truncate_text(context, max_context_chars)
    return (
        rendered
        + "\n\n--- CONTEXT START ---\n"
        + ctx
        + "\n--- CONTEXT END ---\n"
    )


@lru_cache(maxsize=4)
def get_hf_runner(model_name: str):
    """
    Transformers v4/v5-safe runner: AutoModel + generate (no pipeline tasks).
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

    model.eval()

    def runner(prompt: str, max_new_tokens: int = 96) -> str:
        with torch.no_grad():
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            text = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

            if kind == "causal" and text.startswith(prompt):
                text = text[len(prompt):].strip()
            return text

    return runner


def _parse_confidence(raw: str) -> float:
    cm = CONF_LINE_RE.search(raw or "")
    if not cm:
        return 0.0
    try:
        c = float(cm.group(1))
        if c != c:
            return 0.0
        return max(0.0, min(1.0, c))
    except Exception:
        return 0.0


def _parse_evidence(raw: str) -> str:
    em = EVID_LINE_RE.search(raw or "")
    return (em.group(1).strip() if em else "")


def _answer_text(raw: str) -> str:
    m = ANSWER_LINE_RE.search(raw or "")
    if m:
        return (m.group(1) or "").strip()
    raw = (raw or "").strip()
    return raw.splitlines()[0].strip() if raw else ""


def _looks_like_ambiguous_choice(ans: str) -> bool:
    a = (ans or "").lower()
    return (" or " in a and "0" in a and "1" in a) or ("0/1" in a) or ("1/0" in a)


def _parse_binary(ans: str) -> Optional[float]:
    if not ans:
        return None
    if _looks_like_ambiguous_choice(ans):
        return None

    nm = NUM_RE.search(ans)
    if nm:
        try:
            return 1.0 if float(nm.group(0)) >= 1 else 0.0
        except Exception:
            pass

    # fallback: yes/no
    if YES_RE.search(ans) and not NO_RE.search(ans):
        return 1.0
    if NO_RE.search(ans) and not YES_RE.search(ans):
        return 0.0

    return None


def _parse_count(ans: str) -> Optional[float]:
    if not ans:
        return None
    if _looks_like_ambiguous_choice(ans):
        return None
    nm = NUM_RE.search(ans)
    if not nm:
        return None
    try:
        return float(nm.group(0))
    except Exception:
        return None


def _parse_date(ans: str) -> Optional[str]:
    if not ans:
        return None
    if ans.strip().upper() == "UNKNOWN":
        return None
    if PLACEHOLDER_DATE_RE.search(ans):
        return None
    dm = DATE_RE.search(ans)
    return dm.group(1) if dm else None


def infer_symbol(
    symbol: str,
    context: str,
    N: int,
    prompt_defs: Dict[str, Any],
    hf_runner,
    extra_values: Dict[str, Any],
) -> Tuple[Optional[float | str], str, float, str]:
    """
    Returns: (value_or_none, raw_text, confidence_0_to_1, evidence_string)

    None = did not work
    0.0 = real zero
    """
    cfg = prompt_defs.get(symbol)
    if cfg is None or hf_runner is None:
        return None, "", 0.0, ""

    typ = str(cfg.get("type", "binary"))
    prompt = format_prompt(
        prompt_template=str(cfg.get("prompt", "")),
        context=context,
        values={**(extra_values or {}), "N": N},
        max_context_chars=int((extra_values or {}).get("max_context_chars", 3500)),
    )

    raw = hf_runner(prompt, max_new_tokens=96)
    raw_str = str(raw or "").strip()

    ans_text = _answer_text(raw_str)
    conf = _parse_confidence(raw_str)
    evid = _parse_evidence(raw_str)

    # If confidence missing but answer parseable, add conservative base confidence
    if conf == 0.0:
        if typ == "date" and _parse_date(ans_text) is not None:
            conf = 0.5
        elif typ in ("count_0_to_N", "count") and _parse_count(ans_text) is not None:
            conf = 0.5
        elif typ == "binary" and _parse_binary(ans_text) is not None:
            conf = 0.45

    if typ == "date":
        return _parse_date(ans_text), raw_str, conf, evid

    if typ in ("count_0_to_N", "count"):
        v = _parse_count(ans_text)
        if v is None:
            return None, raw_str, 0.0, evid
        v = max(0.0, min(float(N), float(v)))
        return float(v), raw_str, conf, evid

    v = _parse_binary(ans_text)
    if v is None:
        return None, raw_str, 0.0, evid
    return float(v), raw_str, conf, evid


def date_to_ordinal(value: Any) -> Optional[float]:
    """
    Convert ISO date / datetime / pandas timestamp -> float ordinal.
    Returns None if cannot parse.
    """
    try:
        import pandas as pd
    except Exception:
        pd = None

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, _dt.datetime):
        return float(value.date().toordinal())
    if isinstance(value, _dt.date):
        return float(value.toordinal())

    if pd is not None:
        if isinstance(value, getattr(pd, "Timestamp", ())):
            return float(value.date().toordinal())

    if isinstance(value, str):
        m = DATE_RE.search(value)
        if not m:
            return None
        try:
            d = _dt.date.fromisoformat(m.group(1))
            return float(d.toordinal())
        except Exception:
            return None

    return None
