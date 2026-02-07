from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple
import re
import json

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# Strict answer/evidence lines
ANSWER_LINE_RE = re.compile(r"(?im)^\s*answer\s*[:=]\s*(.+?)\s*$")
EVIDENCE_LINE_RE = re.compile(r"(?im)^\s*evidence\s*[:=]\s*(.*)\s*$")

# Number parsing (supports 0.14 and 0,14)
NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

def _sanitize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\x00", "")
    s = re.sub(r"[\x01-\x08\x0B\x0C\x0E-\x1F]", " ", s)
    return s

def _parse_number(text: str) -> Optional[float]:
    if not text:
        return None
    m = NUMBER_RE.search(text)
    if not m:
        return None
    num = m.group(0).replace(",", ".")
    try:
        return float(num)
    except Exception:
        return None

def format_prompt(prompt_template: str, context: str, N: int) -> str:
    """
    Fix #2: stricter output contract so model doesn't copy '1 or 0' / 'YYYY-MM-DD'.
    """
    return (
        prompt_template.strip()
        + "\n\n--- CONTEXT START ---\n"
        + _sanitize_text(context).strip()
        + "\n--- CONTEXT END ---\n"
        + f"\nN={N}\n"
        + "\nOUTPUT RULES (MUST FOLLOW):\n"
        + "Return EXACTLY two lines and nothing else:\n"
        + "answer: <value>\n"
        + "evidence: <short quote or none>\n"
        + "\nWhere <value> is:\n"
        + "- for binary: 0 or 1\n"
        + "- for count: a number (0..N)\n"
        + "- for date: YYYY-MM-DD or UNKNOWN\n"
    )

@lru_cache(maxsize=4)
def get_hf_runner(model_name: str):
    """
    Transformers v5 safe: use AutoModel + generate (no pipeline task strings).
    Also: use_fast=False to avoid sentencepiece byte-fallback warning.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    kind = "seq2seq"
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception:
        kind = "causal"
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()

    def runner(prompt: str, max_new_tokens: int = 64) -> str:
        prompt = _sanitize_text(prompt)
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
                num_beams=1,
            )
            text = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

            if kind == "causal" and text.startswith(prompt):
                text = text[len(prompt):].strip()

            return text

    return runner

def _parse_value_and_confidence(generated_text: str, typ: str) -> Tuple[Optional[float | str], float]:
    """
    Returns (value, confidence). If parsing fails => (None, 0.0).
    IMPORTANT: None means "did not work", 0.0 means "worked and answer was 0".
    """
    txt = _sanitize_text(generated_text).strip()

    # Optional JSON format: {"value":..., "confidence":...}
    jm = JSON_RE.search(txt)
    if jm:
        try:
            obj = json.loads(jm.group(0))
            val = obj.get("value", None)
            conf = float(obj.get("confidence", 0.0) or 0.0)
            if val is None:
                return None, conf
            # date value from JSON
            if typ == "date":
                if isinstance(val, str) and val.strip().upper() == "UNKNOWN":
                    return None, conf
                m = DATE_RE.search(str(val))
                return (m.group(1) if m else None), conf
            return float(val), conf
        except Exception:
            pass

    # Strict answer line
    am = ANSWER_LINE_RE.search(txt)
    ans = am.group(1).strip() if am else ""

    # Guard against copied placeholders
    if ans.upper() in ("", "UNKNOWN") and typ == "date":
        return None, 0.6 if ans.upper() == "UNKNOWN" else 0.0

    if "YYYY-MM-DD" in ans and typ == "date":
        return None, 0.0

    # Avoid "1 or 0" / "0 or 1" style
    if re.search(r"\bor\b", ans.lower()) and ("0" in ans and "1" in ans):
        return None, 0.0

    if typ == "date":
        m = DATE_RE.search(ans) or DATE_RE.search(txt)
        if not m:
            return None, 0.0
        return m.group(1), 0.7 if am else 0.4

    # numeric types
    if am:
        v = _parse_number(ans)
        if v is None:
            return None, 0.0
        return v, 0.7

    # fallback: numeric-only whole output (rare)
    v = _parse_number(txt)
    if v is None:
        return None, 0.0
    return v, 0.4

def infer_symbol(
    symbol: str,
    context: str,
    N: int,
    prompt_defs: Dict[str, Any],
    hf_runner,
) -> Tuple[Optional[float | str], str, float]:
    """
    Fix #3: retry when parse/confidence fails.
    Returns (value or None, raw_text, confidence).
    None means: did not work. 0.0 means: worked and answer was 0.
    """
    cfg = prompt_defs.get(symbol)
    if cfg is None or hf_runner is None:
        return None, "", 0.0

    typ = cfg.get("type", "binary")
    prompt = format_prompt(cfg["prompt"], context, N)

    # 1st attempt
    raw1 = hf_runner(prompt, max_new_tokens=64)
    val1, conf1 = _parse_value_and_confidence(raw1, typ)

    # Retry if not parsed or too low confidence
    if val1 is None or conf1 <= 0.0:
        retry_prompt = (
            prompt
            + "\n\nFINAL REMINDER: Output ONLY two lines:\n"
            + "answer: <value>\n"
            + "evidence: <quote or none>\n"
        )
        raw2 = hf_runner(retry_prompt, max_new_tokens=48)
        val2, conf2 = _parse_value_and_confidence(raw2, typ)

        if (val2 is not None and conf2 >= conf1) or (val1 is None and val2 is not None):
            val1, conf1, raw1 = val2, conf2, raw2

    # Type handling
    if val1 is None:
        return None, raw1, conf1

    if typ == "binary":
        # keep 0/1 but accept float inputs
        v = 1.0 if float(val1) >= 0.5 else 0.0
        return v, raw1, conf1

    if typ in ("count_0_to_N", "count"):
        v = float(val1)
        v = max(0.0, min(float(N), v))
        return v, raw1, conf1

    # generic numeric / float (supports 0.14 etc)
    try:
        return float(val1), raw1, conf1
    except Exception:
        return None, raw1, 0.0
