from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Union
import json
import re

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
NUM_ONLY_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*$", re.DOTALL)
ANSWER_RE = re.compile(r"(?im)^\s*answer\s*[:=]\s*([^\n]+)\s*$")
EVIDENCE_RE = re.compile(r"(?im)^\s*evidence\s*[:=]\s*(.*)\s*$")

YES_RE = re.compile(r"^\s*(yes|true)\s*$", re.IGNORECASE)
NO_RE = re.compile(r"^\s*(no|false)\s*$", re.IGNORECASE)


def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _extract_first_number(text: str) -> Optional[float]:
    m = re.search(r"[-+]?\d*\.?\d+", text)
    if not m:
        return None
    return _to_float(m.group(0))


def format_prompt(prompt_template: str, context: str, N: int) -> str:
    return (
        prompt_template.strip()
        + "\n\n--- CONTEXT START ---\n"
        + context.strip()
        + "\n--- CONTEXT END ---\n"
        + f"\nN={N}\n"
        + "\nReturn ONLY the answer in the requested format. No extra text."
    )


@lru_cache(maxsize=4)
def get_hf_runner(model_name: str):
    """
    Transformers v5-safe: create a lightweight generate() runner
    (avoids pipeline task-string incompatibilities).

    Returns: runner(prompt, max_new_tokens=64) -> generated_text
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

    def runner(prompt: str, max_new_tokens: int = 64) -> str:
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


def _parse_answer_line(text: str) -> str:
    m = ANSWER_RE.search(text)
    if not m:
        return ""
    return m.group(1).strip()


def _parse_value_and_confidence(
    generated_text: str,
    typ: str,
    N: int,
) -> Tuple[Optional[Union[float, str]], float, str]:
    evidence = ""
    em = EVIDENCE_RE.search(generated_text or "")
    if em:
        evidence = em.group(1).strip()

    gt = (generated_text or "").strip()

    # 1) answer: ...
    ans = _parse_answer_line(gt)
    if ans:
        if typ == "date":
            m = DATE_RE.search(ans)
            if m:
                return m.group(1), 0.85, evidence
            if ans.strip().upper() == "UNKNOWN":
                return None, 0.0, evidence
            m2 = DATE_RE.search(gt)
            if m2:
                return m2.group(1), 0.75, evidence
            return None, 0.0, evidence

        if typ == "binary":
            if YES_RE.match(ans):
                return 1.0, 0.7, evidence
            if NO_RE.match(ans):
                return 0.0, 0.7, evidence
            num = _extract_first_number(ans)
            if num is None:
                return None, 0.0, evidence
            return (1.0 if num >= 1 else 0.0), 0.85, evidence

        num = _extract_first_number(ans)
        if num is None:
            return None, 0.0, evidence
        if typ in ("count_0_to_N", "count"):
            num = float(int(round(num)))
            num = max(0.0, min(float(N), num))
            return num, 0.85, evidence
        return float(num), 0.85, evidence

    # 2) JSON
    jm = JSON_RE.search(gt)
    if jm:
        try:
            obj = json.loads(jm.group(0))
            if evidence == "" and isinstance(obj, dict) and obj.get("evidence"):
                evidence = str(obj.get("evidence")).strip()
            val = obj.get("value", None)
            conf = obj.get("confidence", 0.0)
            conf_f = float(conf) if conf is not None else 0.0

            if val is None:
                return None, conf_f, evidence

            if typ == "date":
                m = DATE_RE.search(str(val))
                return (m.group(1) if m else None), max(0.0, min(1.0, conf_f)), evidence

            num = _to_float(val)
            if num is None:
                return None, max(0.0, min(1.0, conf_f)), evidence

            if typ == "binary":
                return (1.0 if num >= 1 else 0.0), max(0.0, min(1.0, conf_f)), evidence

            if typ in ("count_0_to_N", "count"):
                num = float(int(round(num)))
                num = max(0.0, min(float(N), num))
                return num, max(0.0, min(1.0, conf_f)), evidence

            return float(num), max(0.0, min(1.0, conf_f)), evidence
        except Exception:
            pass

    # 3) Standalone yes/no
    if typ == "binary":
        if YES_RE.match(gt):
            return 1.0, 0.55, evidence
        if NO_RE.match(gt):
            return 0.0, 0.55, evidence

    # 4) single number only
    nm = NUM_ONLY_RE.match(gt)
    if nm:
        num = _to_float(nm.group(1))
        if num is None:
            return None, 0.0, evidence

        if typ == "binary":
            return (1.0 if num >= 1 else 0.0), 0.75, evidence

        if typ in ("count_0_to_N", "count"):
            num = float(int(round(num)))
            num = max(0.0, min(float(N), num))
            return num, 0.75, evidence

        return float(num), 0.75, evidence

    # 5) date anywhere
    if typ == "date":
        m = DATE_RE.search(gt)
        if m:
            return m.group(1), 0.6, evidence
        return None, 0.0, evidence

    # 6) first number anywhere
    num = _extract_first_number(gt)
    if num is not None:
        if typ == "binary":
            return (1.0 if num >= 1 else 0.0), 0.5, evidence
        if typ in ("count_0_to_N", "count"):
            num = float(int(round(num)))
            num = max(0.0, min(float(N), num))
            return num, 0.5, evidence
        return float(num), 0.5, evidence

    return None, 0.0, evidence


def infer_symbol(
    symbol: str,
    context: str,
    N: int,
    prompt_defs: Dict[str, Any],
    hf_runner,
) -> Tuple[Optional[Union[float, str]], str, float, str]:
    """
    Returns: (value, raw_text, confidence, evidence)
    value is None when inference did not work.
    value can be 0.0 if the model answered 0 (a real zero).
    """
    cfg = prompt_defs.get(symbol)
    if cfg is None or hf_runner is None:
        return None, "", 0.0, ""

    typ = cfg.get("type", "binary")
    prompt = format_prompt(cfg["prompt"], context, N)

    generated_text = hf_runner(prompt, max_new_tokens=64)
    val, conf, evidence = _parse_value_and_confidence(generated_text, typ, N)

    return val, (generated_text or ""), float(conf), evidence
