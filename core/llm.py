from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple
import re
import json

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
NUM_ONLY_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*$", re.DOTALL)
ANSWER_RE = re.compile(r"\banswer\s*[:=]\s*([01])\b", re.IGNORECASE)

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
    Transformers v5 safe: use AutoModel + generate (no pipeline task strings).
    Returns a callable: runner(prompt, max_new_tokens) -> generated_text
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
            # keep input short to avoid long sequence issues
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

            # causal models often echo prompt
            if kind == "causal" and text.startswith(prompt):
                text = text[len(prompt):].strip()
            return text

    return runner

def _parse_value_and_confidence(generated_text: str, typ: str) -> Tuple[Optional[float | str], float]:
    confidence = 0.0

    if typ == "date":
        m = DATE_RE.search(generated_text)
        if not m or generated_text.strip().upper() == "UNKNOWN":
            return None, 0.0
        return m.group(1), 0.8

    jm = JSON_RE.search(generated_text)
    if jm:
        try:
            obj = json.loads(jm.group(0))
            val = obj.get("value", None)
            conf = obj.get("confidence", 0.0)
            confidence = float(conf) if conf is not None else 0.0
            if val is None:
                return None, confidence
            return float(val), confidence
        except Exception:
            pass

    am = ANSWER_RE.search(generated_text)
    if am:
        return float(am.group(1)), 0.6

    nm = NUM_ONLY_RE.match(generated_text)
    if nm:
        return float(nm.group(1)), 0.8

    return None, 0.0

def infer_symbol(
    symbol: str,
    context: str,
    N: int,
    prompt_defs: Dict[str, Any],
    hf_runner,
) -> Tuple[Optional[float | str], str, float]:
    cfg = prompt_defs.get(symbol)
    if cfg is None or hf_runner is None:
        return None, "", 0.0

    typ = cfg.get("type", "binary")
    prompt = format_prompt(cfg["prompt"], context, N)

    generated_text = hf_runner(prompt, max_new_tokens=64)

    parsed_val, conf = _parse_value_and_confidence(generated_text, typ)

    if parsed_val is None:
        if typ in ("binary", "count_0_to_N", "count"):
            return 0.0, generated_text, 0.0
        return None, generated_text, 0.0

    if typ == "binary":
        v = 1.0 if float(parsed_val) >= 1 else 0.0
        return v, generated_text, conf

    if typ in ("count_0_to_N", "count"):
        v = max(0.0, min(float(N), float(parsed_val)))
        return v, generated_text, conf

    return float(parsed_val), generated_text, conf
