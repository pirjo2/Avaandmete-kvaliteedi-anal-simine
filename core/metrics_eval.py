from __future__ import annotations

from typing import Any, Dict, Tuple
import math
import re
import pandas as pd

from core.llm import infer_symbol

COND_ALLOWED_RE = re.compile(r"^[0-9a-zA-Z_ .<>=!()+\-*/]+$")

def _safe_eval_condition(expr: str, env: Dict[str, float]) -> bool:
    if not COND_ALLOWED_RE.match(expr):
        return False
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False

def _eval_expr(node: Any, env: Dict[str, Any]) -> float:
    if node is None:
        return float("nan")

    if isinstance(node, str):
        v = env.get(node, None)
        if v is None:
            return float("nan")
        try:
            return float(v)
        except Exception:
            return float("nan")

    if isinstance(node, (int, float)):
        return float(node)

    if isinstance(node, dict) and "operator" in node:
        op = node["operator"]

        if op == "identity":
            return _eval_expr(node.get("of"), env)

        if op == "add":
            return _eval_expr(node.get("left"), env) + _eval_expr(node.get("right"), env)

        if op == "subtract":
            return _eval_expr(node.get("left"), env) - _eval_expr(node.get("right"), env)

        if op == "multiply":
            return _eval_expr(node.get("left"), env) * _eval_expr(node.get("right"), env)

        if op == "divide":
            denom = _eval_expr(node.get("right"), env)
            if denom == 0 or math.isnan(denom):
                return float("nan")
            return _eval_expr(node.get("left"), env) / denom

        if op == "abs_diff":
            return abs(_eval_expr(node.get("left"), env) - _eval_expr(node.get("right"), env))

        if op == "conditional":
            for rule in node.get("conditions", []):
                if "if" in rule and _safe_eval_condition(rule["if"], env):
                    return _eval_expr(rule.get("then"), env)
                if "elif" in rule and _safe_eval_condition(rule["elif"], env):
                    return _eval_expr(rule.get("then"), env)
                if "else" in rule:
                    return _eval_expr(rule.get("else"), env)
            return float("nan")

    return float("nan")

def _profile_df(df: pd.DataFrame, sample_n: int = 2) -> Dict[str, Any]:
    profile = {}
    for col in df.columns[:40]:
        s = df[col]
        missing = float(s.isna().mean())
        dtype = str(s.dtype)
        samples = [x for x in s.dropna().head(sample_n).astype(str).tolist()]
        profile[col] = {"dtype": dtype, "missing": round(missing, 6), "samples": samples}
    return profile

def _build_llm_context(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    profile = _profile_df(df, sample_n=2)

    parts = []
    parts.append(f"Columns (N={len(cols)}): {', '.join(cols[:40])}")
    parts.append("Column profile (dtype, missing ratio, samples):")
    for col, info in profile.items():
        parts.append(f"- {col}: dtype={info['dtype']}, missing={info['missing']}, samples={info['samples']}")
    return "\n".join(parts)

def compute_metrics(
    df: pd.DataFrame,
    formulas_cfg: Dict[str, Any],
    prompt_cfg: Dict[str, Any],
    use_llm: bool,
    hf_runner,  # callable runner from llm.py
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    # robustly get methodology dict
    vetro = (formulas_cfg or {}).get("vetro_methodology") or {}
    if not isinstance(vetro, dict):
        vetro = {}

    prompts = (prompt_cfg or {}).get("symbols", {}) or {}

    env: Dict[str, Any] = {}
    env["N"] = int(df.shape[1])
    env["R"] = int(df.shape[0])

    # Auto inputs
    auto_inputs: Dict[str, Any] = {}

    # Detect a date-like column for sd/edp if possible
    date_col = None
    for c in df.columns:
        if "date" in str(c).lower():
            date_col = c
            break

    if date_col is not None:
        dt = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        if dt.notna().any():
            auto_inputs["sd_col"] = str(date_col)
            auto_inputs["sd"] = dt.min().date().isoformat()
            auto_inputs["edp"] = dt.max().date().isoformat()
            auto_inputs["max_date"] = dt.max().date().isoformat()

    for k, v in auto_inputs.items():
        env[k] = v

    # Build LLM context (short)
    context = _build_llm_context(df)

    llm_raw: Dict[str, str] = {}
    llm_conf: Dict[str, float] = {}

    # collect required symbols safely (skip None dimensions/metrics)
    required_symbols = set()
    for dim, dim_obj in vetro.items():
        if not isinstance(dim_obj, dict):
            continue
        for metric_key, metric_obj in dim_obj.items():
            if not isinstance(metric_obj, dict):
                continue
            for inp in metric_obj.get("inputs", []) or []:
                if isinstance(inp, dict):
                    for _, sym in inp.items():
                        required_symbols.add(sym)

    # Fill missing symbols
    for sym in sorted(required_symbols):
        if sym in env and env[sym] not in (None, ""):
            continue

        if use_llm and sym in prompts and hf_runner is not None:
            val, raw, conf = infer_symbol(sym, context=context, N=int(df.shape[1]), prompt_defs=prompts, hf_runner=hf_runner)
            llm_raw[sym] = raw
            llm_conf[sym] = float(conf)

            # Confidence gating (optional, keeps stable)
            CONF_THRESHOLD = 0.4
            env[sym] = 0.0 if conf < CONF_THRESHOLD else val
        else:
            env[sym] = 0.0

    # Compute metrics
    rows = []
    for dim, dim_obj in vetro.items():
        if not isinstance(dim_obj, dict):
            continue
        for metric_key, metric_obj in dim_obj.items():
            if not isinstance(metric_obj, dict):
                continue

            f_assign = (metric_obj.get("formula") or {}).get("assign")
            f_expr = (metric_obj.get("formula") or {}).get("expression")
            if f_assign and f_expr:
                env[f_assign] = _eval_expr(f_expr, env)

            n_assign = (metric_obj.get("normalization") or {}).get("assign")
            n_expr = (metric_obj.get("normalization") or {}).get("expression")
            value = float("nan")
            if n_assign and n_expr:
                value = _eval_expr(n_expr, env)
                env[n_assign] = value

            metric_id = f"{dim}.{metric_key}"
            rows.append({
                "dimension": dim,
                "metric": metric_key,
                "metric_id": metric_id,
                "value": value,
                "description": metric_obj.get("description", ""),
            })

    metrics_df = pd.DataFrame(rows)

    details = {
        "auto_inputs": auto_inputs,
        "llm_raw": llm_raw,
        "llm_confidence": llm_conf,
    }
    return metrics_df, details
