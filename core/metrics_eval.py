from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import math
import re
import pandas as pd

from core.llm import infer_symbol

COND_ALLOWED_RE = re.compile(r"^[0-9a-zA-Z_ .<>=!()+\-*/]+$")

def _safe_eval_condition(expr: str, env: Dict[str, float]) -> bool:
    """
    YAML has conditions like 'dae <= 0'. We eval them with restricted env.
    """
    if not COND_ALLOWED_RE.match(expr):
        return False
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False

def _eval_expr(node: Any, env: Dict[str, Any]) -> float:
    """
    Evaluate operator trees from test2.yaml.
    Supports operators present in your YAML: add, subtract, multiply, divide,
    abs_diff, sum, identity, conditional.
    """
    if node is None:
        return float("nan")

    # symbol reference
    if isinstance(node, str):
        v = env.get(node, None)
        if v is None:
            return float("nan")
        try:
            return float(v)
        except Exception:
            return float("nan")

    # literal number
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

        if op == "sum":
            # Optional: if env contains lists for dav_i and oav_i, sum abs diffs
            inner = node.get("of")
            # If sum requires dav_i/oav_i as arrays, try:
            if isinstance(inner, dict) and inner.get("operator") == "abs_diff":
                left = inner.get("left")
                right = inner.get("right")
                if isinstance(left, str) and isinstance(right, str):
                    lv = env.get(left)
                    rv = env.get(right)
                    if isinstance(lv, list) and isinstance(rv, list) and len(lv) == len(rv):
                        s = 0.0
                        for a, b in zip(lv, rv):
                            try:
                                s += abs(float(a) - float(b))
                            except Exception:
                                continue
                        return s
            return float("nan")

        if op == "conditional":
            # format in YAML:
            # conditions:
            # - if: "dae <= 0"
            #   then: 0
            # - elif: "dae <= 1"
            #   then: dae
            # - else: 1
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
    """
    Build short profiling info for LLM context (avoid token explosion).
    """
    profile = {}
    for col in df.columns[:40]:
        s = df[col]
        missing = float(s.isna().mean())
        dtype = str(s.dtype)
        samples = [x for x in s.dropna().head(sample_n).astype(str).tolist()]
        profile[col] = {"dtype": dtype, "missing": round(missing, 6), "samples": samples}
    return profile

def _build_llm_context(df: pd.DataFrame, extra_metadata: Dict[str, str]) -> str:
    """
    Small, readable context. This is where AI can behave 'intuitively'.
    """
    cols = list(df.columns)
    profile = _profile_df(df, sample_n=2)

    parts = []
    if extra_metadata.get("title"):
        parts.append(f"Title: {extra_metadata['title']}")
    if extra_metadata.get("publisher"):
        parts.append(f"Publisher: {extra_metadata['publisher']}")
    if extra_metadata.get("licence"):
        parts.append(f"Licence: {extra_metadata['licence']}")
    if extra_metadata.get("description"):
        parts.append(f"Description: {extra_metadata['description']}")

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
    hf_pipe,
    extra_metadata: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    vetro = formulas_cfg.get("vetro_methodology", {})
    prompts = (prompt_cfg or {}).get("symbols", {})

    env: Dict[str, Any] = {}

    # --- Auto inputs (from data) ---
    env["N"] = int(df.shape[1])
    env["R"] = int(df.shape[0])

    # completeness (used by several metrics)
    total_cells = float(df.shape[0] * df.shape[1]) if df.shape[0] and df.shape[1] else 0.0
    non_empty = float(df.notna().sum().sum())
    complete_cells = non_empty / total_cells if total_cells > 0 else float("nan")
    env["pcc"] = complete_cells  # you can map if your YAML expects a symbol
    # some YAML uses other symbols; we still compute core metrics later from formula trees

    # Try detect date column for currentness
    date_col = None
    for c in df.columns:
        if "date" in str(c).lower():
            date_col = c
            break

    auto_inputs = {}
    auto_inputs["cd"] = pd.Timestamp.utcnow().date().isoformat()

    if date_col is not None:
        auto_inputs["sd_col"] = str(date_col)
        # parse dates
        dt = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        if dt.notna().any():
            min_d = dt.min().date().isoformat()
            max_d = dt.max().date().isoformat()
            auto_inputs["sd"] = min_d
            auto_inputs["edp"] = max_d
            auto_inputs["max_date"] = max_d

    # push auto inputs into env if symbol names match
    for k, v in auto_inputs.items():
        env[k] = v

    # --- LLM context ---
    context = _build_llm_context(df, extra_metadata)
    llm_raw: Dict[str, str] = {}
    llm_conf: Dict[str, float] = {}

    # collect required symbols from YAML inputs
    required_symbols = set()
    for dim, dim_obj in vetro.items():
        for metric_key, metric_obj in dim_obj.items():
            for inp in metric_obj.get("inputs", []):
                for _, sym in inp.items():
                    required_symbols.add(sym)

    # infer missing symbols via LLM (or set 0)
    for sym in sorted(required_symbols):
        if sym in env and env[sym] is not None and env[sym] != "":
            continue

        if use_llm and sym in prompts and hf_pipe is not None:
            val, raw, conf = infer_symbol(sym, context=context, N=int(df.shape[1]), prompt_defs=prompts, hf_pipe=hf_pipe)
            llm_raw[sym] = raw
            llm_conf[sym] = float(conf)
            env[sym] = val
        else:
            # fallback for symbols not available
            env[sym] = 0.0

    # --- Compute metrics ---
    rows = []
    for dim, dim_obj in vetro.items():
        for metric_key, metric_obj in dim_obj.items():
            # formula assigns intermediate var
            f_assign = metric_obj.get("formula", {}).get("assign")
            f_expr = metric_obj.get("formula", {}).get("expression")
            if f_assign and f_expr:
                env[f_assign] = _eval_expr(f_expr, env)

            n_assign = metric_obj.get("normalization", {}).get("assign")
            n_expr = metric_obj.get("normalization", {}).get("expression")
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
        "env": {k: env[k] for k in sorted(env.keys()) if k in required_symbols or k in ("N", "R")},
    }
    return metrics_df, details