from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import math
import re
from datetime import date as _date_type

import pandas as pd


# --- Turvaline tingimusavaldiste eval --- #
COND_ALLOWED_RE = re.compile(r"^[0-9a-zA-Z_ .<>=!()+\\-*/]+$")


def _safe_eval_condition(expr: str, env: Dict[str, Any]) -> bool:
    """
    Safely evaluate a simple boolean expression used in normalization conditionals.
    Only a restricted character set is allowed and builtins are disabled.
    """
    if not isinstance(expr, str) or not COND_ALLOWED_RE.match(expr):
        return False
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False


# --- Väljendipuu hindamine (Variant B) --- #


def _eval_expr(node: Any, env: Dict[str, Any]) -> float:
    """
    Evaluate a metric expression tree against the environment `env`.

    Supports the operators used in the Vetrò YAML:
    - identity (with `operand` or `of`)
    - add (binary with left/right OR n-ary with `terms`)
    - subtract, multiply, divide
    - abs_diff
    - sum (with `of`: single expr or list)
    - conditional (with if/elif/else and simple boolean expressions)
    """
    if node is None:
        return float("nan")

    # Numeric literal
    if isinstance(node, (int, float)):
        return float(node)

    # Symbol / string literal
    if isinstance(node, str):
        if node in env:
            v = env[node]
            if v is None:
                return 0.0
            try:
                return float(v)
            except Exception:
                # fall through to try literal parsing
                pass
        # Try interpret as numeric literal
        try:
            return float(node)
        except Exception:
            return 0.0

    # Operator node
    if isinstance(node, dict):
        op = node.get("operator")

        if op == "identity":
            # Support both "operand" (new) and "of" (older version)
            return _eval_expr(node.get("operand", node.get("of")), env)

        if op == "add":
            # N-ary add with "terms"
            if "terms" in node:
                total = 0.0
                for term in (node.get("terms") or []):
                    v = _eval_expr(term, env)
                    if isinstance(v, float) and math.isnan(v):
                        v = 0.0
                    total += v
                return total
            # Binary add
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
            items = node.get("of")
            if not isinstance(items, list):
                items = [items]
            total = 0.0
            for term in items:
                v = _eval_expr(term, env)
                if isinstance(v, float) and math.isnan(v):
                    v = 0.0
                total += v
            return total

        if op == "conditional":
            for rule in node.get("conditions", []) or []:
                if not isinstance(rule, dict):
                    continue
                # if / elif / else style, where each branch has a "then" or "else" expression
                if "if" in rule and _safe_eval_condition(rule["if"], env):
                    return _eval_expr(rule.get("then"), env)
                if "elif" in rule and _safe_eval_condition(rule["elif"], env):
                    return _eval_expr(rule.get("then"), env)
                if "else" in rule:
                    return _eval_expr(rule.get("else"), env)
            return float("nan")

    return float("nan")


def _date_to_num(value: Any) -> Optional[float]:
    """
    Convert a date-like value (ISO string, pandas Timestamp or datetime/date)
    into a numeric day count using .toordinal(). Returns None if parsing fails.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # assume already numeric
        return float(value)
    try:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(ts):
            return None
        return float(ts.date().toordinal())
    except Exception:
        return None


# --- Automaatsete sisendite arvutamine --- #


def _auto_inputs(df: pd.DataFrame, file_ext: Optional[str] = None) -> Dict[str, Any]:
    """
    Derive basic metrics inputs directly from the tabular data, without LLMs.

    The goal is to produce sensible defaults that work across many datasets and
    approximate the intended Vetrò metrics when possible.
    """
    auto: Dict[str, Any] = {}

    N = int(df.shape[1])
    R = int(df.shape[0])
    auto["nc"] = float(N)
    auto["nr"] = float(R)

    # Completeness: total cells, incomplete cells, incomplete rows
    auto["ncl"] = float(N * R)
    na_mask = df.isna()
    auto["ic"] = float(na_mask.sum().sum())
    auto["nir"] = float(na_mask.any(axis=1).sum())

    # Accuracy: syntactically invalid cells (simple heuristic: assume 0 beyond missing values)
    auto["nce"] = 0.0

    # Currentness: detect a primary date column
    date_col = None
    for c in df.columns:
        if "date" in str(c).lower():
            date_col = c
            break

    if date_col is not None:
        dt_series = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        if dt_series.notna().any():
            sd = dt_series.min().date()
            edp = dt_series.max().date()
            auto["sd_col"] = str(date_col)
            auto["sd"] = sd.isoformat()
            auto["edp"] = edp.isoformat()
            auto["max_date"] = edp.isoformat()
            auto["ed"] = edp.isoformat()  # expiration ≈ previous end of period
            auto["ncr"] = float((dt_series != dt_series.max()).sum())

    # "Today" as the current date for delay-after-expiration
    auto["cd"] = _date_type.today().isoformat()

    # Publication / update dates from typical columns like ModifiedAt or UpdatedAt
    mod_col = None
    for c in df.columns:
        cl = str(c).lower()
        if cl in ("modifiedat", "modified_at", "updatedat", "updated_at", "lastmodified", "last_modified"):
            mod_col = c
            break

    if mod_col is not None:
        dtm = pd.to_datetime(df[mod_col], errors="coerce", utc=True)
        if dtm.notna().any():
            dp = dtm.max().date()
            auto["dp"] = dp.isoformat()
            auto["du"] = 1.0  # at least one update recorded

    # Fallback: if we have a max_date from the time series, reuse it as publication date
    if "dp" not in auto and "max_date" in auto:
        auto["dp"] = auto["max_date"]

    # Standardised columns (ns: with standards applicable, nsc: actually standardised)
    def _infer_ns_nsc(df2: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        ns = 0.0
        nsc = 0.0

        for col in df2.columns:
            name = str(col)
            lname = name.lower()
            s = df2[col].dropna()
            if s.empty:
                continue

            is_candidate = False
            is_standardised = False

            # Dates
            if "date" in lname or "kuup" in lname:
                is_candidate = True
                dt2 = pd.to_datetime(s, errors="coerce", utc=True)
                if dt2.notna().mean() > 0.9:
                    is_standardised = True

            # Years
            elif "year" in lname:
                is_candidate = True
                vals = pd.to_numeric(s, errors="coerce")
                if vals.notna().mean() > 0.9 and vals.between(1900, 2100).mean() > 0.9:
                    is_standardised = True

            # Official codes (EHAK, ISO, etc.)
            elif any(tok in lname for tok in ("ehak", "iso", "code")):
                is_candidate = True
                is_standardised = True

            # Geography / coverage columns without explicit code marker
            elif any(tok in lname for tok in ("country", "county", "commune", "region", "maakond", "vald", "linn")):
                is_candidate = True
                # treat as applicable, but not necessarily fully standardised

            if is_candidate:
                ns += 1.0
                if is_standardised:
                    nsc += 1.0

        if ns == 0:
            return None, None
        return ns, nsc

    ns, nsc = _infer_ns_nsc(df)
    if ns is not None:
        auto["ns"] = ns
    if nsc is not None:
        auto["nsc"] = nsc

    # Understandability: simple heuristic – treat all columns as having some metadata
    auto["ncm"] = float(N)   # columns with metadata
    auto["ncuf"] = float(N)  # columns in understandable & machine-readable format

    # 5-star open data heuristics
    ext = (file_ext or "").lower()
    auto["s1"] = 1.0  # available on the web (the user uploaded it)
    auto["s2"] = 1.0  # structured data (tabular)
    auto["s3"] = 1.0 if ext in (".csv", ".tsv", ".json", ".xml") else 0.5  # open formats
    cols_lower = [str(c).lower() for c in df.columns]
    auto["s4"] = 1.0 if any(("id" in c or "uuid" in c or "uri" in c) for c in cols_lower) else 0.0
    auto["s5"] = 0.0  # we rarely have explicit linked data information in the raw file alone

    # Traceability proxies: source presence & creation date
    auto["s"] = 1.0 if ("dp" in auto or "sd" in auto) else 0.0
    auto["dc"] = 1.0 if "dp" in auto else 0.0

    # List-of-updates flag (lu): no explicit changelog in a single CSV
    auto["lu"] = 0.0
    if "du" not in auto and "dp" in auto and "sd" in auto and auto["dp"] != auto["sd"]:
        auto["du"] = 1.0

    # Aggregation accuracy: by default we assume perfect aggregation if nothing else is known
    auto["sc"] = 1.0  # scale
    auto["oav"] = 0.0
    auto["dav"] = 0.0
    auto["e"] = 0.0   # error in aggregation

    return auto


# --- Põhifunktsioon: compute_metrics --- #


def compute_metrics(
    df: pd.DataFrame,
    formulas_cfg: Dict[str, Any],
    prompt_defs: Optional[Dict[str, Any]],
    use_llm: bool,
    hf_runner,
    file_ext: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute all Vetrò metrics for the given dataframe.

    Parameters
    ----------
    df : DataFrame
        Input tabular dataset.
    formulas_cfg : dict
        Parsed formulas.yaml content (either full file or the vetro_methodology section).
    prompt_defs : dict | None
        Prompt definitions loaded from prompts.yaml["symbols"] (may be None).
    use_llm : bool
        Whether LLM-based symbol inference is enabled.
    hf_runner : callable | None
        Hugging Face runner returned by core.llm.get_hf_runner, or None.
    file_ext : str | None
        File extension of the uploaded dataset (e.g. ".csv") used for some heuristics.

    Returns
    -------
    metrics_df : DataFrame
        One row per metric with dimension, metric, metric_id, value, description and label.
    details : dict
        Debug information – auto-derived inputs and symbol-level inference details.
    """
    # Accept either the full config or just vetro_methodology
    vetro = formulas_cfg.get("vetro_methodology", formulas_cfg)

    # Labels are stored as flattened keys like "completeness.percentage_of_complete_cells"
    label_map: Dict[str, str] = {
        k: v for k, v in vetro.items() if isinstance(v, str) and "." in k
    }

    N = int(df.shape[1])
    R = int(df.shape[0])

    auto_inputs = _auto_inputs(df, file_ext)

    # Environment used to evaluate the expressions – everything must be numeric
    env: Dict[str, Any] = {
        "N": float(N),
        "R": float(R),
    }

    # Install numeric auto inputs into env (we convert dates separately below)
    for k, v in auto_inputs.items():
        if k in ("sd", "edp", "ed", "cd", "dp"):
            continue
        if isinstance(v, (int, float)):
            env[k] = float(v)

    # Debug details
    details: Dict[str, Any] = {
        "auto_inputs": auto_inputs,
        "symbol_values": {},
        "symbol_source": {},
        "llm_confidence": {},
        "llm_raw": {},
        "llm_evidence": {},
    }

    # Collect all symbols that appear in the metric "inputs" definitions
    required_symbols = set()
    for dim, dim_obj in vetro.items():
        if not isinstance(dim_obj, dict):
            continue
        for metric_key, metric_obj in dim_obj.items():
            if not isinstance(metric_obj, dict):
                continue
            for inp in metric_obj.get("inputs", []) or []:
                if isinstance(inp, dict):
                    for sym in inp.values():
                        if isinstance(sym, str):
                            required_symbols.add(sym)

    # First, record all auto-derived symbols for debugging
    for sym in sorted(required_symbols):
        if sym in auto_inputs:
            details["symbol_values"][sym] = auto_inputs[sym]
            details["symbol_source"][sym] = "auto"
        else:
            # We may later fill this from the LLM
            details["symbol_values"].setdefault(sym, None)
            details["symbol_source"].setdefault(sym, "missing")

    # Optional LLM-based inference for symbols that are not auto-derived
    if use_llm and hf_runner is not None and prompt_defs:
        # Build a compact, but informative, context for the model
        context_lines = []
        context_lines.append("Columns:")
        context_lines.append(", ".join(str(c) for c in df.columns))
        context_lines.append("")
        context_lines.append("Basic column profiles:")
        for col in df.columns:
            s = df[col]
            dtype = str(s.dtype)
            missing_ratio = float(s.isna().mean())
            sample_vals = list(s.dropna().unique()[:3])
            context_lines.append(
                f"- {col}: dtype={dtype}, missing={missing_ratio:.3f}, samples={sample_vals}"
            )
        context = "\n".join(context_lines)

        # Local import to avoid a hard dependency at module import time
        from core.llm import infer_symbol as _infer_symbol  # type: ignore

        for sym in sorted(required_symbols):
            if details["symbol_source"].get(sym) != "missing":
                continue
            if sym not in (prompt_defs or {}):
                continue

            val, raw, conf, evid = _infer_symbol(
                symbol=sym,
                context=context,
                N=N,
                prompt_defs=prompt_defs,
                hf_runner=hf_runner,
                extra_values={"N": N},
            )
            details["llm_raw"][sym] = raw
            details["llm_confidence"][sym] = conf
            details["llm_evidence"][sym] = evid

            # If the model is unsure, treat this as a failed inference
            if val is None or (conf is not None and conf < 0.4):
                details["symbol_source"][sym] = "llm_fail"
                details["symbol_values"][sym] = None
                if sym not in env:
                    env[sym] = 0.0
            else:
                details["symbol_source"][sym] = "llm"
                details["symbol_values"][sym] = val
                if isinstance(val, (int, float)):
                    env[sym] = float(val)
                else:
                    # may be a date string; we convert below
                    env[sym] = val

    # Convert date-like symbols (from either auto_inputs or LLM) into numeric values
    for sym in ("sd", "edp", "ed", "cd", "dp"):
        raw_val = details["symbol_values"].get(sym, auto_inputs.get(sym))
        num = _date_to_num(raw_val)
        if num is not None:
            env[sym] = num

    # Ensure all required symbols at least exist in env to avoid KeyErrors
    for sym in required_symbols:
        env.setdefault(sym, 0.0)

    rows = []

    for dim, dim_obj in vetro.items():
        if not isinstance(dim_obj, dict):
            continue

        for metric_key, metric_obj in dim_obj.items():
            if not isinstance(metric_obj, dict):
                continue

            desc = metric_obj.get("description", "")
            inputs = metric_obj.get("inputs", [])
            if not inputs:
                continue

            # Optional intermediate calculations (e.g. ncl = nr * nc, or custom errors)
            interm = metric_obj.get("intermediate_calculation")
            if interm:
                if isinstance(interm, dict) and "assign" in interm:
                    interms = [interm]
                elif isinstance(interm, list):
                    interms = [x for x in interm if isinstance(x, dict)]
                else:
                    interms = []
                for ic in interms:
                    name = ic.get("assign")
                    expr = ic.get("expression")
                    if name and expr:
                        env[name] = _eval_expr(expr, env)

            formula = metric_obj.get("formula", {})
            norm = metric_obj.get("normalization", {})
            if not formula or not norm:
                continue

            f_assign = formula.get("assign")
            f_expr = formula.get("expression")
            n_assign = norm.get("assign")
            n_expr = norm.get("expression")
            if not (f_assign and f_expr and n_assign and n_expr):
                continue

            env[f_assign] = _eval_expr(f_expr, env)
            val = _eval_expr(n_expr, env)

            metric_id = f"{dim}.{metric_key}"
            label = label_map.get(metric_id, metric_id)

            if isinstance(val, (int, float)) and not math.isnan(val):
                out_val: Optional[float] = float(val)
            else:
                out_val = None

            rows.append(
                {
                    "dimension": dim,
                    "metric": metric_key,
                    "metric_id": metric_id,
                    "value": out_val,
                    "description": desc,
                    "metric_label": label,
                }
            )

    metrics_df = pd.DataFrame(rows)
    return metrics_df, details
