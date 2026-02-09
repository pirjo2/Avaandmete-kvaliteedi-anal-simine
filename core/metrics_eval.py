from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import datetime as _dt
import math
import re

import pandas as pd

from core.llm import infer_symbol, date_to_ordinal

COND_ALLOWED_RE = re.compile(r"^[0-9a-zA-Z_ .<>=!()+\-*/]+$")
URL_RE = re.compile(r"^https?://", re.IGNORECASE)
EHAK_RE = re.compile(r"^\d{4}$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _safe_eval_condition(expr: str, env: Dict[str, Any]) -> bool:
    if not isinstance(expr, str) or not COND_ALLOWED_RE.match(expr):
        return False
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False


def _env_to_float(v: Any) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        d = date_to_ordinal(v)
        if d is not None:
            return float(d)
    try:
        return float(v)
    except Exception:
        return float("nan")


def _eval_expr(node: Any, env: Dict[str, Any]) -> float:
    if node is None:
        return float("nan")

    if isinstance(node, str):
        return _env_to_float(env.get(node, None))

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
            for rule in node.get("conditions", []) or []:
                if not isinstance(rule, dict):
                    continue
                if "if" in rule and _safe_eval_condition(rule["if"], env):
                    return _eval_expr(rule.get("then"), env)
                if "elif" in rule and _safe_eval_condition(rule["elif"], env):
                    return _eval_expr(rule.get("then"), env)
                if "else" in rule:
                    return _eval_expr(rule.get("else"), env)
            return float("nan")

    return float("nan")


def _profile_df(df: pd.DataFrame, sample_n: int = 3, max_cols: int = 40) -> Dict[str, Any]:
    profile: Dict[str, Any] = {}
    cols = list(df.columns)[:max_cols]
    for col in cols:
        s = df[col]
        missing = float(s.isna().mean()) if len(s) else 0.0
        dtype = str(s.dtype)
        samples = [x for x in s.dropna().head(sample_n).astype(str).tolist()]
        profile[str(col)] = {"dtype": dtype, "missing": round(missing, 6), "samples": samples}
    return profile


def _build_llm_context(df: pd.DataFrame, dataset_description: str = "", file_ext: str = "", file_name: str = "") -> str:
    cols = list(df.columns)
    profile = _profile_df(df, sample_n=3, max_cols=40)

    parts = []
    if file_name:
        parts.append(f"File name: {file_name}")
    if file_ext:
        parts.append(f"File extension: {file_ext}")
    if dataset_description:
        parts.append("Dataset description:\n" + dataset_description.strip())
    parts.append(f"The dataset has {len(cols)} columns (N={len(cols)}) and {len(df)} rows (R={len(df)}).")
    parts.append("Column names: " + ", ".join([str(c) for c in cols[:40]]))
    parts.append("Column profile (dtype, missing ratio, samples):")
    for col, info in profile.items():
        parts.append(f"- {col}: dtype={info['dtype']}, missing={info['missing']}, samples={info['samples']}")
    return "\n".join(parts)


def _best_date_column(df: pd.DataFrame, max_check: int = 2000) -> Optional[str]:
    candidates = [c for c in df.columns if "date" in str(c).lower() or "time" in str(c).lower()]
    if not candidates:
        return None

    best = None
    best_rate = 0.0
    for c in candidates[:10]:
        s = df[c].dropna().astype(str).head(max_check)
        if len(s) == 0:
            continue
        parsed = pd.to_datetime(s, errors="coerce", utc=True)
        rate = float(parsed.notna().mean())
        if rate > best_rate:
            best_rate = rate
            best = c

    return best if best_rate >= 0.2 else None


def _count_incomplete_cells(df: pd.DataFrame) -> int:
    s = df.replace(r"^\s*$", pd.NA, regex=True)
    return int(s.isna().sum().sum())


def _count_incomplete_rows(df: pd.DataFrame) -> int:
    s = df.replace(r"^\s*$", pd.NA, regex=True)
    return int(s.isna().any(axis=1).sum())


def _infer_non_proprietary(file_ext: str) -> float:
    ext = (file_ext or "").lower().lstrip(".")
    if ext in ("csv", "json", "xml", "geojson"):
        return 1.0
    if ext in ("xlsx", "xls"):
        return 0.0
    return 0.0


def _infer_uses_uris(df: pd.DataFrame) -> float:
    cols_lower = [str(c).lower() for c in df.columns]
    if any(k in cols_lower for k in ["uri", "url", "link", "permalink"]):
        return 1.0
    for c in df.columns[:25]:
        s = df[c].dropna().astype(str).head(200)
        if any(URL_RE.match(x) for x in s.tolist()):
            return 1.0
    return 0.0


def _infer_links_other_datasets(df: pd.DataFrame) -> float:
    for c in df.columns[:25]:
        s = df[c].dropna().astype(str).head(200)
        for x in s.tolist():
            if URL_RE.match(x) and any(tok in x.lower() for tok in ["dataset", "api", "sparql", "ckan"]):
                return 1.0
    return 0.0


def _heuristic_doc_like_column_name(name: str) -> bool:
    n = str(name)
    if len(n) < 3:
        return False
    if re.fullmatch(r"[A-Z]{2,6}", n):
        return False
    if re.fullmatch(r"\d+", n):
        return False
    return True


def _infer_ncm(df: pd.DataFrame) -> int:
    return int(sum(_heuristic_doc_like_column_name(c) for c in df.columns))


def _infer_ncuf(df: pd.DataFrame) -> int:
    count = 0
    for c in df.columns:
        name_ok = _heuristic_doc_like_column_name(c)
        s = df[c].dropna().astype(str).head(200)
        val_ok = False
        for v in s.tolist():
            if ISO_DATE_RE.fullmatch(v.strip()):
                val_ok = True
                break
            if re.search(r"[A-Za-zÀ-ž]", v):
                val_ok = True
                break
        if name_ok and val_ok:
            count += 1
    return int(count)


def _infer_ns_nsc(df: pd.DataFrame) -> Tuple[int, int]:
    ns = 0
    nsc = 0
    for c in df.columns:
        name = str(c).lower()
        s = df[c].dropna().astype(str).head(300)

        if "date" in name or "time" in name:
            ns += 1
            parsed = pd.to_datetime(s, errors="coerce", utc=True)
            if len(s) and float(parsed.notna().mean()) >= 0.8:
                nsc += 1
            continue

        if "ehak" in name or name.endswith("ehak"):
            ns += 1
            if len(s) and float(s.apply(lambda x: bool(EHAK_RE.fullmatch(x.strip()))).mean()) >= 0.8:
                nsc += 1
            continue

        if name.endswith("id") or "code" in name:
            ns += 1
            if len(s) and float(s.apply(lambda x: bool(re.fullmatch(r"[0-9A-Za-z\-_.]+", x.strip()))).mean()) >= 0.8:
                nsc += 1
            continue

    ns = min(ns, len(df.columns))
    nsc = min(nsc, ns)
    return int(ns), int(nsc)


def compute_metrics(
    df: pd.DataFrame,
    formulas_cfg: Dict[str, Any],
    prompt_cfg: Dict[str, Any],
    use_llm: bool,
    hf_runner,
    dataset_description: str = "",
    file_ext: str = "",
    file_name: str = "",
    confidence_weighting: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    vetro = (formulas_cfg or {}).get("vetro_methodology") or {}
    if not isinstance(vetro, dict):
        vetro = {}

    prompts = (prompt_cfg or {}).get("symbols", {}) or {}
    if not isinstance(prompts, dict):
        prompts = {}

    env: Dict[str, Any] = {}
    env["N"] = int(df.shape[1])
    env["R"] = int(df.shape[0])

    auto_inputs: Dict[str, Any] = {}
    auto_inputs["nc"] = int(df.shape[1])
    auto_inputs["nr"] = int(df.shape[0])
    auto_inputs["ncl"] = int(df.shape[0] * df.shape[1])
    auto_inputs["ic"] = _count_incomplete_cells(df)
    auto_inputs["nir"] = _count_incomplete_rows(df)

    auto_inputs["cd"] = _dt.date.today().isoformat()

    date_col = _best_date_column(df)
    if date_col is not None:
        dt = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        if dt.notna().any():
            auto_inputs["sd_col"] = str(date_col)
            auto_inputs["sd"] = dt.min().date().isoformat()
            auto_inputs["edp"] = dt.max().date().isoformat()
            auto_inputs["max_date"] = dt.max().date().isoformat()
            auto_inputs["ncr"] = int((dt != dt.max()).sum())

    for c in df.columns:
        if str(c).lower() in ("modifiedat", "modified_at", "updatedat", "updated_at", "lastmodified", "last_modified"):
            dtt = pd.to_datetime(df[c], errors="coerce", utc=True)
            if dtt.notna().any():
                auto_inputs["dp"] = dtt.max().date().isoformat()
                break

    # more auto/heuristics
    auto_inputs["s2"] = 1.0
    auto_inputs["s3"] = _infer_non_proprietary(file_ext)
    auto_inputs["s4"] = _infer_uses_uris(df)
    auto_inputs["s5"] = _infer_links_other_datasets(df)
    auto_inputs["ncm"] = float(_infer_ncm(df))
    auto_inputs["ncuf"] = float(_infer_ncuf(df))
    ns, nsc = _infer_ns_nsc(df)
    auto_inputs["ns"] = float(ns)
    auto_inputs["nsc"] = float(nsc)

    # Put auto inputs into env; dates -> ordinal for formulas
    for k, v in auto_inputs.items():
        if k in ("sd", "edp", "max_date", "dp", "ed", "cd"):
            env[k] = date_to_ordinal(v)
        else:
            env[k] = v

    context = _build_llm_context(df, dataset_description, file_ext, file_name)

    required_symbols = set()
    for dim, dim_obj in vetro.items():
        if not isinstance(dim_obj, dict):
            continue
        for _, metric_obj in dim_obj.items():
            if not isinstance(metric_obj, dict):
                continue
            for inp in metric_obj.get("inputs", []) or []:
                if isinstance(inp, dict):
                    for _, sym in inp.items():
                        required_symbols.add(sym)

    symbol_rows = []

    for sym in sorted(required_symbols):
        # if already auto-filled
        if sym in env and env[sym] is not None:
            symbol_rows.append({"symbol": sym, "value": auto_inputs.get(sym, None), "source": "auto", "confidence": 0.95, "evidence": "auto", "raw": ""})
            continue

        if use_llm and hf_runner is not None and sym in prompts:
            val, raw, conf, evid = infer_symbol(
                symbol=sym,
                context=context,
                N=int(df.shape[1]),
                prompt_defs=prompts,
                hf_runner=hf_runner,
                extra_values={
                    "dataset_description": dataset_description,
                    "columns": ", ".join([str(c) for c in df.columns]),
                    "file_ext": file_ext,
                    "N": int(df.shape[1]),
                },
            )

            if val is None:
                env[sym] = None
                symbol_rows.append({"symbol": sym, "value": None, "source": "fail", "confidence": 0.0, "evidence": evid, "raw": raw})
                continue

            # date symbol -> ordinal
            if isinstance(val, str) and (sym in ("sd", "edp", "max_date", "dp", "ed", "cd")):
                ordv = date_to_ordinal(val)
                env[sym] = ordv
                symbol_rows.append({"symbol": sym, "value": val, "source": "llm", "confidence": float(conf), "evidence": evid, "raw": raw})
                continue

            num_val = float(val)
            env[sym] = num_val * float(conf) if confidence_weighting else num_val
            symbol_rows.append({"symbol": sym, "value": num_val, "source": "llm", "confidence": float(conf), "evidence": evid, "raw": raw})
            continue

        env[sym] = None
        symbol_rows.append({"symbol": sym, "value": None, "source": "fail", "confidence": 0.0, "evidence": "", "raw": ""})

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

            rows.append(
                {
                    "dimension": dim,
                    "metric": metric_key,
                    "metric_id": f"{dim}.{metric_key}",
                    "value": value,
                    "description": metric_obj.get("description", ""),
                }
            )

    metrics_df = pd.DataFrame(rows)

    details = {
        "auto_inputs": auto_inputs,
        "symbol_table": pd.DataFrame(symbol_rows),
    }
    return metrics_df, details
