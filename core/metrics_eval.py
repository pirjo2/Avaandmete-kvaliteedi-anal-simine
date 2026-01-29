from __future__ import annotations

import math
import re
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .llm import infer_symbol

def read_dataframe_from_path(path: str | Path) -> pd.DataFrame:
    """Read CSV/XLSX/JSON with a few safe fallbacks."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    if suf == ".json":
        return pd.read_json(p)

    # CSV (try a few encodings)
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError:
            continue
    # last resort: let pandas guess
    return pd.read_csv(p, encoding_errors="replace")

def build_context(df: pd.DataFrame, description: str = "", max_samples: int = 5) -> str:
    """Build a compact text context for LLM prompts."""
    col_preview = ", ".join([str(c) for c in df.columns.tolist()])
    profile_lines: List[str] = []
    for col in df.columns:
        s = df[col]
        missing = float(s.isna().mean() + s.eq("").mean())
        uniq = int(s.nunique(dropna=True))
        samples = s.dropna().astype(str).head(max_samples).tolist()
        profile_lines.append(f"- {col}: dtype={s.dtype}, missing={missing:.3f}, unique={uniq}, samples={samples}")
    profile_text = "\n".join(profile_lines)

    description = (description or "").strip()
    if description:
        description += "\n\n"

    return (
        description
        + f"The dataset has {len(df.columns)} columns (N={len(df.columns)}).\n"
        + "Column names:\n" + col_preview + "\n\n"
        + "Column profile summary:\n" + profile_text
    )

def datetime_to_days(dt: pd.Timestamp) -> float:
    if pd.isna(dt):
        return math.nan
    if dt.tzinfo is None:
        dt = dt.tz_localize(timezone.utc)
    epoch = pd.Timestamp("1970-01-01", tz=timezone.utc)
    return (dt - epoch).total_seconds() / 86400.0

def parse_mixed_datetime(series: pd.Series) -> pd.Series:
    """Parse mixed date/datetime strings robustly."""
    s = series.astype(str).str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "NaT": pd.NA})
    out = pd.Series(pd.NaT, index=series.index)

    mask_dot = s.str.match(r"^\d{1,2}\.\d{1,2}\.\d{4}")
    mask_dash = s.str.match(r"^\d{4}-\d{1,2}-\d{1,2}")
    mask_other = ~(mask_dot | mask_dash)

    if mask_dot.any():
        part = s[mask_dot]
        parsed = pd.to_datetime(part, format="%d.%m.%Y %H:%M", errors="coerce")
        parsed2 = pd.to_datetime(part, format="%d.%m.%Y", errors="coerce")
        out.loc[mask_dot] = parsed.fillna(parsed2)

    if mask_dash.any():
        part = s[mask_dash]
        parsed = pd.to_datetime(part, format="%Y-%m-%d %H:%M", errors="coerce")
        parsed2 = pd.to_datetime(part, format="%Y-%m-%d", errors="coerce")
        out.loc[mask_dash] = parsed.fillna(parsed2)

    if mask_other.any():
        part = s[mask_other]
        out.loc[mask_other] = pd.to_datetime(part, errors="coerce")

    return out

def find_best_datetime_column(
    df: pd.DataFrame,
    name_keywords: Tuple[str, ...] = ("date", "time", "modified", "updated", "created")
) -> Tuple[float, Optional[str], Optional[pd.Series]]:
    candidates = []
    for col in df.columns:
        low = str(col).lower()
        if any(k in low for k in name_keywords):
            parsed = parse_mixed_datetime(df[col])
            coverage = float(parsed.notna().mean())
            if coverage > 0:
                candidates.append((coverage, str(col), parsed))
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0] if candidates else (0.0, None, None)

def derive_aggregation_pairs(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Heuristic for Vetrò 'accuracy in aggregation' inputs.

    The original notebook example was domain-specific (LocationLevel/TotalCount).
    Here we keep that pattern if present; otherwise we return None (metric becomes NaN).
    """
    needed = {"LocationLevel", "TotalCount"}
    if needed.issubset(df.columns):
        candidate_keys = ["StatisticsDate", "VaccinationSeason", "AgeGroup", "AgeGroupFilter", "StatisticsYear", "PatientCountryEHAK"]
        keys = [k for k in candidate_keys if k in df.columns]
        if not keys:
            return None

        lvl = df["LocationLevel"].astype(str).str.lower()
        country = df[lvl == "country"]
        county  = df[lvl == "county"]
        if country.empty or county.empty:
            return None

        country_g = country.groupby(keys)["TotalCount"].sum()
        county_g  = county.groupby(keys)["TotalCount"].sum()

        common = country_g.index.intersection(county_g.index)
        if len(common) == 0:
            return None

        dav_list = [float(country_g.loc[idx]) for idx in common]
        oav_list = [float(county_g.loc[idx]) for idx in common]
        sc = sum(dav_list) if sum(dav_list) != 0 else (max(dav_list) if dav_list else 0.0)
        return {"dav": dav_list, "oav": oav_list, "sc": float(sc)}

    return None

OPS_BINARY = {
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide":   lambda a, b: (a / b) if b != 0 else math.nan,
}

def eval_expr(expr: Any, env: Dict[str, Any]) -> float:
    if isinstance(expr, (int, float)):
        return float(expr)
    if isinstance(expr, str):
        if expr in env:
            return float(env[expr])
        try:
            return float(expr)
        except ValueError:
            raise KeyError(f"Unknown variable '{expr}'")

    if isinstance(expr, dict):
        op = expr.get("operator")
        if not op:
            raise ValueError(f"Missing operator: {expr}")

        if op == "add":
            if "terms" in expr:
                return sum(float(eval_expr(t, env)) for t in expr["terms"])
            return float(eval_expr(expr["left"], env)) + float(eval_expr(expr["right"], env))

        if op in OPS_BINARY:
            return OPS_BINARY[op](float(eval_expr(expr["left"], env)), float(eval_expr(expr["right"], env)))

        if op == "identity":
            return float(eval_expr(expr.get("of", expr.get("operand")), env))

        if op == "abs_diff":
            return abs(float(eval_expr(expr["left"], env)) - float(eval_expr(expr["right"], env)))

        if op == "sum":
            if "terms" in expr:
                return sum(float(eval_expr(t, env)) for t in expr["terms"])
            if "of" in expr:
                template = expr["of"]
                dav = env.get("dav")
                oav = env.get("oav")
                if isinstance(dav, (int, float)) and isinstance(oav, (int, float)):
                    local = dict(env)
                    local["dav_i"] = float(dav)
                    local["oav_i"] = float(oav)
                    return float(eval_expr(template, local))
                if isinstance(dav, (list, tuple, pd.Series)) and isinstance(oav, (list, tuple, pd.Series)):
                    n = min(len(dav), len(oav))
                    tot = 0.0
                    for i in range(n):
                        local = dict(env)
                        local["dav_i"] = float(dav[i])
                        local["oav_i"] = float(oav[i])
                        local["i"] = i + 1
                        local["n"] = n
                        tot += float(eval_expr(template, local))
                    return tot
                return math.nan
            raise ValueError("sum requires 'terms' or 'of'")

        if op == "conditional":
            for cond in expr.get("conditions", []):
                if "else" in cond:
                    return float(eval_expr(cond["else"], env))
                cond_str = cond.get("if") or cond.get("elif")
                if not cond_str:
                    continue
                if eval(cond_str, {}, env):
                    return float(eval_expr(cond["then"], env))
            return math.nan

        raise ValueError(f"Unknown operator: {op}")

    raise TypeError(f"Unsupported expression type: {type(expr)}")

def apply_intermediate(metric: Dict[str, Any], env: Dict[str, Any]) -> None:
    interm = metric.get("intermediate_calculation")
    if not interm:
        return
    steps = interm if isinstance(interm, list) else [interm]
    for step in steps:
        env[step["assign"]] = float(eval_expr(step["expression"], env))

def build_env(
    df: pd.DataFrame,
    context: str,
    input_symbols: List[str],
    prompt_defs: Dict[str, Any],
    hf_pipe,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    env: Dict[str, Any] = {}
    debug: Dict[str, Any] = {"llm_raw": {}, "auto": {}}

    # structural
    env["nr"] = float(len(df))
    env["nc"] = float(len(df.columns))
    env["ncl"] = env["nr"] * env["nc"]

    # missing cells/rows
    n_incomplete_cells = df.isna().sum().sum() + df.eq("").sum().sum()
    env["ic"] = float(n_incomplete_cells)
    env["nce"] = float(n_incomplete_cells)  # heuristic
    nir_bool = df.isna().any(axis=1) | df.eq("").any(axis=1)
    env["nir"] = float(nir_bool.sum())

    # current date
    now = pd.Timestamp.now(tz=timezone.utc)
    env["cd"] = datetime_to_days(now)
    debug["auto"]["cd"] = str(now.date())

    # best 'data date' column for sd/edp and ncr
    _, date_col, parsed = find_best_datetime_column(df, name_keywords=("statisticsdate", "date"))
    if date_col is not None and parsed is not None:
        mn = parsed.min()
        mx = parsed.max()
        env["sd"] = datetime_to_days(mn)
        env["edp"] = datetime_to_days(mx)
        debug["auto"]["sd_col"] = date_col
        debug["auto"]["sd"] = str(mn.date()) if not pd.isna(mn) else None
        debug["auto"]["edp"] = str(mx.date()) if not pd.isna(mx) else None

        n_current = float((parsed == mx).sum())
        env["ncr"] = float(env["nr"] - n_current)
        debug["auto"]["ncr_definition"] = "rows_not_at_max_date"
        debug["auto"]["max_date"] = str(mx.date()) if not pd.isna(mx) else None

    # publication date dp
    _, mod_col, mod_parsed = find_best_datetime_column(df, name_keywords=("modified", "updated", "last"))
    if mod_col is not None and mod_parsed is not None:
        dp_dt = mod_parsed.max()
        env["dp"] = datetime_to_days(dp_dt)
        debug["auto"]["dp_col"] = mod_col
        debug["auto"]["dp"] = str(dp_dt.date()) if not pd.isna(dp_dt) else None

    # aggregation inputs (optional)
    agg = derive_aggregation_pairs(df)
    if agg is not None:
        env["dav"] = agg["dav"]
        env["oav"] = agg["oav"]
        env["sc"] = agg["sc"]
        debug["auto"]["aggregation_pairs"] = len(agg["dav"])

    # LLM-derived symbols
    N = int(env["nc"])
    for sym in input_symbols:
        if sym in env:
            continue
        if sym not in prompt_defs or hf_pipe is None:
            continue

        val, raw = infer_symbol(sym, context, N, prompt_defs, hf_pipe)
        debug["llm_raw"][sym] = raw

        if val is None:
            continue

        if prompt_defs[sym].get("type") == "date":
            dt = pd.to_datetime(val, errors="coerce")
            if pd.isna(dt):
                continue
            env[sym] = datetime_to_days(dt)
        else:
            env[sym] = float(val)

    return env, debug

def compute_metrics(
    df: pd.DataFrame,
    vetro_dict: Dict[str, Any],
    context: str,
    input_symbols: List[str],
    prompt_defs: Dict[str, Any],
    hf_pipe,
) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, Any]]:
    env, debug = build_env(df, context, input_symbols, prompt_defs, hf_pipe)
    results: Dict[str, float] = {}

    for dim, metrics in vetro_dict.items():
        for mname, metric in metrics.items():
            formula = metric.get("formula")
            norm = metric.get("normalization")
            if not formula or not norm:
                continue
            key = f"{dim}.{mname}"
            try:
                apply_intermediate(metric, env)
                raw = float(eval_expr(formula["expression"], env))
                env[formula["assign"]] = raw
                norm_val = float(eval_expr(norm["expression"], env))
                env[norm["assign"]] = norm_val
                results[key] = norm_val
            except Exception:
                results[key] = math.nan

    return results, env, debug
