from __future__ import annotations

import io
import hashlib
import os
from typing import Tuple

import pandas as pd
import streamlit as st

from core.pipeline import run_quality_assessment

# ---- Paths (repo root) ----
FORMULAS = "configs/formulas.yaml"
PROMPTS = "configs/prompts.yaml"

st.set_page_config(page_title="Open Data Quality Assessment", layout="wide")


def sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def sha16_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def load_dataset(uploaded_file, max_rows: int) -> Tuple[pd.DataFrame, str, str]:
    """
    Returns: df, file_name, file_ext
    max_rows=0 => load all rows
    """
    file_name = uploaded_file.name
    ext = os.path.splitext(file_name)[1].lower().lstrip(".")
    data = uploaded_file.getvalue()

    nrows = None if (not max_rows or max_rows <= 0) else int(max_rows)

    if ext in ("csv", "txt"):
        # Try a few encodings; keep it simple and robust.
        last_err = None
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
            try:
                df = pd.read_csv(
                    io.BytesIO(data),
                    encoding=enc,
                    low_memory=False,
                    nrows=nrows,
                )
                return df, file_name, ext
            except Exception as e:
                last_err = e
        raise ValueError(
            "Failed to read CSV (encoding issue). Try saving the file as UTF-8."
        ) from last_err

    if ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(data))
        if nrows is not None:
            df = df.head(nrows).copy()
        return df, file_name, ext

    if ext in ("json",):
        df = pd.read_json(io.BytesIO(data))
        if nrows is not None:
            df = df.head(nrows).copy()
        return df, file_name, ext

    raise ValueError(f"Unsupported file type .{ext}. Please use CSV/XLSX/JSON.")


def make_chart_df(metrics_df: pd.DataFrame) -> pd.DataFrame:
    out = metrics_df.copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])
    out["label"] = out.get("metric_label", out.get("metric_id", "")).fillna(out["metric_id"])
    return out[["label", "value", "dimension", "metric_id"]]


st.title("Open Data Quality Assessment (Vetrò + AI)")

with st.sidebar:
    st.header("Input")

    uploaded = st.file_uploader(
        "Upload a dataset (CSV / XLSX / JSON)",
        type=["csv", "txt", "xlsx", "xls", "json"],
    )

    max_rows = st.number_input("Max rows (0 = all rows)", min_value=0, value=0, step=1000)

    st.divider()
    st.header("AI (optional)")

    use_llm = st.toggle("Use a Hugging Face model", value=True)

    model_choices = [
        "google/flan-t5-small",
        "google/flan-t5-base",
        # A lightweight extra option (often OK on CPU, still not “instant”):
        "google/flan-t5-large",
    ]
    hf_model = st.selectbox("Model", options=model_choices, index=1)
    custom_model = st.text_input("Or type a custom HF model id (optional)", value="")
    if custom_model.strip():
        hf_model = custom_model.strip()

    st.divider()
    st.header("Extra context for AI (optional)")

    dataset_description = st.text_area(
        "Short description (1–5 sentences). Source, meaning of columns, time period, etc.",
        value="",
        height=120,
    )

run_btn = st.button("Run assessment", type="primary", width="stretch")

# Keep last results even if Streamlit reruns (e.g., after download)
if "results" not in st.session_state:
    st.session_state["results"] = {}
if "last_key" not in st.session_state:
    st.session_state["last_key"] = None

if uploaded is None and st.session_state["last_key"] is None:
    st.info("Upload a file and click **Run assessment**.")
    st.stop()

cache_key = None
if uploaded is not None:
    file_bytes = uploaded.getvalue()
    desc_hash = sha16_text(dataset_description)
    cache_key = f"{sha16(file_bytes)}:{int(max_rows)}:{int(use_llm)}:{hf_model}:{desc_hash}"

if run_btn:
    if uploaded is None:
        st.warning("Please upload a file first.")
        st.stop()

    with st.spinner("Loading file and computing metrics..."):
        df, file_name, ext = load_dataset(uploaded, int(max_rows))
        _, metrics_df, details = run_quality_assessment(
            df=df,
            formulas_yaml_path=FORMULAS,
            prompts_yaml_path=PROMPTS,
            use_llm=use_llm,
            hf_model_name=hf_model,
            dataset_description=dataset_description,
            file_name=file_name,
            file_ext=ext,
        )
        st.session_state["results"][cache_key] = (df, metrics_df, details)
        st.session_state["last_key"] = cache_key

key_to_show = cache_key if (cache_key in st.session_state["results"]) else st.session_state["last_key"]
if key_to_show is None or key_to_show not in st.session_state["results"]:
    st.warning("Click **Run assessment** to see results.")
    st.stop()

df, metrics_df, details = st.session_state["results"][key_to_show]

st.subheader("Preview")
st.dataframe(df.head(20), width="stretch")

st.subheader("Metrics")
cols = [c for c in ["dimension", "metric", "metric_id", "value", "metric_label"] if c in metrics_df.columns]
st.dataframe(metrics_df[cols], width="stretch")

chart_df = make_chart_df(metrics_df)
if not chart_df.empty:
    st.subheader("Bar chart")
    st.bar_chart(chart_df.set_index("label")["value"], height=380)

with st.expander("Debug (auto inputs, symbols, LLM raw)"):
    st.write("Auto inputs (derived from data):")
    st.json(details.get("auto_inputs", {}))

    sym_vals = details.get("symbol_values", {}) or {}
    sym_src = details.get("symbol_source", {}) or {}
    sym_conf = details.get("llm_confidence", {}) or {}
    sym_ev = details.get("llm_evidence", {}) or {}
    sym_raw = details.get("llm_raw", {}) or {}

    rows = []
    for k in sorted(set(sym_vals.keys()) | set(sym_src.keys())):
        v = sym_vals.get(k, None)
        rows.append(
            {
                "symbol": k,
                "value": v,
                "source": sym_src.get(k, ""),
                "confidence": sym_conf.get(k, None),
                "evidence": sym_ev.get(k, ""),
                "raw": (sym_raw.get(k, "") or "")[:500],  # keep UI snappy
            }
        )

    sym_df = pd.DataFrame(rows)

    # ---- Arrow-safe display: split numeric vs text ----
    sym_df["value_num"] = pd.to_numeric(sym_df["value"], errors="coerce")
    sym_df["value_text"] = sym_df["value"].apply(lambda x: "" if x is None else str(x))
    sym_df["confidence"] = pd.to_numeric(sym_df["confidence"], errors="coerce")
    sym_df = sym_df.drop(columns=["value"])

    st.dataframe(sym_df, width="stretch")
