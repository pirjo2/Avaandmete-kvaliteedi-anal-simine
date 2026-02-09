from __future__ import annotations

import io
import math
import os
from typing import Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from core.pipeline import run_quality_assessment

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
FORMULAS = os.path.join(REPO_ROOT, "configs", "formulas.yaml")
PROMPTS = os.path.join(REPO_ROOT, "configs", "prompts.yaml")


def _read_dataset(uploaded_file) -> Tuple[pd.DataFrame, str, str]:
    name = uploaded_file.name
    ext = os.path.splitext(name)[1].lower().lstrip(".")
    data = uploaded_file.getvalue()

    if ext in ("csv", "txt"):
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(io.BytesIO(data), encoding=enc), ext, name
            except Exception:
                pass
        return pd.read_csv(io.BytesIO(data)), ext, name

    if ext in ("xlsx", "xls"):
        return pd.read_excel(io.BytesIO(data)), ext, name

    if ext == "json":
        return pd.read_json(io.BytesIO(data)), ext, name

    raise ValueError(f"Unsupported file type: .{ext}")


def _df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].apply(lambda v: "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v))
    return out


def _nan_to_none(x):
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


st.set_page_config(page_title="Open Data Quality Assessment", layout="wide")

st.title("Open Data Quality Assessment (Vetrò methodology)")
st.caption("Upload a dataset and compute quality metrics. Optional: use an open-source LLM for heuristic symbols.")

uploaded = st.file_uploader("Upload dataset (CSV / XLSX / JSON)", type=["csv", "txt", "xlsx", "xls", "json"])

colA, colB = st.columns([2, 1], gap="large")

with colA:
    dataset_description = st.text_area(
        "Dataset description (optional, improves metadata-related symbols)",
        placeholder="Paste portal metadata (title, publisher, license, update frequency, etc.).",
        height=120,
    )

with colB:
    use_llm = st.checkbox("Use LLM for heuristic symbols", value=True)
    confidence_weighting = st.checkbox("Weight LLM numeric answers by confidence", value=True)

    model_options = [
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/mt5-small",
    ]
    hf_model_name = st.selectbox("HF model", options=model_options, index=1)

    st.caption("First run may be slow if the model needs to download and load.")

run_btn = st.button("Run assessment", type="primary", disabled=(uploaded is None))

if "results" not in st.session_state:
    st.session_state["results"] = None

if run_btn and uploaded is not None:
    try:
        df, ext, fname = _read_dataset(uploaded)

        _, metrics_df, details = run_quality_assessment(
            df=df,
            formulas_yaml_path=FORMULAS,
            prompts_yaml_path=PROMPTS,
            use_llm=use_llm,
            hf_model_name=hf_model_name,
            dataset_description=dataset_description or "",
            file_ext=ext,
            file_name=fname,
            confidence_weighting=confidence_weighting,
        )

        st.session_state["results"] = {
            "df_head": df.head(25),
            "metrics_df": metrics_df,
            "details": details,
        }
    except Exception as e:
        st.error(f"Failed to run assessment: {e}")
        st.stop()

res = st.session_state.get("results")
if res is None:
    st.info("Upload a dataset and click **Run assessment**.")
    st.stop()

metrics_df: pd.DataFrame = res["metrics_df"]
details = res["details"]
symbol_table: pd.DataFrame = details.get("symbol_table", pd.DataFrame())

st.subheader("Computed quality metrics")

metrics_show = metrics_df.copy()
metrics_show["value"] = metrics_show["value"].apply(_nan_to_none)
st.dataframe(_df_for_display(metrics_show), use_container_width=True, hide_index=True)

plot_df = metrics_df.copy()
plot_df = plot_df[pd.to_numeric(plot_df["value"], errors="coerce").notna()].copy()
plot_df["value"] = pd.to_numeric(plot_df["value"], errors="coerce")

if len(plot_df):
    fig = px.bar(plot_df, x="metric_id", y="value", hover_data=["dimension", "metric"])
    fig.update_layout(xaxis_title="Metric", yaxis_title="Value (normalized)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No numeric metrics to plot (many symbols might be missing).")

st.download_button(
    "Download metrics CSV",
    data=metrics_df.to_csv(index=False).encode("utf-8"),
    file_name="metrics.csv",
    mime="text/csv",
)

st.subheader("Per-symbol diagnostics (inputs to formulas)")

if not symbol_table.empty:
    sym_disp = symbol_table.copy()
    sym_disp["value"] = sym_disp["value"].apply(lambda v: "N/A" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v))
    st.dataframe(_df_for_display(sym_disp), use_container_width=True, hide_index=True)

    st.download_button(
        "Download symbols CSV",
        data=sym_disp.to_csv(index=False).encode("utf-8"),
        file_name="symbols.csv",
        mime="text/csv",
    )
else:
    st.info("No symbol diagnostics available.")

st.subheader("Dataset preview")
st.dataframe(_df_for_display(res["df_head"]), use_container_width=True, hide_index=True)
