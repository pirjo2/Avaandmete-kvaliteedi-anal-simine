from __future__ import annotations

import os
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import plotly.express as px
import streamlit as st

from core.pipeline import run_quality_assessment

# --- Paths --- #
FORMULAS_YAML = "configs/formulas.yaml"
PROMPTS_YAML = "configs/prompts.yaml"

DEFAULT_MODEL = "google/flan-t5-base"
MODEL_OPTIONS = [
    "google/flan-t5-base",
    "google/flan-t5-small",
]


# --- Page config --- #
st.set_page_config(
    page_title="Open Data Quality Assessment",
    layout="wide",
)


st.title("Open Data Quality Assessment (Vetrò et al. 2016)")

st.markdown(
    """
Upload an open data table (CSV or Excel) and this tool will approximate
data quality metrics following Vetrò et al.'s framework:
traceability, currentness, completeness, compliance, understandability and accuracy.

The AI assistance is used only for metadata-like signals (e.g., publisher, language, coverage),
while numeric indicators are derived directly from the data.
"""
)

# --- File upload --- #
uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "tsv", "txt", "xls", "xlsx"],
)

col_settings1, col_settings2, col_settings3 = st.columns(3)
with col_settings1:
    row_limit = st.number_input(
        "Row limit (0 = all rows)",
        min_value=0,
        value=500_000,
        step=10_000,
    )
with col_settings2:
    use_llm = st.checkbox("Use AI assistance for metadata (beta)", value=True)
with col_settings3:
    hf_model_name = st.selectbox(
        "Hugging Face model",
        options=MODEL_OPTIONS,
        index=0,
        disabled=not use_llm,
    )

run_btn = st.button(
    "Run assessment",
    type="primary",
    disabled=uploaded_file is None,
)

if run_btn and uploaded_file is not None:
    try:
        name = uploaded_file.name
        ext = os.path.splitext(name)[1].lower()

        # Load dataframe
        if ext in [".csv", ".tsv", ".txt"]:
            df = pd.read_csv(uploaded_file, sep=None, engine="python")
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(uploaded_file)
        else:
            st.error(f"Unsupported file type: {ext}")
            st.stop()

        if row_limit and row_limit > 0:
            df = df.head(row_limit)

        st.subheader("Preview of the dataset")
        st.dataframe(df.head(20), width="stretch")
        st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns used for metrics.")

        with st.spinner("Computing quality metrics..."):
            metrics_df, details = run_quality_assessment(
                df=df,
                formulas_yaml_path=FORMULAS_YAML,
                prompts_yaml_path=PROMPTS_YAML,
                use_llm=use_llm,
                hf_model_name=hf_model_name,
                file_ext=ext,
            )

        st.subheader("Quality metrics")

        if metrics_df.empty or metrics_df["value"].dropna().empty:
            st.info("No metrics could be computed.")
        else:
            metrics_non_null = metrics_df.dropna(subset=["value"]).copy()
            metrics_non_null["value_clamped"] = metrics_non_null["value"].clip(0.0, 1.0)

            fig = px.bar(
                metrics_non_null,
                x="metric_label",
                y="value_clamped",
                color="dimension",
                range_y=[0, 1],
                labels={
                    "metric_label": "Metric",
                    "value_clamped": "Normalised value (0–1)",
                    "dimension": "Dimension",
                },
            )
            fig.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig, width="stretch")

            st.dataframe(
                metrics_non_null[
                    ["dimension", "metric_label", "value", "metric_id"]
                ].sort_values(["dimension", "metric_id"]),
                width="stretch",
            )

        # --- Debug / explanations --- #
        with st.expander("Debug: auto-derived inputs and AI inferences"):
            st.markdown("**Auto-derived inputs (from the table only):**")
            st.json(details.get("auto_inputs", {}))

            symbol_values = details.get("symbol_values", {})
            if not symbol_values:
                st.write("No symbol debug information available.")
            else:
                src = details.get("symbol_source", {})
                conf = details.get("llm_confidence", {})
                evid = details.get("llm_evidence", {})
                rows = []
                for sym in sorted(symbol_values.keys()):
                    rows.append(
                        {
                            "symbol": sym,
                            "value": symbol_values.get(sym),
                            "source": src.get(sym, ""),
                            "llm_confidence": conf.get(sym),
                            "llm_evidence": evid.get(sym, ""),
                        }
                    )
                st.dataframe(pd.DataFrame(rows), width="stretch")

    except Exception as e:
        st.exception(e)
