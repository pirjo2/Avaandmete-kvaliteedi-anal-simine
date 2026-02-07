import streamlit as st
import pandas as pd
import plotly.express as px

from core.pipeline import run_quality_assessment

st.set_page_config(page_title="Open Data Quality (Vetrò + AI)", layout="wide")
st.title("Open Data Quality Assessment (Vetrò 2016 + YAML + optional AI)")

with st.expander("How this works", expanded=False):
    st.markdown(
        """
Upload an open data file (CSV or Excel).  
The app computes dataset-level quality metrics using a YAML-driven implementation of the Vetrò et al. (2016) framework.  

**Important:** many Vetrò inputs relate to *metadata* (publisher, licence, description, coverage, update info).  
A plain CSV often does not include that, so those parts may be **unknown** unless you paste portal metadata in the description box.
"""
    )

# Pretty labels for chart/table (you can expand this later)
PRETTY_METRIC_NAMES = {
    "traceability.track_of_creation": "Traceability: track of creation",
    "traceability.track_of_updates": "Traceability: track of updates",
    "currentness.percentage_of_current_rows": "Currentness: % current rows",
    "currentness.delay_in_publication": "Currentness: delay in publication",
    "currentness.delay_after_expiration": "Currentness: delay after expiration",
    "completeness.percentage_of_complete_cells": "Completeness: % complete cells",
    "completeness.percentage_of_complete_rows": "Completeness: % complete rows",
    "compliance.percentage_of_standardized_columns": "Compliance: % standardized columns",
    "compliance.egms_compliance": "Compliance: eGMS compliance",
    "compliance.five_stars_open_data": "Compliance: 5-star open data",
    "understandability.percentage_of_columns_with_metadata": "Understandability: % columns with metadata",
    "understandability.percentage_of_columns_in_comprehensible_format": "Understandability: % columns comprehensible",
    "accuracy.percentage_of_syntactically_accurate_cells": "Accuracy: % syntactically accurate cells",
    "accuracy.accuracy_in_aggregation": "Accuracy: aggregation accuracy",
}

FORMULAS = "configs/formulas.yaml"
PROMPTS = "configs/prompts.yaml"

# -------- UI --------
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    uploaded = st.file_uploader("Upload dataset (CSV / XLSX)", type=["csv", "xlsx", "xls"])
    description = st.text_area(
        "Optional: paste portal metadata / description (helps LLM estimate metadata-related symbols)",
        height=120,
        placeholder="Paste dataset description, publisher, licence, update info here (optional).",
    )

with col2:
    use_llm = st.toggle("Use Hugging Face LLM", value=False)

    model_choices = [
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large",
    ]
    hf_model = st.selectbox("HF model", options=model_choices, index=1, disabled=not use_llm)
    custom_model = st.text_input("Or custom model name", value="", disabled=not use_llm)
    hf_model_name = custom_model.strip() if (use_llm and custom_model.strip()) else hf_model

with col3:
    max_rows = st.number_input(
        "Max rows to process (0 = all)",
        min_value=0,
        value=0,
        step=10000,
        help="0 = use the full dataset (no hard limit).",
    )
    run_btn = st.button("Run assessment", type="primary", width="stretch")

# -------- Helpers --------
def read_uploaded(file) -> pd.DataFrame:
    name = file.name.lower()

    if name.endswith(".csv"):
        # Try common encodings
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                return pd.read_csv(file, encoding=enc)
            except UnicodeDecodeError:
                file.seek(0)
                continue
        file.seek(0)
        return pd.read_csv(file, encoding_errors="replace")

    # Excel
    return pd.read_excel(file)

def build_symbol_debug_table(details: dict) -> pd.DataFrame:
    # support both "new" and "old" detail shapes
    input_symbols = details.get("input_symbols", [])

    auto_inputs = (
        details.get("auto_inputs")
        or details.get("debug", {}).get("auto", {})
        or {}
    )

    llm_raw = (
        details.get("llm_raw")
        or details.get("debug", {}).get("llm_raw", {})
        or {}
    )

    llm_conf = (
        details.get("llm_confidence")
        or details.get("debug", {}).get("llm_confidence", {})
        or {}
    )

    llm_evidence = (
        details.get("llm_evidence")
        or details.get("debug", {}).get("llm_evidence", {})
        or {}
    )

    symbol_values = details.get("symbol_values") or {}
    symbol_source = details.get("symbol_source") or {}

    # If symbol_values absent but env exists, recover values from env
    env = details.get("env") or {}
    if not symbol_values and env and input_symbols:
        symbol_values = {s: env.get(s) for s in input_symbols}

    # If still no input_symbols, infer from union of seen keys
    if not input_symbols:
        input_symbols = sorted(
            set(symbol_values.keys()) | set(llm_raw.keys()) | set(auto_inputs.keys())
        )

    rows = []
    for s in input_symbols:
        v = symbol_values.get(s, None)

        src = symbol_source.get(s, "")
        if not src:
            if s in auto_inputs:
                src = "auto"
            elif s in llm_raw:
                src = "llm"
            else:
                src = "fail"

        rows.append(
            {
                "symbol": s,
                "value": v,  # None = did not work, 0.0 = real zero
                "source": src,
                "confidence": llm_conf.get(s, None),
                "evidence": llm_evidence.get(s, ""),
                "raw": llm_raw.get(s, ""),
            }
        )

    return pd.DataFrame(rows)

# -------- Main flow --------
if uploaded is None:
    st.info("Upload a dataset file to start.")
    st.stop()

df = read_uploaded(uploaded)

# max_rows = 0 => all rows
if max_rows and int(max_rows) > 0:
    df = df.head(int(max_rows)).copy()

st.subheader("Preview")
st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
st.dataframe(df.head(50), width="stretch")

if not run_btn:
    st.stop()

with st.spinner("Computing metrics..."):
    try:
        _, metrics_df, details = run_quality_assessment(
            df=df,
            formulas_yaml_path=FORMULAS,
            prompts_yaml_path=PROMPTS,
            dataset_description=description or "",
            use_llm=use_llm,
            hf_model_name=hf_model_name,
        )
    except Exception as e:
        st.error("Run failed. See error below.")
        st.exception(e)
        st.stop()

# Add metric_id + pretty label
metrics_df = metrics_df.copy()
metrics_df["metric_id"] = metrics_df["dimension"] + "." + metrics_df["metric"]
metrics_df["metric_label"] = metrics_df["metric_id"].map(PRETTY_METRIC_NAMES).fillna(metrics_df["metric_id"])

st.subheader("Metric results")
st.dataframe(metrics_df[["dimension", "metric", "value", "metric_label"]], width="stretch")

st.subheader("Bar chart (metric scores)")
plot_df = metrics_df.dropna(subset=["value"]).copy()
plot_df = plot_df.sort_values("value", ascending=True)

fig = px.bar(
    plot_df,
    x="value",
    y="metric_label",
    orientation="h",
)
fig.update_layout(xaxis_title="Score (normalized)", yaxis_title="Metric")
st.plotly_chart(fig, width="stretch")

st.subheader("Download")
st.download_button(
    "Download metric table (CSV)",
    data=metrics_df.to_csv(index=False).encode("utf-8"),
    file_name="quality_metrics.csv",
    mime="text/csv",
)

with st.expander("Debug info (inputs used)", expanded=False):
    # Show auto inputs plainly
    auto_inputs = details.get("auto_inputs") or details.get("debug", {}).get("auto", {}) or {}
    st.markdown("**Auto inputs (derived from data):**")
    st.json(auto_inputs)

    # Show per-symbol table (value None = did not work; 0.0 = real 0)
    sym_df = build_symbol_debug_table(details)
    st.markdown("**Per-symbol results (None = did not work; 0.0 = real zero):**")
    st.dataframe(sym_df, width="stretch")

    st.download_button(
        "Download debug table (CSV)",
        data=sym_df.to_csv(index=False).encode("utf-8"),
        file_name="symbol_debug.csv",
        mime="text/csv",
    )
