import io
import yaml
import pandas as pd
import streamlit as st
import plotly.express as px

from core.pipeline import run_quality_assessment

st.set_page_config(page_title="Open data quality (Vetrò + LLM)", layout="wide")

st.title("Open data quality assessment (Vetrò + AI)")

st.markdown(
    "Upload a dataset (CSV/XLSX). The app computes Vetrò-style dataset-level quality metrics. "
    "Optional: enable LLM-based inference for metadata-like symbols."
)

# Paths inside the repo
FORMULAS = "configs/formulas.yaml"
PROMPTS = "configs/prompts.yaml"

# --- Load formulas YAML once (for labels etc.) ---
with open(FORMULAS, "r", encoding="utf-8") as f:
    formulas_cfg = yaml.safe_load(f)

labels_map = formulas_cfg.get("labels", {})  # top-level labels (we add to formulas yaml)

# --- Sidebar controls ---
st.sidebar.header("Settings")

use_llm = st.sidebar.checkbox("Use Hugging Face LLM", value=False)

MODEL_OPTIONS = [
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/mt5-small",
    # You can add more models later
    # "tiiuae/falcon-7b-instruct",  # example (might be too heavy for CPU)
]
hf_model = st.sidebar.selectbox("Hugging Face model", MODEL_OPTIONS, index=0)

max_rows = st.sidebar.number_input("Max rows to load (performance)", min_value=1000, max_value=500000, value=200000, step=1000)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Large models are slow on Streamlit Cloud (CPU). Start with flan-t5-base.")

# Optional metadata inputs for LLM context (helps the “human-like guessing”)
st.subheader("Optional dataset metadata (helps AI)")
colA, colB, colC = st.columns(3)
with colA:
    meta_title = st.text_input("Title", value="")
with colB:
    meta_publisher = st.text_input("Publisher", value="")
with colC:
    meta_licence = st.text_input("Licence", value="")

meta_description = st.text_area("Description", value="", height=100)

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

def load_df(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        data = file.getvalue()
        # Try UTF-8 first, then latin1 fallback (encoding issues are common)
        for enc in ("utf-8", "utf-8-sig", "latin1"):
            try:
                return pd.read_csv(io.BytesIO(data), encoding=enc)
            except Exception:
                pass
        # last resort
        return pd.read_csv(io.BytesIO(data), engine="python")
    else:
        return pd.read_excel(file)

run_btn = st.button("Run quality assessment", type="primary", disabled=uploaded is None)

if run_btn and uploaded is not None:
    with st.spinner("Loading dataset..."):
        df = load_df(uploaded)
        if len(df) > int(max_rows):
            df = df.head(int(max_rows)).copy()

    st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    st.dataframe(df.head(20), width="stretch")

    extra_meta = {
        "title": meta_title.strip(),
        "description": meta_description.strip(),
        "publisher": meta_publisher.strip(),
        "licence": meta_licence.strip(),
    }

    with st.spinner("Computing metrics..."):
        _, metrics_df, details = run_quality_assessment(
            df=df,
            formulas_yaml_path=FORMULAS,
            prompts_yaml_path=PROMPTS,
            use_llm=use_llm,
            hf_model_name=hf_model,
            extra_metadata=extra_meta,
        )

    st.subheader("Results (table)")
    st.dataframe(metrics_df, width="stretch")

    # --- Pretty labels for chart (2) ---
    # metrics_df has columns: dimension, metric, value, metric_id
    chart_df = metrics_df.dropna(subset=["value"]).copy()
    chart_df["label"] = chart_df["metric_id"].map(labels_map).fillna(chart_df["metric_id"])
    chart_df = chart_df.sort_values("value", ascending=False)

    st.subheader("Quality scores (bar chart)")
    fig = px.bar(chart_df, x="label", y="value", color="dimension", text="value")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        xaxis_title="Metric",
        yaxis_title="Score",
        xaxis_tickangle=-30,
        margin=dict(t=40, b=140),
        legend_title_text="Dimension",
    )
    st.plotly_chart(fig, width="stretch")

    # --- Debug panels ---
    with st.expander("Debug: inputs / LLM raw outputs"):
        st.write("auto_inputs", details.get("auto_inputs", {}))
        st.write("llm_confidence", details.get("llm_confidence", {}))
        st.write("llm_raw", details.get("llm_raw", {}))