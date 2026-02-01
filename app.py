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
Optionally, you can enable an LLM (Hugging Face) to estimate metadata-related inputs from the dataset context (column names, samples, and description).
"""
    )

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    uploaded = st.file_uploader("Upload dataset (CSV / XLSX)", type=["csv", "xlsx", "xls"])
    description = st.text_area("Optional: dataset description / notes (used for LLM prompts)", height=100, placeholder="Paste portal description here (optional).")
with col2:
    use_llm = st.toggle("Use Hugging Face LLM", value=False)
    hf_model = st.text_input("HF model name", value="google/flan-t5-base", help="Example: google/flan-t5-base")
with col3:
    max_rows = st.number_input("Max rows to process (0 = all)", min_value=0, value=0, step=10000)
    run_btn = st.button("Run assessment", type="primary", use_container_width=True)

FORMULAS = "configs/formulas.yaml"
PROMPTS = "configs/prompts.yaml"

def read_uploaded(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        # try encodings
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                return pd.read_csv(file, encoding=enc)
            except UnicodeDecodeError:
                file.seek(0)
                continue
        file.seek(0)
        return pd.read_csv(file, encoding_errors="replace")
    return pd.read_excel(file)

if uploaded is not None:
    df = read_uploaded(uploaded)
    if max_rows and max_rows > 0:
        df = df.head(int(max_rows)).copy()

    st.subheader("Preview")
    st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    st.dataframe(df, width="stretch")

    if run_btn:
        with st.spinner("Computing metrics..."):
            _, metrics_df, details = run_quality_assessment(
                df=df,
                formulas_yaml_path=FORMULAS,
                prompts_yaml_path=PROMPTS,
                dataset_description=description or "",
                use_llm=use_llm,
                hf_model_name=hf_model,
            )

        st.subheader("Metric results")
        st.dataframe(metrics_df, use_container_width=True)

        st.subheader("Bar chart (metric scores)")
        plot_df = metrics_df.dropna(subset=["value"]).copy()
        plot_df["metric_full"] = plot_df["dimension"] + "." + plot_df["metric"]
        plot_df = plot_df.sort_values("value", ascending=False)

        fig = px.bar(plot_df, x="metric_full", y="value")
        fig.update_layout(xaxis_title="Metric", yaxis_title="Score (normalized)", xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Download")
        st.download_button(
            "Download metric table (CSV)",
            data=metrics_df.to_csv(index=False).encode("utf-8"),
            file_name="quality_metrics.csv",
            mime="text/csv",
        )

        with st.expander("Debug info (inputs used)", expanded=False):
            st.json({
                "input_symbols": details.get("input_symbols", []),
                "auto_inputs": details.get("debug", {}).get("auto", {}),
                "llm_raw": details.get("debug", {}).get("llm_raw", {}),
            })
else:
    st.info("Upload a dataset file to start.")
