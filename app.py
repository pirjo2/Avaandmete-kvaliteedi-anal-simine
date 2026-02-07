from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from core.pipeline import run_quality_assessment

# ---- Paths (repo root) ----
FORMULAS = "configs/formulas.yaml"
PROMPTS = "configs/prompts.yaml"

st.set_page_config(page_title="Open Data Quality Assessment", layout="wide")

st.title("Open Data Quality Assessment (Vetrò-style + optional AI)")
st.write(
    "Upload a dataset file (CSV / Excel / JSON). The app computes Vetrò-style quality metrics. "
    "Optional AI inference helps with symbols that are hard to compute from the table alone."
)

uploaded = st.file_uploader("Dataset file", type=["csv", "xlsx", "xls", "json"])

if uploaded is not None:
    st.session_state["file_name"] = uploaded.name
    st.session_state["file_bytes"] = uploaded.getvalue()

file_bytes = st.session_state.get("file_bytes", None)
file_name = st.session_state.get("file_name", "")

if not file_bytes:
    st.stop()

ext = Path(file_name).suffix.lower().lstrip(".")

dataset_description = st.text_area(
    "Optional: dataset description / portal metadata (paste text)",
    value=st.session_state.get("dataset_description", ""),
    height=120,
    placeholder="Paste a short description, license text, publisher info, update notes, etc.",
)
st.session_state["dataset_description"] = dataset_description

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    max_rows = st.number_input(
        "Max rows to load (0 = all)",
        min_value=0,
        value=int(st.session_state.get("max_rows", 0)),
        step=10_000,
    )
    st.session_state["max_rows"] = int(max_rows)

with col2:
    use_llm = st.checkbox("Use AI (Hugging Face model)", value=bool(st.session_state.get("use_llm", True)))
    st.session_state["use_llm"] = bool(use_llm)

with col3:
    hf_model_name = st.selectbox(
        "Model",
        options=[
            "google/flan-t5-small",
            "google/flan-t5-base",
            "google/flan-t5-large",
        ],
        index=1,
    )

def load_df(file_bytes: bytes, ext: str, max_rows: int) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    if ext == "csv":
        try:
            df = pd.read_csv(bio, nrows=(None if max_rows == 0 else max_rows))
        except UnicodeDecodeError:
            bio.seek(0)
            df = pd.read_csv(bio, encoding="latin-1", nrows=(None if max_rows == 0 else max_rows))
        return df
    if ext in ("xlsx", "xls"):
        return pd.read_excel(bio, nrows=(None if max_rows == 0 else max_rows))
    if ext == "json":
        return pd.read_json(bio)
    raise ValueError(f"Unsupported file type: {ext}")

run_btn = st.button("Analyze", type="primary")

if run_btn:
    df = load_df(file_bytes, ext, int(max_rows))

    if df.empty:
        st.error("The file was loaded, but the table is empty.")
        st.stop()

    with st.spinner("Computing metrics..."):
        _, metrics_df, details = run_quality_assessment(
            df=df,
            formulas_yaml_path=FORMULAS,
            prompts_yaml_path=PROMPTS,
            use_llm=use_llm,
            hf_model_name=hf_model_name,
            dataset_description=dataset_description,
            file_name=file_name,
            file_ext=ext,
        )

    st.session_state["metrics_df"] = metrics_df
    st.session_state["details"] = details
    st.session_state["df_preview"] = df.head(30)

metrics_df = st.session_state.get("metrics_df", None)
details = st.session_state.get("details", None)
df_preview = st.session_state.get("df_preview", None)

if metrics_df is None or details is None:
    st.info("Click **Analyze** to compute metrics.")
    st.stop()

st.subheader("Preview")
st.dataframe(df_preview, use_container_width=True)

st.subheader("Quality metrics")
show_df = metrics_df.copy()
show_df["value_display"] = show_df["value"]

st.dataframe(show_df[["dimension", "metric", "metric_label", "value", "value_display"]], use_container_width=True)

chart_df = show_df.dropna(subset=["value_display"]).copy()
fig = px.bar(
    chart_df,
    x="metric_label",
    y="value_display",
    color="dimension",
    title="Quality metric scores",
)
st.plotly_chart(fig, width="stretch")

csv_metrics = metrics_df.to_csv(index=False).encode("utf-8")
st.download_button("Download metrics CSV", data=csv_metrics, file_name="metrics.csv", mime="text/csv")

st.subheader("Debug (symbols)")
sym_rows = []
sym_vals = details.get("symbol_values", {})
sym_src = details.get("symbol_source", {})
llm_conf = details.get("llm_confidence", {})
llm_raw = details.get("llm_raw", {})
llm_ev = details.get("llm_evidence", {})

for sym in sorted(set(sym_vals.keys()) | set(sym_src.keys())):
    sym_rows.append(
        {
            "symbol": sym,
            "value (None=did not work)": sym_vals.get(sym),
            "source": sym_src.get(sym),
            "confidence": llm_conf.get(sym, None),
            "evidence": llm_ev.get(sym, ""),
            "raw": llm_raw.get(sym, ""),
        }
    )

sym_df = pd.DataFrame(sym_rows)
st.dataframe(sym_df, use_container_width=True)

st.download_button(
    "Download symbol debug CSV",
    data=sym_df.to_csv(index=False).encode("utf-8"),
    file_name="symbol_debug.csv",
    mime="text/csv",
)

with st.expander("Auto inputs (derived from data)"):
    st.json(details.get("auto_inputs", {}))

with st.expander("LLM context used"):
    st.code(details.get("context_used", ""), language="text")
