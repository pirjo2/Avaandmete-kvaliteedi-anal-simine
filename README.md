# Open Data Quality (Vetrò 2016) — YAML-driven + optional Hugging Face LLM

This repo contains a small Streamlit web app that:
- loads a user-uploaded open data file (CSV / Excel),
- computes dataset-level quality metrics using a YAML implementation of the **Vetrò et al. (2016)** framework,
- optionally uses a Hugging Face model (LLM) to estimate metadata-related inputs via prompts defined in YAML,
- shows results as a table + bar chart, and lets you download results as CSV.

## Files
- `configs/formulas.yaml` — Vetrò methodology (dimensions, metrics, formulas)
- `configs/prompts.yaml` — prompt templates for symbols that can be inferred by an LLM
- `core/` — Python implementation (YAML loader, expression evaluator, metrics, LLM helpers)
- `app.py` — Streamlit UI

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy options
- Streamlit Community Cloud
- Hugging Face Spaces (Streamlit)
- Render / Railway

## Notes
- Large datasets (hundreds of thousands of rows) can be slow. Use the **Max rows** setting.
- Some Vetrò metrics require metadata not present in a raw CSV/Excel file. In that case:
  - metrics may be `NaN`, or
  - enabling the LLM can estimate some inputs based on the dataset context.
