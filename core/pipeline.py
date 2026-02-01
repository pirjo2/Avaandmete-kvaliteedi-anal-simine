from __future__ import annotations

from typing import Any, Dict, Tuple
import yaml
import pandas as pd

from core.llm import get_hf_pipe
from core.metrics_eval import compute_metrics

def run_quality_assessment(
    df: pd.DataFrame,
    formulas_yaml_path: str,
    prompts_yaml_path: str,
    use_llm: bool,
    hf_model_name: str,
    extra_metadata: Dict[str, str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if extra_metadata is None:
        extra_metadata = {"title": "", "description": "", "publisher": "", "licence": ""}

    with open(formulas_yaml_path, "r", encoding="utf-8") as f:
        formulas_cfg = yaml.safe_load(f)

    with open(prompts_yaml_path, "r", encoding="utf-8") as f:
        prompt_cfg = yaml.safe_load(f)

    hf_pipe = None
    if use_llm:
        hf_pipe = get_hf_pipe(hf_model_name)

    metrics_df, details = compute_metrics(
        df=df,
        formulas_cfg=formulas_cfg,
        prompt_cfg=prompt_cfg,
        use_llm=use_llm,
        hf_pipe=hf_pipe,
        extra_metadata=extra_metadata,
    )

    # For compatibility with earlier code return signature
    return df, metrics_df, details