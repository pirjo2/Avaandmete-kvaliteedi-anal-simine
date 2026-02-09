from __future__ import annotations

from typing import Any, Dict, Tuple
import pandas as pd

from core.yaml_loader import load_vetro_yaml, load_prompts_yaml
from core.llm import get_hf_runner
from core.metrics_eval import compute_metrics


def run_quality_assessment(
    df: pd.DataFrame,
    formulas_yaml_path: str,
    prompts_yaml_path: str,
    use_llm: bool,
    hf_model_name: str,
    dataset_description: str = "",
    file_ext: str = "",
    file_name: str = "",
    confidence_weighting: bool = True,
    **_ignored_kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    formulas_cfg = load_vetro_yaml(formulas_yaml_path)
    prompt_cfg = load_prompts_yaml(prompts_yaml_path)

    hf_runner = None
    if use_llm:
        hf_runner = get_hf_runner(hf_model_name)

    metrics_df, details = compute_metrics(
        df=df,
        formulas_cfg=formulas_cfg,
        prompt_cfg=prompt_cfg,
        use_llm=use_llm,
        hf_runner=hf_runner,
        dataset_description=dataset_description,
        file_ext=file_ext,
        file_name=file_name,
        confidence_weighting=confidence_weighting,
    )

    return df, metrics_df, details
