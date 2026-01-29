from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import pandas as pd

from .yaml_loader import load_vetro_yaml, load_prompts_yaml
from .metrics_eval import build_context, compute_metrics
from .llm import get_hf_pipe

def run_quality_assessment(
    df: pd.DataFrame,
    formulas_yaml_path: str,
    prompts_yaml_path: str,
    dataset_description: str = "",
    use_llm: bool = False,
    hf_model_name: str = "google/flan-t5-base",
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, Any]]:
    vetro_dict, _ = load_vetro_yaml(formulas_yaml_path)
    prompt_defs, _ = load_prompts_yaml(prompts_yaml_path)

    # Collect all symbols referenced by YAML inputs
    input_symbols = []
    for dim, metrics in vetro_dict.items():
        for mname, metric in metrics.items():
            for inp in metric.get("inputs", []):
                if isinstance(inp, dict):
                    # {'some_name': 'symbol'}
                    for _, sym in inp.items():
                        input_symbols.append(sym)
                elif isinstance(inp, str):
                    input_symbols.append(inp)
    input_symbols = sorted(set(input_symbols))

    context = build_context(df, description=dataset_description)

    hf_pipe = None
    if use_llm:
        hf_pipe = get_hf_pipe(hf_model_name)

    results, env, debug = compute_metrics(
        df=df,
        vetro_dict=vetro_dict,
        context=context,
        input_symbols=input_symbols,
        prompt_defs=prompt_defs,
        hf_pipe=hf_pipe,
    )

    rows = []
    for k, v in results.items():
        dim, metric = k.split(".", 1)
        rows.append({"dimension": dim, "metric": metric, "value": v})
    metrics_df = pd.DataFrame(rows).sort_values(["dimension", "metric"]).reset_index(drop=True)

    return results, metrics_df, {"env": env, "debug": debug, "input_symbols": input_symbols}
