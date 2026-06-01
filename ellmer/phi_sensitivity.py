"""Helpers for phi sensitivity experiments (hybrid CERTA and Lemon/Minun)."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd


def phi_tag(phi: float) -> str:
    """Stable filename fragment for a phi value, e.g. 0.25 -> '0p25'."""
    return str(phi).replace(".", "p")


def _faithfulness_scalar(faithfulness: Any, model_key: str) -> Optional[float]:
    if faithfulness is None or faithfulness == "nan":
        return None
    if isinstance(faithfulness, dict):
        v = faithfulness.get(model_key)
        return float(v) if v is not None and isinstance(v, (int, float)) else None
    return None


def _cf_row(counterfactual_metrics: Any, model_key: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {"validity": None, "proximity": None, "sparsity": None}
    if not isinstance(counterfactual_metrics, dict):
        return out
    row = counterfactual_metrics.get(model_key)
    if not isinstance(row, dict):
        return out
    for k in out:
        if k in row:
            v = row[k]
            out[k] = float(v) if v is not None and v == v else None
    return out


def parse_phi_from_model_key(model_key: str) -> Optional[float]:
    """Parse phi from keys like ``hybrid_certa_phi0p25_mytag`` or ``hybrid_certa_phi1_mytag``."""
    m = re.search(r"_phi(0p[0-9]+|1)(?:_|$)", model_key)
    if not m:
        return None
    frag = m.group(1)
    if frag == "1":
        return 1.0
    return float(frag.replace("p", ".", 1))


def row_from_results_json(path: str) -> Optional[Dict[str, Any]]:
    """
    Build one flat dict of metrics from a single ``*_results.json`` written by ``run_explainer``.
    """
    model_key = os.path.basename(path).replace("_results.json", "")
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    metrics = payload.get("metrics") or {}
    faithfulness = metrics.get("faithfulness")
    cf_metrics = metrics.get("counterfactual_metrics")
    cf = _cf_row(cf_metrics, model_key)
    f_sc = _faithfulness_scalar(faithfulness, model_key)
    if model_key.startswith("hybrid_certa_phi"):
        method = "hybrid_certa"
    elif model_key.startswith("hybrid_lemon_minun_phi"):
        method = "hybrid_lemon_minun"
    else:
        method = "other"
    phi_val = parse_phi_from_model_key(model_key)
    return {
        "model_key": model_key,
        "method": method,
        "phi": phi_val,
        "avg_latency": payload.get("avg_latency"),
        "avg_latency_llm": payload.get("avg_latency_llm"),
        "avg_latency_local": payload.get("avg_latency_local"),
        "total_time": payload.get("total_time"),
        "total_local_time": payload.get("total_local_time"),
        "tokens": payload.get("tokens"),
        "faithfulness_auc": f_sc,
        "validity": cf["validity"],
        "proximity": cf["proximity"],
        "sparsity": cf["sparsity"],
        "results_path": path,
    }


def aggregate_results_directory(
    directory: str,
    pattern: str = "*_results.json",
) -> pd.DataFrame:
    """
    Scan ``directory`` (non-recursive) for result JSON files and return a comparison table.

    Typical layout: output from ``scripts/phi_sensitivity.py`` with one folder per dataset.
    """
    import fnmatch

    rows: List[Dict[str, Any]] = []
    if not os.path.isdir(directory):
        raise FileNotFoundError(directory)
    for name in sorted(os.listdir(directory)):
        if not fnmatch.fnmatch(name, pattern):
            continue
        if not name.endswith("_results.json"):
            continue
        path = os.path.join(directory, name)
        rows.append(row_from_results_json(path))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
