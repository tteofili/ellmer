#!/usr/bin/env python3
"""
Walk experiments/ and aggregate faithfulness, counterfactual metrics, and
inter-explainer concordance (alignment) metrics.

Outputs (default prefix ``metrics_aggregate`` in --output-dir):

- ``{prefix}_metrics.csv`` — long-form rows with grouping keys and ``prediction_split``
  (``all``, ``correct``, ``incorrect``). Fast path reads JSON: top-level ``metrics`` yields
  ``all`` only; if ``metrics_by_prediction_split`` is present (e.g. from eval), three splits
  without recomputation. Otherwise ``correct`` / ``incorrect`` require ``--recompute`` + ``--base-dir``.

- ``{prefix}_alignment.csv`` — concordance CSV aggregates (mean kt, cos_sim, pred agreement).

**Concordance filtering** (``--concordance-filter``):

- ``all``: use every instance row in the concordance CSV.
- ``both_correct``: keep rows where both explainers match the label
  (``int(pred1)==int(label)`` and ``int(pred2)==int(label)``, with label as 0/1).
- ``either_incorrect``: complement of ``both_correct`` among rows with valid preds.

Recomputation calls the same explainer stack as ``scripts/eval.py`` (LLM / API access required).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Repo root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ellmer.experiment_concordance_path import parse_concordance_path
from ellmer.experiment_metrics_recompute import (
    recompute_metrics_for_file,
    safe_load_results_json,
)
from ellmer.experiment_paths import parse_results_json_path
from ellmer.metrics_json_rows import (
    pred_int as _pred_int,
    fast_rows_from_payload as _fast_rows_from_payload,
)


def _filter_concordance_df(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if df.empty or mode == "all":
        return df
    if "pred1" not in df.columns or "pred2" not in df.columns or "label" not in df.columns:
        return df
    p1 = df["pred1"].map(_pred_int)
    p2 = df["pred2"].map(_pred_int)
    lb = df["label"].map(lambda x: _pred_int(x) if not isinstance(x, (int, float)) else int(x))
    both = (p1 == lb) & (p2 == lb)
    if mode == "both_correct":
        return df.loc[both].copy()
    if mode == "either_incorrect":
        return df.loc[~both].copy()
    return df


def aggregate_concordance_csvs(
    experiments_root: Path,
    concordance_filter: str,
) -> pd.DataFrame:
    rows = []
    pattern = str(experiments_root / "**" / "concordance" / "**" / "*.csv")
    import glob

    for csv_path in glob.glob(pattern, recursive=True):
        meta = parse_concordance_path(csv_path)
        if meta is None:
            continue
        try:
            df = pd.read_csv(csv_path, index_col=0)
        except Exception:
            continue
        df_f = _filter_concordance_df(df, concordance_filter)
        n = len(df_f)
        row: Dict[str, Any] = {
            **{k: v for k, v in meta.items() if k != "csv_path"},
            "concordance_filter": concordance_filter,
            "n_instances": n,
        }
        if "kt" in df_f.columns:
            row["mean_kt"] = float(df_f["kt"].mean()) if n else np.nan
        if "cos_sim" in df_f.columns:
            row["mean_cos_sim"] = float(df_f["cos_sim"].mean()) if n else np.nan
        if "agree" in df_f.columns:
            row["pred_pair_agreement_rate"] = float(df_f["agree"].mean()) if n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics from experiments/.")
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Dataset root (required with --recompute or --prediction-splits)",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Emit prediction_split rows all, correct, and incorrect by recomputing faithfulness and CF "
        "metrics from raw result data using the same explainers as eval (requires --base-dir; LLM/API).",
    )
    parser.add_argument(
        "--prediction-splits",
        action="store_true",
        dest="prediction_splits",
        help="Alias for --recompute: same behavior (emit rows for all, correct, incorrect).",
    )
    parser.add_argument("--model-type", type=str, default=None, help="Override model type for recompute")
    parser.add_argument("--deployment-name", type=str, default="gpt-35-turbo")
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--num-triangles", type=int, default=10)
    parser.add_argument("--samples", type=int, default=-1)
    parser.add_argument("--tag", type=str, default="sample", help="Explainer tag suffix (zs_<tag>)")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--output-prefix", type=str, default="metrics_aggregate")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset name")
    parser.add_argument("--granularity", type=str, default=None, help="Filter attribute|token")
    parser.add_argument(
        "--concordance-filter",
        choices=["all", "both_correct", "either_incorrect", "each"],
        default="each",
        help="Filter concordance rows: 'each' writes all three modes (all, both_correct, either_incorrect); "
        "otherwise a single mode.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--verbose-json",
        action="store_true",
        help="With skipped JSON files, print path for each (default: one summary line only)",
    )
    args = parser.parse_args()

    do_recompute = bool(args.recompute or args.prediction_splits)
    if do_recompute and not args.base_dir:
        parser.error("--recompute / --prediction-splits requires --base-dir")
    if do_recompute and args.workers != 1:
        print("Note: using --workers 1 for --recompute to avoid overlapping LLM calls", file=sys.stderr)
        args.workers = 1

    root: Path = args.experiments_root
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    results_paths = sorted(root.rglob("*_results.json"))
    metric_rows: List[Dict[str, Any]] = []
    skipped_json_paths: List[str] = []

    def process_one(p: Path) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Returns (rows, skipped_path_if_invalid_json)."""
        path_str = str(p)
        info = parse_results_json_path(path_str)
        if info is None:
            return [], None
        if args.dataset and info.dataset != args.dataset:
            return [], None
        if args.granularity and info.granularity != args.granularity:
            return [], None
        if do_recompute:
            if not args.base_dir:
                print("--recompute requires --base-dir", file=sys.stderr)
                return [], None
            mt = args.model_type or info.model_type
            lemon_kwargs = {
                "lem_num_samples": None,
                "cf_max_evals": 800,
                "max_iterations": 10,
                "cf_method": "greedy",
                "lime_cap_samples": True,
            }
            rows = recompute_metrics_for_file(
                path_str,
                info,
                args.base_dir,
                args.samples,
                mt,
                args.deployment_name,
                args.temperature,
                args.num_triangles,
                args.tag,
                lemon_kwargs,
                source="recompute",
            )
            return rows, None
        payload = safe_load_results_json(p)
        if payload is None:
            return [], path_str
        if payload.get("metrics"):
            return _fast_rows_from_payload(payload, info), None
        return [], None

    if args.workers <= 1:
        for p in results_paths:
            rows, skip = process_one(p)
            metric_rows.extend(rows)
            if skip:
                skipped_json_paths.append(skip)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_one, p) for p in results_paths]
            for fut in as_completed(futs):
                rows, skip = fut.result()
                metric_rows.extend(rows)
                if skip:
                    skipped_json_paths.append(skip)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    df_m = pd.DataFrame(metric_rows)
    df_m.to_csv(args.output_dir / f"{prefix}_metrics.csv", index=False)

    if args.concordance_filter == "each":
        df_a = pd.concat(
            [aggregate_concordance_csvs(root, m) for m in ("all", "both_correct", "either_incorrect")],
            ignore_index=True,
        )
    else:
        df_a = aggregate_concordance_csvs(root, args.concordance_filter)
    df_a.to_csv(args.output_dir / f"{prefix}_alignment.csv", index=False)

    print(f"Wrote {args.output_dir / (prefix + '_metrics.csv')} ({len(df_m)} rows)")
    print(f"Wrote {args.output_dir / (prefix + '_alignment.csv')} ({len(df_a)} rows)")
    if not do_recompute:
        print(
            "Note: metrics rows use only JSON in each result file (fast path). "
            "For all/correct/incorrect without pre-stored splits, use --recompute --base-dir <datasets>, "
            "or produce result JSON with eval --split-metrics-by-correctness.",
            file=sys.stderr,
        )
    if skipped_json_paths:
        print(
            f"Skipped {len(skipped_json_paths)} result file(s) with invalid or unreadable JSON "
            f"(see warnings above).",
            file=sys.stderr,
        )
        if args.verbose_json:
            for sp in skipped_json_paths:
                print(f"  skipped: {sp}", file=sys.stderr)


if __name__ == "__main__":
    main()
