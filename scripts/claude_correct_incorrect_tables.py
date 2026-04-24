#!/usr/bin/env python3
"""
Build per-dataset, per-granularity tables of Mean and std dev of
(correct - incorrect) for Faithfulness and Validity, for Claude / Anthropic runs.

Input: a long-form CSV from ``experiment_eval_metrics_by_correctness.py`` (or any table
with prediction_split, faithfulness_auc, validity, and the usual stratifier columns).
Rows must include ``correct`` and ``incorrect`` for each stratum to pair.

Output: Markdown (and optional CSV) with one section per (dataset, granularity) and
two sub-tables (Faithfulness, Validity), rows = standard explainers in fixed order.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ellmer.claude_table_report import aggregate_by_explainer, write_markdown
from ellmer.paired_metrics_diff import paired_difference_long


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Claude/Anthropic correct−incorrect faithfulness and validity tables per dataset and granularity"
    )
    ap.add_argument(
        "input_csv",
        type=Path,
        help="Long-form metrics CSV (must include correct/incorrect rows, e.g. from experiment_eval_metrics_by_correctness)",
    )
    ap.add_argument(
        "--model-name-pattern",
        type=str,
        default="claude|anthropic",
        help="Regex (case-insensitive) applied to model_name; default matches Claude/Anthropic Bedrock style ids",
    )
    ap.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="If set, also require this model_type (e.g. bedrock)",
    )
    ap.add_argument("--output-md", type=Path, default=None, help="Markdown path (default: <input>_claude_tables.md next to input)")
    ap.add_argument(
        "--output-agg-csv",
        type=Path,
        default=None,
        help="Optional: write aggregated (dataset, granularity, explainer, metric) mean, std, n",
    )
    ap.add_argument(
        "--include-unfiltered-paired-csv",
        type=Path,
        default=None,
        help="Optional: write the long-form paired diffs (after model filter) to CSV for debugging",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if "model_name" not in df.columns:
        raise SystemExit("input_csv must have model_name")
    m = df["model_name"].astype(str).str.contains(
        args.model_name_pattern, case=False, na=False, regex=True
    )
    df = df[m]
    if args.model_type and "model_type" in df.columns:
        df = df[df["model_type"].astype(str) == args.model_type]
    if df.empty:
        raise SystemExit("No rows left after model filter. Adjust --model-name-pattern or --model-type")

    pl = paired_difference_long(df)
    if pl.empty:
        raise SystemExit("No paired correct/minus incorrect rows. Ensure input has matching correct/incorrect strata")

    if args.include_unfiltered_paired_csv:
        pl.to_csv(args.include_unfiltered_paired_csv, index=False)

    agg = aggregate_by_explainer(pl)
    if args.output_agg_csv:
        args.output_agg_csv.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(args.output_agg_csv, index=False)

    out_md = args.output_md or (args.input_csv.parent / f"{args.input_csv.stem}_claude_tables.md")
    out_md = out_md.resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8") as f:
        model_note = f"model_name ~ /{args.model_name_pattern}/"
        if args.model_type:
            model_note += f", model_type={args.model_type!r}"
        write_markdown(agg, model_note, f)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
