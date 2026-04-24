#!/usr/bin/env python3
"""
One CSV per Markdown table in claude_tables.md, from claude_tables_agg.csv (same numbers).

With ``--pooled-fill`` and a long correct/incorrect export, stratum-missing explainer cells are
filled using mean and std pooled over all (dataset, granularity) for that (explainer, metric).

Writes ``<out-dir>/{dataset}__{granularity}__{faithfulness|validity}.csv`` with
Explainer, mean, stdDev.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from ellmer.claude_table_report import export_per_table_csvs, export_per_table_csvs_pooled
from ellmer.paired_metrics_diff import paired_difference_long


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export per-table CSVs from claude_tables_agg.csv (matches claude_tables.md)"
    )
    ap.add_argument(
        "--agg-csv",
        type=Path,
        default=Path("out/claude/claude_tables_agg.csv"),
        help="Aggregate CSV from render_out_claude_eval_tables (default: out/claude/claude_tables_agg.csv)",
    )
    ap.add_argument(
        "--long-csv",
        type=Path,
        default=Path("out/claude/claude_by_correctness_long.csv"),
        help="Long export with correct/incorrect rows; required for --pooled-fill",
    )
    ap.add_argument(
        "--pooled-fill",
        action="store_true",
        help="Fill empty explainer cells with explainer+metric means/std over all paired strata",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out/claude/csv_tables"),
        help="Output directory (default: out/claude/csv_tables)",
    )
    args = ap.parse_args()
    p = args.agg_csv.resolve()
    if not p.is_file():
        raise SystemExit(f"Missing: {p} (run render_out_claude_eval_tables.py --write-agg-csv first)")

    agg = pd.read_csv(p)
    if args.pooled_fill:
        lp = args.long_csv.resolve()
        if not lp.is_file():
            raise SystemExit(f"Missing: {lp} (run render_out_claude_eval_tables.py --write-long-csv first)")
        long_df = pd.read_csv(lp)
        pl = paired_difference_long(long_df)
        if pl.empty:
            raise SystemExit("Paired diffs are empty; cannot fill")
        written = export_per_table_csvs_pooled(agg, pl, args.out_dir)
    else:
        written = export_per_table_csvs(agg, args.out_dir)
    for w in written:
        print(f"Wrote {w}")


if __name__ == "__main__":
    main()
