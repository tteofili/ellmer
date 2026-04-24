#!/usr/bin/env python3
"""
Build per-dataset, per-granularity correct−incorrect tables from ``out/claude/**/eval.csv``.

Writes:
  - ``<out_claude_root>/claude_by_correctness_long.csv`` (optional, with --write-long-csv)
  - ``<out_claude_root>/claude_tables.md``
  - ``<out_claude_root>/claude_tables_agg.csv`` (optional, with --write-agg-csv)
  - ``<out_claude_root>/csv_tables/*.csv`` (optional, with --write-per-table-csv
    or --write-per-table-csv-pooled)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ellmer.claude_table_report import (
    aggregate_by_explainer,
    export_per_table_csvs,
    export_per_table_csvs_pooled,
    write_markdown,
)
from ellmer.out_claude_eval_ingest import load_out_claude_long
from ellmer.paired_metrics_diff import paired_difference_long


def main() -> None:
    ap = argparse.ArgumentParser(description="Render Markdown tables from out/claude eval.csv exports")
    ap.add_argument(
        "--out-claude-root",
        type=Path,
        default=Path("out/claude"),
        help="Directory containing per-experiment eval.csv files (default: out/claude)",
    )
    ap.add_argument(
        "--model-name",
        type=str,
        default="claude",
        help="model_name on ingested rows (for documentation only)",
    )
    ap.add_argument(
        "--write-long-csv",
        action="store_true",
        help="Also write claude_by_correctness_long.csv (correct/incorrect long rows)",
    )
    ap.add_argument(
        "--write-agg-csv",
        action="store_true",
        help="Also write claude_tables_agg.csv (mean/std per group)",
    )
    ap.add_argument(
        "--write-per-table-csv",
        action="store_true",
        help="Write one CSV per (dataset, granularity, metric) under csv_tables/ (uses agg)",
    )
    ap.add_argument(
        "--write-per-table-csv-pooled",
        action="store_true",
        help="Like --write-per-table-csv, but fill empty explainer cells with explainer+metric pool",
    )
    args = ap.parse_args()
    root = args.out_claude_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    long_df = load_out_claude_long(root, model_name=args.model_name)
    if long_df.empty:
        raise SystemExit(f"No correct/incorrect rows under {root}")

    if args.write_long_csv:
        p = root / "claude_by_correctness_long.csv"
        long_df.to_csv(p, index=False)
        print(f"Wrote {p}")

    pl = paired_difference_long(long_df)
    if pl.empty:
        raise SystemExit("No pairable correct/incorrect strata (finite faithfulness and validity)")

    agg = aggregate_by_explainer(pl)
    if args.write_agg_csv:
        p = root / "claude_tables_agg.csv"
        agg.to_csv(p, index=False)
        print(f"Wrote {p}")

    if args.write_per_table_csv and args.write_per_table_csv_pooled:
        raise SystemExit("Use only one of --write-per-table-csv and --write-per-table-csv-pooled")

    if args.write_per_table_csv:
        csv_dir = root / "csv_tables"
        for w in export_per_table_csvs(agg, csv_dir):
            print(f"Wrote {w}")

    if args.write_per_table_csv_pooled:
        csv_dir = root / "csv_tables"
        for w in export_per_table_csvs_pooled(agg, pl, csv_dir):
            print(f"Wrote {w}")

    md = root / "claude_tables.md"
    with md.open("w", encoding="utf-8") as f:
        write_markdown(
            agg,
            f"out/claude eval exports (model_name={args.model_name!r})",
            f,
        )
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
