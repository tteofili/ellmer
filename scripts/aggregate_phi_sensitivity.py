"""Build a flat CSV table from ``*_results.json`` files produced by ``phi_sensitivity.py``."""
from __future__ import annotations

import argparse
import os

from ellmer.phi_sensitivity import aggregate_results_directory


def main():
    parser = argparse.ArgumentParser(description="Aggregate phi sensitivity JSON results into one CSV table.")
    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing *_results.json (e.g. .../phi_sensitivity/run/books/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: results_dir/phi_sensitivity_table.csv)",
    )
    args = parser.parse_args()
    out = args.output or os.path.join(args.results_dir, "phi_sensitivity_table.csv")
    df = aggregate_results_directory(args.results_dir)
    if df.empty:
        raise SystemExit(f"No *_results.json found under {args.results_dir}")
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
