"""
Aggregate multi-run evaluation results and compute significance tests.

Reads run_*/eval.csv or eval_all_runs.csv from a session directory, computes
mean, std, and confidence intervals for the mean (Student's t) per
(dataset, model, metric), and pairwise Wilcoxon p-values plus CIs for the
paired mean difference (method_a minus method_b) per (dataset, metric).
Default confidence level is 95% (override with --confidence).
"""

import argparse
import os
import glob
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.join(_ROOT, "..") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, ".."))

from ellmer.stats_inference import compute_run_pairwise_wilcoxon, compute_run_summary  # noqa: E402

# Metric columns to aggregate (numeric); exclude identifiers and run_id
# total_local_time / avg_latency_local exclude LLM call time for stable timing across runs
METRIC_COLUMNS = [
    "total_time",
    "total_llm_time",
    "total_local_time",
    "avg_latency_llm",
    "avg_latency_local",
    "tokens",
    "predictions",
    "faithfulness",
    "validity",
    "proximity",
    "sparsity",
    "diversity",
]


def load_runs(session_dir=None, eval_all_runs_path=None):
    """
    Load evaluation data from either a session dir (run_*/eval.csv or eval_all_runs.csv)
    or a direct path to eval_all_runs.csv.
    Returns a DataFrame with run_id, dataset, model, and all metric columns.
    """
    if eval_all_runs_path and os.path.isfile(eval_all_runs_path):
        df = pd.read_csv(eval_all_runs_path, index_col=0)
        if "run_id" not in df.columns:
            df["run_id"] = 0
        return df
    if not session_dir or not os.path.isdir(session_dir):
        raise FileNotFoundError("Provide a valid --session_dir or --eval_all_runs path.")

    all_runs_path = os.path.join(session_dir, "eval_all_runs.csv")
    if os.path.isfile(all_runs_path):
        df = pd.read_csv(all_runs_path, index_col=0)
        return df

    run_dirs = sorted(glob.glob(os.path.join(session_dir, "run_*")))
    if not run_dirs:
        raise FileNotFoundError(
            f"No run_* subdirs or eval_all_runs.csv found in {session_dir}"
        )
    frames = []
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        run_id = int(run_name.split("_")[1])
        csv_path = os.path.join(run_dir, "eval.csv")
        if not os.path.isfile(csv_path):
            continue
        run_df = pd.read_csv(csv_path, index_col=0)
        run_df["run_id"] = run_id
        frames.append(run_df)
    if not frames:
        raise FileNotFoundError(f"No eval.csv found under run_* in {session_dir}")
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-run eval results and compute significance tests."
    )
    parser.add_argument(
        "--session_dir",
        type=str,
        default=None,
        help="Session directory containing run_0, run_1, ... or eval_all_runs.csv",
    )
    parser.add_argument(
        "--eval_all_runs",
        type=str,
        default=None,
        help="Path to eval_all_runs.csv (alternative to session_dir)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write summary.csv and significance.csv (default: session_dir or cwd)",
    )
    parser.add_argument(
        "--bonferroni",
        action="store_true",
        help="Apply Bonferroni correction to p-values in significance.csv",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for all CIs (summary mean CIs and paired diff CIs). Default: 0.95",
    )
    args = parser.parse_args()

    confidence = float(args.confidence)
    if not (0.0 < confidence < 1.0):
        raise ValueError("--confidence must be in (0, 1)")

    df = load_runs(session_dir=args.session_dir, eval_all_runs_path=args.eval_all_runs)
    metric_cols = [c for c in METRIC_COLUMNS if c in df.columns]
    if not metric_cols:
        metric_cols = [c for c in df.columns if c not in ("dataset", "model", "run_id")]
        metric_cols = [
            c
            for c in metric_cols
            if df[c].dtype in (np.float64, np.int64) or pd.api.types.is_numeric_dtype(df[c])
        ]

    summary_df = compute_run_summary(df, metric_cols, confidence=confidence)
    sig_df = compute_run_pairwise_wilcoxon(df, metric_cols, confidence=confidence)

    if args.bonferroni and len(sig_df) > 0:
        n_tests = len(sig_df)
        sig_df = sig_df.copy()
        sig_df["p_value_corrected"] = (sig_df["p_value"] * n_tests).clip(upper=1.0)

    out_dir = args.output_dir or args.session_dir or "."
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.csv")
    sig_path = os.path.join(out_dir, "significance.csv")
    summary_df.to_csv(summary_path, index=False)
    sig_df.to_csv(sig_path, index=False)
    print(f"Wrote {summary_path}")
    print(f"Wrote {sig_path}")


if __name__ == "__main__":
    main()
