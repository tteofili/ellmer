"""
Aggregate multi-run evaluation results and compute significance tests.

Reads run_*/eval.csv or eval_all_runs.csv from a session directory, computes
mean, std, and 95% CI per (dataset, model, metric), and pairwise Wilcoxon
signed-rank p-values for method comparisons per (dataset, metric).
"""

import argparse
import os
import glob

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Metric columns to aggregate (numeric); exclude identifiers and run_id
METRIC_COLUMNS = [
    "total_time",
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


def to_numeric_series(s):
    return pd.to_numeric(s, errors="coerce")


def compute_summary(df, metric_columns):
    """Per (dataset, model) and per metric: mean, std, 95% CI (normal approximation)."""
    rows = []
    for (dataset, model), grp in df.groupby(["dataset", "model"]):
        n = len(grp["run_id"].unique())
        for col in metric_columns:
            if col not in grp.columns:
                continue
            vals = to_numeric_series(grp[col]).dropna()
            if len(vals) == 0:
                continue
            mean = vals.mean()
            std = vals.std()
            if std != std or std == 0:
                se = 0.0
            else:
                se = std / np.sqrt(len(vals))
            ci_half = 1.96 * se
            rows.append({
                "dataset": dataset,
                "model": model,
                "metric": col,
                "mean": mean,
                "std": std,
                "ci_lower": mean - ci_half,
                "ci_upper": mean + ci_half,
                "n_runs": len(vals),
            })
    return pd.DataFrame(rows)


def compute_significance(df, metric_columns):
    """
    For each (dataset, metric), compare every pair of models with Wilcoxon
    signed-rank test (paired across runs). Returns dataset, metric, method_a,
    method_b, p_value.
    """
    rows = []
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]
        for metric in metric_columns:
            if metric not in sub.columns:
                continue
            wide = sub.pivot_table(
                index="run_id",
                columns="model",
                values=metric,
                aggfunc="first",
            )
            wide = wide.apply(to_numeric_series)
            models = [c for c in wide.columns if wide[c].notna().any()]
            if len(models) < 2:
                continue
            for i, method_a in enumerate(models):
                for method_b in models[i + 1 :]:
                    a_vals = wide[method_a].dropna()
                    b_vals = wide[method_b].dropna()
                    common_idx = a_vals.index.intersection(b_vals.index)
                    if len(common_idx) < 3:
                        continue
                    a = a_vals.loc[common_idx].values
                    b = b_vals.loc[common_idx].values
                    try:
                        _, p_value = wilcoxon(a, b, alternative="two-sided")
                    except Exception:
                        p_value = np.nan
                    rows.append({
                        "dataset": dataset,
                        "metric": metric,
                        "method_a": method_a,
                        "method_b": method_b,
                        "p_value": p_value,
                    })
    return pd.DataFrame(rows)


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
    args = parser.parse_args()

    df = load_runs(session_dir=args.session_dir, eval_all_runs_path=args.eval_all_runs)
    metric_cols = [c for c in METRIC_COLUMNS if c in df.columns]
    if not metric_cols:
        metric_cols = [c for c in df.columns if c not in ("dataset", "model", "run_id")]
        metric_cols = [c for c in metric_cols if df[c].dtype in (np.float64, np.int64) or pd.api.types.is_numeric_dtype(df[c])]

    summary_df = compute_summary(df, metric_cols)
    sig_df = compute_significance(df, metric_cols)

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
