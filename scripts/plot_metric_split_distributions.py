#!/usr/bin/env python3
"""
Distributions of faithfulness and validity for ``correct`` vs ``incorrect`` (per stratum: session,
``run_id``, explainer), with a reference line at the **mean of metrics on**
``prediction_split == all`` (same filter).

Data are **aggregate** per eval run from ``out/claude/**/eval.csv``, not per test instance; see
``out_claude`` ingest.

Example:

  venv/bin/python scripts/plot_metric_split_distributions.py \\
    --out-claude-root out/claude --dataset books --granularity token \\
    --out-dir out/claude/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ellmer.out_claude_eval_ingest import load_out_claude_long


def _finite(a) -> np.ndarray:
    x = np.asarray(a, dtype=float)
    return x[np.isfinite(x)]


def _plot_metric_axis(ax, ylabel: str, correct: np.ndarray, incorrect: np.ndarray, all_vals: np.ndarray) -> None:
    c, ic, al = _finite(correct), _finite(incorrect), _finite(all_vals)
    bins = 15
    parts = [p for p in (c, ic) if p.size > 0]
    if parts:
        lo = min(float(p.min()) for p in parts)
        hi = max(float(p.max()) for p in parts)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = 0.0, 1.0
        edges: np.ndarray | int = np.linspace(lo, hi, bins + 1)
    else:
        edges = 15
    for x, name, color in ((c, "correct", "C0"), (ic, "incorrect", "C1")):
        if x.size == 0:
            continue
        ax.hist(
            x,
            bins=edges,
            density=True,
            alpha=0.5,
            color=color,
            label=f"{name} (n={x.size})",
        )
    if al.size > 0:
        m = float(np.mean(al))
        ax.axvline(
            m,
            color="k",
            ls="--",
            lw=1.3,
            label=f"mean (all)={m:.4f} (n={al.size})",
        )
    ax.set_ylabel("density")
    ax.set_xlabel(ylabel)
    ax.legend(loc="best", fontsize=7)
    ax.set_title(ylabel)


def _plot_figure(
    sub,
    dataset: str,
    granularity: str,
    explainer: str):
    import matplotlib.pyplot as plt

    s = sub[(sub["explainer"] == explainer)].copy()
    c_f = s.loc[s["prediction_split"] == "correct", "faithfulness_auc"]
    i_f = s.loc[s["prediction_split"] == "incorrect", "faithfulness_auc"]
    a_f = s.loc[s["prediction_split"] == "all", "faithfulness_auc"]
    c_v = s.loc[s["prediction_split"] == "correct", "validity"]
    i_v = s.loc[s["prediction_split"] == "incorrect", "validity"]
    a_v = s.loc[s["prediction_split"] == "all", "validity"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    _plot_metric_axis(axes[0], "faithfulness (AUC)", c_f, i_f, a_f)
    _plot_metric_axis(axes[1], "validity", c_v, i_v, a_v)
    fig.suptitle(
        f"{dataset} / {granularity} — {explainer}\nper-stratum values; dashed line = mean of 'all' runs",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot faithfulness & validity for correct vs incorrect with mean(all) reference"
    )
    ap.add_argument(
        "--out-claude-root",
        type=Path,
        default=Path("out/claude"),
        help="Tree containing **/eval.csv (default: out/claude)",
    )
    ap.add_argument("--dataset", required=True, help="Dataset name (e.g. books, carparts)")
    ap.add_argument(
        "--granularity",
        required=True,
        choices=("attribute", "token", "unknown"),
        help="Granularity (must match experiment folder naming logic)",
    )
    ap.add_argument(
        "--explainer",
        default=None,
        help="Comma-separated explainer keys; default: all in filtered data",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out/claude/figures"),
        help="Output directory for PNGs (default: out/claude/figures)",
    )
    ap.add_argument(
        "--format",
        default="png",
        choices=("png", "svg", "pdf"),
        help="Figure format (default: png)",
    )
    args = ap.parse_args()
    from matplotlib import pyplot as plt

    root = args.out_claude_root.resolve()
    if not root.is_dir():
        print("Not a directory:", root, file=sys.stderr)
        return 1

    long_df = load_out_claude_long(
        root,
        prediction_splits=("all", "correct", "incorrect"),
    )
    if long_df.empty:
        print("No eval rows; check", root, file=sys.stderr)
        return 1

    mask = (long_df["dataset"] == args.dataset) & (long_df["granularity"] == args.granularity)
    sub = long_df.loc[mask]
    if sub.empty:
        print(
            f"No rows for dataset={args.dataset!r} granularity={args.granularity!r}",
            file=sys.stderr,
        )
        return 1

    if args.explainer:
        expls = [x.strip() for x in args.explainer.split(",") if x.strip()]
    else:
        expls = sorted(sub["explainer"].dropna().unique().tolist())
    if not expls:
        print("No explainers after filter.", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ex_set = set(sub["explainer"].dropna().astype(str).unique())
    for e in expls:
        if e not in ex_set:
            print(f"Skip missing explainer: {e}", file=sys.stderr)
            continue
        fig = _plot_figure(sub, args.dataset, args.granularity, e)
        safe = e.replace("/", "_")
        out = args.out_dir / f"split_metrics_{args.dataset}_{args.granularity}_{safe}.{args.format}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
