#!/usr/bin/env python3
"""
Plot distribution of (metric on correct) − (metric on incorrect) from the long
correctness table produced for the out/claude pipeline.

In this repository, the same per-split aggregates are not exported for
Azure OpenAI (ChatGPT) or HuggingFace Llama eval runs; their eval.csv files
typically omit prediction_split. Point this script at a future
`*_by_correctness_long.csv` that includes those model families to extend
the same plots.

Example:
  uv run python scripts/plot_pred_split_diff_distributions.py \\
    --dataset carparts --granularity attribute --out-dir paper_stats_out/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _pivot_diffs(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (keys... explainer) with d_faith, d_valid."""
    sub = df.copy()
    sub = sub[sub["prediction_split"].isin(["correct", "incorrect"])]
    idx = [
        c
        for c in [
            "model_name",
            "model_type",
            "dataset",
            "granularity",
            "session",
            "run_id",
            "explainer",
        ]
        if c in sub.columns
    ]
    p = sub.pivot_table(
        index=idx,
        columns="prediction_split",
        values=["faithfulness_auc", "validity"],
        aggfunc="first",
    )
    p.columns = ["_".join(tpl).strip("_") for tpl in p.columns.to_list()]
    out = p.reset_index()
    if (
        "faithfulness_auc_correct" not in out.columns
        or "faithfulness_auc_incorrect" not in out.columns
    ):
        return pd.DataFrame()
    for col in (
        "faithfulness_auc_correct",
        "faithfulness_auc_incorrect",
        "validity_correct",
        "validity_incorrect",
    ):
        if col not in out.columns:
            out[col] = np.nan
    out["faith_diff"] = pd.to_numeric(
        out["faithfulness_auc_correct"], errors="coerce"
    ) - pd.to_numeric(out["faithfulness_auc_incorrect"], errors="coerce")
    if "validity_correct" in out and "validity_incorrect" in out:
        out["valid_diff"] = pd.to_numeric(out["validity_correct"], errors="coerce") - pd.to_numeric(
            out["validity_incorrect"], errors="coerce"
        )
    else:
        out["valid_diff"] = np.nan
    return out


def _plot(
    dplot: pd.DataFrame,
    out_dir: Path,
    dataset: str,
    explainer_order: list[str] | None,
) -> list[Path]:
    import matplotlib.pyplot as plt

    written: list[Path] = []

    out_dir.mkdir(parents=True, exist_ok=True)
    order = explainer_order or sorted(dplot["explainer"].dropna().unique().tolist())
    dplot = dplot[dplot["explainer"].isin(order)].copy()

    def violin_simple(ax, y, ylab):
        positions, parts_data, tick_labels = [], [], []
        for j, e in enumerate(order, start=1):
            ys = dplot.loc[dplot["explainer"] == e, y].dropna().to_numpy()
            if len(ys) == 0:
                continue
            positions.append(j)
            parts_data.append(ys.astype(float))
            tick_labels.append(e)
        if not parts_data:
            ax.text(0.5, 0.5, f"no data for {ylab}", ha="center", transform=ax.transAxes)
            return
        v = ax.violinplot(
            parts_data, positions=positions, showmeans=True, showmedians=True, widths=0.7
        )
        for b in v["bodies"]:
            b.set_alpha(0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(ylab)
        ax.axhline(0.0, color="0.3", ls="--", lw=0.8)
        n = dplot[y].notna().sum()
        ax.set_title(f"{ylab} (N={n})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    violin_simple(axes[0], "faith_diff", "Δ faithfulness (correct − incorrect)")
    violin_simple(axes[1], "valid_diff", "Δ validity (correct − incorrect)")
    title = f"{dataset} — out_claude pipeline (splits in `claude_by_correctness_long` only)"
    fig.suptitle(title, fontsize=10, y=1.02)
    p1 = out_dir / f"predsplit_diff_violin_{dataset}.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(p1)

    # --- (1b) ECDF: zs / cot / certa faithfulness diffs
    kset = [e for e in ("zs_sample", "cot_sample", "certa_sample") if e in dplot["explainer"].values]
    if kset and dplot["faith_diff"].notna().any():
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        for e in kset:
            s = dplot.loc[dplot["explainer"] == e, "faith_diff"].dropna().sort_values()
            if len(s) < 1:
                continue
            y = np.linspace(0, 1, len(s), endpoint=True)
            ax.plot(s, y, label=f"{e} (n={len(s)})", lw=1.8)
        ax.set_xlabel("Δ faithfulness (correct − incorrect)")
        ax.set_ylabel("ECDF")
        ax.axvline(0, color="0.4", ls="--", lw=0.8)
        ax.legend(loc="best", fontsize=7)
        ax.set_title(f"{dataset} — ECDF of faithfulness diff (selected explainers)")
        fig.tight_layout()
        p_ecdf = out_dir / f"predsplit_faithdiff_ecdf_{dataset}.png"
        fig.savefig(p_ecdf, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(p_ecdf)

    # --- (2) ZS explainer: histograms of diffs
    z = dplot.loc[dplot["explainer"] == "zs_sample"]
    if len(z) and z["faith_diff"].notna().any():
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        for ax, y, t in zip(axes, ("faith_diff", "valid_diff"), ("faithfulness", "validity")):
            s = z[y].dropna()
            if len(s) > 0:
                ax.hist(
                    s,
                    bins=max(3, min(20, max(3, len(s) // 2))),
                    color="steelblue",
                    edgecolor="white",
                    alpha=0.85,
                )
                ax.axvline(0, color="crimson", ls="--", lw=1.0)
                ax.axvline(s.mean(), color="darkgreen", ls="-", lw=1.0, label=f"mean={s.mean():.3f}")
            ax.set_title(f"zs_sample: Δ {t} (n={len(s)})")
            ax.legend(loc="best", fontsize=7)
        fig.suptitle(f"{dataset} — explainer=zs_sample", fontsize=10)
        fig.tight_layout()
        p2 = out_dir / f"predsplit_diff_hist_zs_{dataset}.png"
        fig.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(p2)
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--long-csv",
        type=Path,
        default=Path("out/claude/claude_by_correctness_long.csv"),
    )
    ap.add_argument("--dataset", default="carparts")
    ap.add_argument("--granularity", default="attribute")
    ap.add_argument(
        "--explainers",
        default="certa_sample,cot_sample,fs_sample,hybrid_lemon_minun_sample,hybrid_sample,zs_sample",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("paper_stats_out/figures"))
    args = ap.parse_args()

    if not args.long_csv.is_file():
        print("Missing", args.long_csv, file=sys.stderr)
        return 1

    df = pd.read_csv(args.long_csv)
    mask = (df["dataset"] == args.dataset) & (df["granularity"] == args.granularity)
    expl = [x.strip() for x in args.explainers.split(",") if x.strip()]
    mask &= df["explainer"].isin(expl)
    sub = df.loc[mask]
    dplot = _pivot_diffs(sub)
    dplot = dplot[dplot["explainer"].isin(expl)]
    dplot = dplot.dropna(subset=["faith_diff", "valid_diff"], how="all")
    dplot = dplot[dplot["faith_diff"].notna() | dplot["valid_diff"].notna()]

    print(
        f"Rows with paired correct/incorrect (any metric): {len(dplot)}  "
        f"dataset={args.dataset}  granularity={args.granularity}"
    )
    if dplot.empty:
        print("Nothing to plot.", file=sys.stderr)
        return 1

    for e in expl:
        sl = dplot.loc[dplot["explainer"] == e]
        fm = sl["faith_diff"].mean()
        vm = sl["valid_diff"].mean()
        fms = f"{fm:.4f}" if len(sl) and not (fm != fm) else "—"
        vms = f"{vm:.4f}" if len(sl) and not (vm != vm) else "—"
        print(f"  {e:28}  n={len(sl)}  faith_Δ mean={fms}  valid_Δ mean={vms}")

    for p in _plot(dplot, args.out_dir, args.dataset, expl):
        print("Wrote", p)
    dplot.to_csv(args.out_dir / f"predsplit_diff_table_{args.dataset}.csv", index=False)
    print("Wrote", args.out_dir / f"predsplit_diff_table_{args.dataset}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
