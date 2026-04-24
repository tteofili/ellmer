#!/usr/bin/env python3
"""
Bootstrap 95% CIs for the macro-mean faithfulness or validity (mean over benchmark
datasets) by resampling datasets with replacement. Uses only precomputed per-dataset
columns (paper-style CSVs under out/claude/ and optional claude_tables_agg).

No LLM or explanation regeneration.
"""

from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ellmer.stats_inference import bootstrap_mean_ci_1d


def _fmt(v: float, n_dec: int = 3) -> str:
    if v != v:  # nan
        return "nan"
    return f"{v:.{n_dec}f}"


def _line_simple(m: float, lo: float, hi: float) -> str:
    hw = (hi - lo) / 2.0
    return f"{_fmt(m)} ± {_fmt(hw)} (95% CI: [{_fmt(lo)}, {_fmt(hi)}])"


def from_paper_csv(path: Path, block_title: str) -> None:
    """Parse *_{attribute,token}.csv (Dataset, ZS, CoT, ICL, CERTA, ...)."""
    dfp = pd.read_csv(path, dtype=str, na_filter=True)
    col0 = dfp.columns[0]
    dfp = dfp[dfp[col0].notna() & (dfp[col0].str.strip() != "Mean")].copy()
    dfp.columns = [c.strip() for c in dfp.columns]
    for c in ("ZS", "CoT", "ICL", "CERTA", "Lemon", "Ellmer_C", "Ellmer_L"):
        if c in dfp.columns:
            dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
    dfp = dfp[dfp["ZS"].notna() & dfp["CERTA"].notna()].copy()
    n = len(dfp)
    if n == 0:
        print(f"(no complete ZS+CERTA rows in {path})")
        return
    names = dfp[dfp.columns[0]].astype(str).tolist()

    zs = dfp["ZS"].to_numpy()
    cta = dfp["CERTA"].to_numpy()
    m0, l0, h0, n0 = bootstrap_mean_ci_1d(
        zs, n_bootstrap=10_000, confidence=0.95, rng=np.random.default_rng(0)
    )
    m1, l1, h1, n1 = bootstrap_mean_ci_1d(
        cta, n_bootstrap=10_000, confidence=0.95, rng=np.random.default_rng(0)
    )

    def one_indented(label: str, m: float, lo: float, hi: float) -> str:
        hw = (hi - lo) / 2.0
        return f"  {label} {_fmt(m)} ± {_fmt(hw)} (95% CI: [{_fmt(lo)}, {_fmt(hi)}])"

    print()
    print(f"Source: {path.name}  (N datasets with ZS + CERTA: {n0})")
    print(f"Datasets: {', '.join(names)}")
    print()
    print(f"{block_title}:")
    if block_title == "Faithfulness":
        print(one_indented("Self (ZS)         ", m0, l0, h0))
        print(one_indented("Post-hoc (CERTA)  ", m1, l1, h1))
    else:
        print(f"Self: {_line_simple(m0, l0, h0)}")
        print(f"Post-hoc: {_line_simple(m1, l1, h1)}")
    if block_title == "Faithfulness" and dfp.get("CoT") is not None and dfp.get("ICL") is not None:
        if dfp["CoT"].notna().all() and dfp["ICL"].notna().all():
            sm3 = (dfp["ZS"] + dfp["CoT"] + dfp["ICL"]) / 3.0
            m2, l2, h2, _ = bootstrap_mean_ci_1d(
                sm3.to_numpy(), n_bootstrap=10_000, confidence=0.95, rng=np.random.default_rng(1)
            )
            print()
            print("  Optional — Self = mean(ZS, CoT, ICL) per dataset:")
            print(one_indented("Self (3-prompt)   ", m2, l2, h2))
    print()


def from_claude_tables_agg(path: Path, granularity: str, metric: str) -> None:
    df = pd.read_csv(path)
    df = df[(df["metric"] == metric) & (df["granularity"] == granularity)]
    pt = df.pivot_table(index="dataset", columns="explainer", values="mean", aggfunc="first")
    if "zs_sample" not in pt.columns or "certa_sample" not in pt.columns:
        print(f"Missing zs or certa in {path} for {granularity}")
        return
    t = pd.concat(
        [pt["zs_sample"].rename("self_zs"), pt["certa_sample"].rename("ph")],
        axis=1,
    ).dropna()
    n = len(t)
    if n == 0:
        return
    a, b = t["self_zs"].to_numpy(), t["ph"].to_numpy()
    m0, l0, h0, _ = bootstrap_mean_ci_1d(
        a, n_bootstrap=10_000, confidence=0.95, rng=np.random.default_rng(0)
    )
    m1, l1, h1, _ = bootstrap_mean_ci_1d(
        b, n_bootstrap=10_000, confidence=0.95, rng=np.random.default_rng(0)
    )
    print()
    print(
        f"Source: {path.name}  granularity={granularity}  (N datasets: {n}, out/claude aggregate)"
    )
    print(f"Datasets: {', '.join(t.index.astype(str))}")
    print()
    title = {
        "faithfulness_auc": "Faithfulness (claude out/claude, zs_sample vs certa_sample)",
        "validity": "Validity (claude out/claude, zs_sample vs certa_sample)",
    }.get(metric, f"{metric} (zs_sample vs certa_sample)")
    print(f"{title}:")
    if metric == "faithfulness_auc":
        def one2(label: str, m: float, lo: float, hi: float) -> str:
            hw = (hi - lo) / 2.0
            return f"  {label} {_fmt(m)} ± {_fmt(hw)} (95% CI: [{_fmt(lo)}, {_fmt(hi)}])"

        print(one2("Self (zs_sample)  ", m0, l0, h0))
        print(one2("Post-hoc (certa)  ", m1, l1, h1))
    else:
        print(f"Self: {_line_simple(m0, l0, h0)}")
        print(f"Post-hoc: {_line_simple(m1, l1, h1)}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bootstrap 95% CIs for mean faithfulness over benchmark datasets (no new LLM calls)."
    )
    ap.add_argument("--paper-dir", type=Path, default=Path("out/claude"))
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="If set, also write the full report to this file (e.g. paper_stats_out/...).",
    )
    args = ap.parse_args()
    pdir = args.paper_dir
    fa = pdir / "faithfulness_claude_attribute.csv"
    ft = pdir / "faithfulness_claude_token.csv"
    va = pdir / "validity_claude_attribute.csv"
    vt = pdir / "validity_claude_token.csv"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print("=" * 72)
        print("MACRO-MEAN + bootstrap over datasets (10_000 resamples, seed 0/1).")
        print(
            "Macro-mean = unweighted mean over the listed N datasets; CI resamples those datasets with replacement.\n"
        )
        if fa.is_file():
            from_paper_csv(fa, "Faithfulness")
        if ft.is_file():
            from_paper_csv(ft, "Faithfulness")
        if va.is_file():
            from_paper_csv(va, "Validity")
        if vt.is_file():
            from_paper_csv(vt, "Validity")
        agg = Path("out/claude/claude_tables_agg.csv")
        if agg.is_file():
            for gr in ("attribute", "token"):
                from_claude_tables_agg(agg, gr, "faithfulness_auc")
            for gr in ("attribute", "token"):
                from_claude_tables_agg(agg, gr, "validity")
        if (
            not fa.is_file()
            and not ft.is_file()
            and not va.is_file()
            and not vt.is_file()
            and not agg.is_file()
        ):
            print("No input files found. Expected at least one of:")
            print("  out/claude/faithfulness_claude_*.csv, validity_claude_*.csv")
            print("  out/claude/claude_tables_agg.csv")
    s = buf.getvalue()
    print(s, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(s, encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.__stdout__)


if __name__ == "__main__":
    main()
