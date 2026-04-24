#!/usr/bin/env python3
"""
Build the paper-style attribute–token agreement table (metrics × explainers) for Claude.

Uses attribute_token_alignment.compare_paper_table_metrics: CF coverage & precision, plus
top-1..5 saliency overlap (k_tokens = k_attrs = k for each k). Averages per dataset, then
across all datasets, then writes a transposed CSV (model row + explainer row + metric rows).

Run: PYTHONPATH=. python scripts/export_claude_attr_token_paper_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from attribute_token_alignment import compare_paper_table_metrics

OUT = ROOT / "out" / "claude"

# Same run pairing as other Claude exports
SUITES = [
    {
        "token": OUT / "dm_token" / "run_0",
        "attr": OUT / "dm_attribute" / "run_3",
        "datasets": [
            "abt_buy",
            "beers",
            "fodo_zaga",
            "walmart_amazon",
            "amazon_google",
        ],
    },
    {
        "token": OUT / "books_token" / "run_4",
        "attr": OUT / "books_attribute" / "run_4",
        "datasets": ["books"],
    },
    {
        "token": OUT / "carparts_token" / "run_4",
        "attr": OUT / "carparts_attribute" / "run_4",
        "datasets": ["carparts"],
    },
]

# Match reference sheet: five baselines, Hybrid = HybridCerta (no Lemon–Minun column)
PAPER_EXPLAINERS = [
    "zs_sample",
    "cot_sample",
    "fs_sample",
    "certa_sample",
    "hybrid_sample",
]

EXPLAINER_COL = {
    "zs_sample": "ZS",
    "cot_sample": "CoT",
    "fs_sample": "ICL",
    "certa_sample": "CERTA",
    "hybrid_sample": "Hybrid",
}

# Shown in the sheet header (like "chatgpt4" in the reference table)
MODEL = "claude"


def collect_per_dataset() -> pd.DataFrame:
    rows = []
    for suite in SUITES:
        for ds in suite["datasets"]:
            pair = (
                str((suite["token"] / ds).resolve()) + "/",
                str((suite["attr"] / ds).resolve()) + "/",
            )
            df = compare_paper_table_metrics(
                pair,
                model=MODEL,
                explainers=PAPER_EXPLAINERS,
                k_grid=5,
                write_debug_csv=False,
            )
            if df is None or df.empty:
                continue
            df = df.copy()
            df["dataset"] = ds
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def aggregate_by_explainer(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        "coverage",
        "precision",
        "top_1_overlap",
        "top_2_overlap",
        "top_3_overlap",
        "top_4_overlap",
        "top_5_overlap",
        "avg_top_k_overlap",
    ]
    g = df.groupby("explainer", as_index=False)[num_cols].mean()
    # Stable column order: ZS … Hybrid
    order = {e: i for i, e in enumerate(PAPER_EXPLAINERS)}
    g["_o"] = g["explainer"].map(order)
    g = g.sort_values("_o").drop(columns="_o")
    return g.reset_index(drop=True)


def to_transposed_clean(g: pd.DataFrame) -> pd.DataFrame:
    """
    Sheet layout: row1 model + model name in each explainer column; row2 explainer + ZS..Hybrid;
    then metric names in first column (coverage, precision, top-1 overlap, ...).
    """
    g = g.set_index("explainer")
    g = g.reindex([e for e in PAPER_EXPLAINERS if e in g.index])
    g.index = g.index.map(EXPLAINER_COL)
    metric_order = [
        "coverage",
        "precision",
        "top_1_overlap",
        "top_2_overlap",
        "top_3_overlap",
        "top_4_overlap",
        "top_5_overlap",
        "avg_top_k_overlap",
    ]
    g = g[[c for c in metric_order if c in g.columns]]
    metric_map = {
        "coverage": "coverage",
        "precision": "precision",
        "top_1_overlap": "top-1 overlap",
        "top_2_overlap": "top-2 overlap",
        "top_3_overlap": "top-3 overlap",
        "top_4_overlap": "top-4 overlap",
        "top_5_overlap": "top-5 overlap",
        "avg_top_k_overlap": "avg top-k overlap",
    }
    g = g.rename(columns=metric_map)
    # explainer x metric -> metric x explainer
    t = g.T
    t = t.reset_index().rename(columns={"index": "metric"})
    num = t.select_dtypes(include=["float", "int", "float64", "int64"])
    t[num.columns] = num.round(6)
    col_order = ["metric", *list(g.index)]
    expl_cols = list(g.index)
    row_model = ["model", MODEL, MODEL, MODEL, MODEL, MODEL]
    row_expl = ["explainer", *expl_cols]
    header = pd.DataFrame([row_model, row_expl], columns=col_order)
    full = pd.concat([header, t], ignore_index=True)
    return full


def build_table() -> pd.DataFrame:
    raw = collect_per_dataset()
    if raw.empty:
        return pd.DataFrame()
    agg = aggregate_by_explainer(raw)
    return to_transposed_clean(agg)


def main():
    full = build_table()
    if full.empty:
        print("No data; check out/claude paths and result JSONs.")
        return
    out_path = OUT / "attr_token_agreement_claude_paper_table.csv"
    full.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(full)} rows)")


if __name__ == "__main__":
    main()
