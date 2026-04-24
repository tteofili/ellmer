#!/usr/bin/env python3
"""
Build Kendall-tau and CF-similarity **sheet-style** tables (attribute vs token) for Claude.

- **KT**: ``avg_kt`` from :func:`attribute_token_alignment.compare` (Kendall τ between
  token- and attribute-level saliency mass rankings, per :func:`saliency_consistency`).
- **CF similarity**: ``avg_cf_sim`` (TF–IDF cosine of token vs attribute first CF text).

If ``out/claude/attribute_token_agreement_claude.csv`` exists, it is read; else ``compare()``
is run (same 5 explainers: ZS, CoT, ICL, CERTA, Ellmer_C).

Run:  PYTHONPATH=. python scripts/export_claude_kt_cf_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from attribute_token_alignment import compare

OUT = ROOT / "out" / "claude"
IN_CSV = OUT / "attribute_token_agreement_claude.csv"

SUITES = [
    {
        "token": OUT / "dm_token" / "run_0",
        "attr": OUT / "dm_attribute" / "run_3",
        "datasets": [
            ("abt_buy", "AB"),
            ("beers", "BR"),
            ("fodo_zaga", "FZ"),
            ("walmart_amazon", "WA"),
            ("amazon_google", "AG"),
        ],
    },
    {
        "token": OUT / "books_token" / "run_4",
        "attr": OUT / "books_attribute" / "run_4",
        "datasets": [("books", "FB")],
    },
    {
        "token": OUT / "carparts_token" / "run_4",
        "attr": OUT / "carparts_attribute" / "run_4",
        "datasets": [("carparts", "FCP")],
    },
]

EXPLAINERS_5 = [
    "zs_sample",
    "cot_sample",
    "fs_sample",
    "certa_sample",
    "hybrid_sample",
]

ORDER_COLS = ["ZS", "CoT", "ICL", "CERTA", "Ellmer_C"]
KEY_TO_COL = {
    "zs_sample": "ZS",
    "cot_sample": "CoT",
    "fs_sample": "ICL",
    "certa_sample": "CERTA",
    "hybrid_sample": "Ellmer_C",
}

DATASET_ORDER = [
    "AB",
    "BR",
    "FZ",
    "WA",
    "AG",
    "FB",
    "FCP",
    "Cameras",
    "Watches",
    "Faker",
]

MODEL = "claude"


def run_compare() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for suite in SUITES:
        for ds, short in suite["datasets"]:
            pair = (
                str((suite["token"] / ds).resolve()) + "/",
                str((suite["attr"] / ds).resolve()) + "/",
            )
            try:
                df = compare(
                    pair,
                    "claude_bedrock",
                    explainers=EXPLAINERS_5,
                    k_tokens=30,
                    k_attrs=3,
                    write_debug_csv=False,
                )
            except Exception:  # noqa: BLE001
                continue
            if df is None or df.empty:
                continue
            df = df.copy()
            df["dataset_label"] = short
            df["dataset"] = ds
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pivot_metric(long5: pd.DataFrame, col: str) -> pd.DataFrame:
    p = long5.pivot_table(
        index="dataset_label", columns="explainer", values=col, aggfunc="first"
    )
    p = p.rename(columns=KEY_TO_COL)
    for c in ORDER_COLS:
        if c not in p.columns:
            p[c] = np.nan
    return p[ORDER_COLS]


def wide_with_mean(p: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=DATASET_ORDER, columns=ORDER_COLS, dtype=float)
    for lab in p.index:
        if lab in out.index:
            out.loc[lab] = p.loc[lab]
    m = out.loc[DATASET_ORDER[:7]].mean(numeric_only=True, skipna=True)
    out.loc["Mean"] = m
    return out


def to_sheet(pretty_title: str, body: pd.DataFrame) -> pd.DataFrame:
    """
    6×N: col0 = row kind / dataset, cols 1–5 = explainer.
    4 header lines + 10 data lines + 1 mean line = 15 rows.
    """
    b = body.copy().round(6)
    data_part = b.drop(index=["Mean"], errors="ignore")
    mean_row = b.loc[["Mean"]] if "Mean" in b.index else pd.DataFrame()
    out_rows: list[list[str | float]] = [
        [pretty_title, MODEL, MODEL, MODEL, MODEL, MODEL],
        ["model", MODEL, MODEL, MODEL, MODEL, MODEL],
        [
            "",
            "Self-Explanations",
            "Self-Explanations",
            "Self-Explanations",
            "Post-hoc",
            "Post-hoc",
        ],
        ["", "ZS", "CoT", "ICL", "CERTA", "Ellmer_C"],
    ]
    for idx in data_part.index:
        row: list[str | float] = [str(idx)]
        for c in ORDER_COLS:
            v = data_part.loc[idx, c]
            row.append("" if pd.isna(v) else float(v))
        out_rows.append(row)
    if not mean_row.empty:
        row_m: list[str | float] = ["Mean"]
        for c in ORDER_COLS:
            v = mean_row.loc["Mean", c]
            row_m.append(float(v) if not pd.isna(v) else "")
        out_rows.append(row_m)
    return pd.DataFrame(
        out_rows, columns=["", "ZS", "CoT", "ICL", "CERTA", "Ellmer_C"]
    )


def main():
    if IN_CSV.is_file():
        long = pd.read_csv(IN_CSV)
        long5 = long[long["explainer"].isin(EXPLAINERS_5)].copy()
    else:
        long5 = run_compare()
        if long5.empty:
            print("No data: run attribute_token pair exports first or check paths.")
            return

    kt_w = pivot_metric(long5, "avg_kt")
    cf_w = pivot_metric(long5, "avg_cf_sim")

    kt_body = wide_with_mean(kt_w)
    cf_body = wide_with_mean(cf_w)

    t_kt = to_sheet("KT", kt_body)
    t_cf = to_sheet("CF similarity", cf_body)

    out_kt = OUT / "claude_kt_attr_token_saliency.csv"
    out_cf = OUT / "claude_cf_similarity_attr_token.csv"
    t_kt.to_csv(out_kt, index=False, header=False)
    t_cf.to_csv(out_cf, index=False, header=False)
    print(f"Wrote {out_kt}")
    print(f"Wrote {out_cf}")


if __name__ == "__main__":
    main()
