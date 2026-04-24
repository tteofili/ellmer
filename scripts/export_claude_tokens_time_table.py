#!/usr/bin/env python3
"""
Build **two** sheet-style tables: per-dataset **tokens** and **total_time** (seconds) for each
explainer, from ``eval.csv`` rows with ``prediction_split=all`` — one table for **token**
explanations and one for **attribute** explanations.

- Token: ``dm_token/run_0``, ``books_token/run_4``, ``carparts_token/run_4``
- Attribute: ``dm_attribute/run_3``, ``books_attribute/run_4``, ``carparts_attribute/run_4``

Outputs:

- ``out/claude/claude_tokens_and_time_token_granularity.csv``
- ``out/claude/claude_tokens_and_time_attribute_granularity.csv``

Run:  PYTHONPATH=. python scripts/export_claude_tokens_time_table.py
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "claude"

EVAL_TOKEN_FILES = [
    OUT / "dm_token" / "run_0" / "eval.csv",
    OUT / "books_token" / "run_4" / "eval.csv",
    OUT / "carparts_token" / "run_4" / "eval.csv",
]

EVAL_ATTRIBUTE_FILES = [
    OUT / "dm_attribute" / "run_3" / "eval.csv",
    OUT / "books_attribute" / "run_4" / "eval.csv",
    OUT / "carparts_attribute" / "run_4" / "eval.csv",
]

DS_TO_LABEL = {
    "abt_buy": "AB",
    "beers": "BR",
    "fodo_zaga": "FZ",
    "walmart_amazon": "WA",
    "amazon_google": "AG",
    "books": "FB",
    "carparts": "FCP",
}

EXPLAINERS = [
    "zs_sample",
    "cot_sample",
    "fs_sample",
    "certa_sample",
    "hybrid_sample",
]

HEAD = {
    "zs_sample": "ZS",
    "cot_sample": "CoT",
    "fs_sample": "FS",
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

MODEL = "Claude"


def load_eval_rows(eval_files: list[Path]):
    """(dataset_label, explainer) -> (tokens, total_time); later files override earlier on conflict."""
    merged: dict[tuple[str, str], tuple[float, float]] = {}
    for path in eval_files:
        if not path.is_file():
            continue
        with open(path, newline="") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                if r.get("prediction_split") != "all":
                    continue
                ds = r.get("dataset", "")
                ex = r.get("model", "")
                if ex not in EXPLAINERS:
                    continue
                lab = DS_TO_LABEL.get(ds)
                if not lab:
                    continue
                try:
                    tok = float(r.get("tokens") or 0)
                    tt = float(r.get("total_time") or 0)
                except ValueError:
                    continue
                merged[(lab, ex)] = (tok, tt)
    return merged


def build_table(merged: dict[tuple[str, str], tuple[float, float]]) -> list[list[str | int | float]]:
    """12 columns: label, 5 token, 5 time."""
    rows: list[list[str | int | float]] = []
    # header: model name across 10 data columns
    rows.append(
        ["", MODEL, MODEL, MODEL, MODEL, MODEL, MODEL, MODEL, MODEL, MODEL, MODEL]
    )
    rows.append(
        [
            "",
            "Tokens",
            "Tokens",
            "Tokens",
            "Tokens",
            "Tokens",
            "Running time",
            "Running time",
            "Running time",
            "Running time",
            "Running time",
        ]
    )
    rows.append(
        [
            "",
            *[HEAD[e] for e in EXPLAINERS],
            *[HEAD[e] for e in EXPLAINERS],
        ]
    )
    for lab in DATASET_ORDER:
        r = [lab]
        for ex in EXPLAINERS:
            v = merged.get((lab, ex))
            if v is None:
                r.append("")
            else:
                tok, tt = v
                r.append(int(round(tok)) if tok == tok else "")
        for ex in EXPLAINERS:
            v = merged.get((lab, ex))
            if v is None:
                r.append("")
            else:
                tok, tt = v
                # wall-clock seconds, round to nearest second (as in sheet examples)
                r.append(int(round(tt)) if tt == tt else "")
        rows.append(r)
    return rows


def main():
    outs = [
        (
            EVAL_TOKEN_FILES,
            OUT / "claude_tokens_and_time_token_granularity.csv",
            "token",
        ),
        (
            EVAL_ATTRIBUTE_FILES,
            OUT / "claude_tokens_and_time_attribute_granularity.csv",
            "attribute",
        ),
    ]
    for eval_files, path, label in outs:
        merged = load_eval_rows(eval_files)
        table = build_table(merged)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerows(table)
        print(f"Wrote {path} ({len(table)} rows, {label})")


if __name__ == "__main__":
    main()
