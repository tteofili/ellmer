#!/usr/bin/env python3
"""
Compute saliency–CF agreement and attribute–token agreement for Claude results under out/claude/.

Saliency–CF: uses scripts/saliency_cf_token_alignment.compute_metrics (top-k overlap + attribution mass on CF).
Per-explainer summaries (mean over datasets): saliency_cf_by_explainer_token.csv, saliency_cf_by_explainer_attribute.csv.
Attribute–token: uses scripts/attribute_token_alignment.compare (saliency alignment + CF alignment).

Run from repo root:  PYTHONPATH=. python scripts/compute_claude_alignment_metrics.py
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

from attribute_token_alignment import compare
from saliency_cf_token_alignment import compute_metrics

OUT = ROOT / "out" / "claude"

# Match eval.csv / faithfulness exports: same runs, paired token↔attribute roots.
SUITES = [
    {
        "name": "dm",
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
        "name": "books",
        "token": OUT / "books_token" / "run_4",
        "attr": OUT / "books_attribute" / "run_4",
        "datasets": [("books", "FB")],
    },
    {
        "name": "carparts",
        "token": OUT / "carparts_token" / "run_4",
        "attr": OUT / "carparts_attribute" / "run_4",
        "datasets": [("carparts", "FCP")],
    },
]

EXPLAINERS = [
    "zs_sample",
    "cot_sample",
    "fs_sample",
    "certa_sample",
    "hybrid_sample",
    "hybrid_lemon_minun_sample",
]

KS = [1, 2, 3, 4, 5]

LABEL = {
    "zs_sample": "ZS",
    "cot_sample": "CoT",
    "fs_sample": "ICL",
    "certa_sample": "CERTA",
    "hybrid_sample": "Ellmer_C",
    "hybrid_lemon_minun_sample": "Ellmer_L",
}


def _per_file_topk_and_mass(results: list) -> dict:
    """Mean attribution mass and top-1..4 overlap over valid instances in one results.json run."""
    nan = float("nan")
    if not results:
        return {
            "avg_attribution_mass_cf": nan,
            "mean_top1_overlap": nan,
            "mean_top2_overlap": nan,
            "mean_top3_overlap": nan,
            "mean_top4_overlap": nan,
        }
    n = len(results)
    out = {
        "avg_attribution_mass_cf": sum(r["attribution_mass_cf"] for r in results) / n,
    }
    for k in (1, 2, 3, 4):
        key = f"top_k_overlap@{k}"
        out[f"mean_top{k}_overlap"] = sum(r.get(key, 0.0) for r in results) / n
    return out


def run_sal_cf_all():
    rows = []
    for suite in SUITES:
        for ds, short in suite["datasets"]:
            for expl in EXPLAINERS:
                for gran, root in ("token", suite["token"]), ("attribute", suite["attr"]):
                    p = root / ds / f"{expl}_results.json"
                    if not p.is_file():
                        continue
                    results, agg = compute_metrics(
                        str(p),
                        ks=KS,
                        granularity="token" if gran == "token" else "attribute",
                        write_per_instance_csv=False,
                    )
                    pkm = _per_file_topk_and_mass(results)
                    row = {
                        "suite": suite["name"],
                        "dataset": ds,
                        "dataset_label": short,
                        "explainer": expl,
                        "explainer_label": LABEL[expl],
                        "granularity": gran,
                        "xd_avg_top_k_overlap": agg["xd_avg_top_k_overlap"],
                        "avg_attribution_mass_cf": pkm["avg_attribution_mass_cf"],
                        "mean_top1_overlap": pkm["mean_top1_overlap"],
                        "mean_top2_overlap": pkm["mean_top2_overlap"],
                        "mean_top3_overlap": pkm["mean_top3_overlap"],
                        "mean_top4_overlap": pkm["mean_top4_overlap"],
                        "n_valid": agg["n_valid"],
                        "n_total": agg["n_total"],
                        "n_skipped": agg["n_skipped"],
                        "valid_ratio": agg["valid_ratio"],
                    }
                    rows.append(row)
    return pd.DataFrame(rows)


def sal_cf_summary_by_explainer(sal: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """
    Average per-dataset means across all datasets, one row per explainer.
    Excludes rows with n_valid == 0 (no usable instances) from the mean.
    """
    if sal.empty:
        return pd.DataFrame()
    sub = sal[sal["granularity"] == granularity].copy()
    if sub.empty:
        return pd.DataFrame()
    has_data = sub["n_valid"] > 0
    sub = sub[has_data]
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "explainer",
                "explainer_label",
                "n_datasets",
                "avg_attribution_mass_cf",
                "top1_overlap",
                "top2_overlap",
                "top3_overlap",
                "top4_overlap",
            ]
        )
    g = sub.groupby("explainer", as_index=False)
    out = g.agg(
        explainer_label=("explainer_label", "first"),
        n_datasets=("dataset_label", "count"),
        avg_attribution_mass_cf=("avg_attribution_mass_cf", "mean"),
        top1_overlap=("mean_top1_overlap", "mean"),
        top2_overlap=("mean_top2_overlap", "mean"),
        top3_overlap=("mean_top3_overlap", "mean"),
        top4_overlap=("mean_top4_overlap", "mean"),
    )
    # Keep stable explainer order
    order = {e: i for i, e in enumerate(EXPLAINERS)}
    out["_o"] = out["explainer"].map(lambda x: order.get(x, 99))
    out = out.sort_values("_o").drop(columns="_o")
    return out.reset_index(drop=True)


def run_attr_token_all():
    frames = []
    for suite in SUITES:
        tok_b, att_b = suite["token"], suite["attr"]
        for ds, short in suite["datasets"]:
            pair = (str((tok_b / ds).resolve()) + "/", str((att_b / ds).resolve()) + "/")
            try:
                df = compare(
                    pair,
                    "claude_bedrock",
                    explainers=EXPLAINERS,
                    k_tokens=30,
                    k_attrs=3,
                    write_debug_csv=False,
                )
            except Exception as e:  # noqa: BLE001
                frames.append(
                    pd.DataFrame(
                        [
                            {
                                "suite": suite["name"],
                                "dataset": ds,
                                "dataset_label": short,
                                "explainer": "__error__",
                                "error": str(e),
                            }
                        ]
                    )
                )
                continue
            if df is None or df.empty:
                continue
            df = df.copy()
            df.insert(0, "dataset_label", short)
            df.insert(0, "dataset", ds)
            df.insert(0, "suite", suite["name"])
            if "explainer" in df.columns:
                df["explainer_label"] = df["explainer"].map(LABEL)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    sal = run_sal_cf_all()
    out_sal = OUT / "saliency_cf_agreement_claude.csv"
    sal.to_csv(out_sal, index=False)
    print(f"Wrote {out_sal} ({len(sal)} rows)")

    tok_s = sal_cf_summary_by_explainer(sal, "token")
    out_tok = OUT / "saliency_cf_by_explainer_token.csv"
    tok_s.to_csv(out_tok, index=False, float_format="%.6f")
    print(f"Wrote {out_tok} ({len(tok_s)} rows)")

    attr_s = sal_cf_summary_by_explainer(sal, "attribute")
    out_attr = OUT / "saliency_cf_by_explainer_attribute.csv"
    attr_s.to_csv(out_attr, index=False, float_format="%.6f")
    print(f"Wrote {out_attr} ({len(attr_s)} rows)")

    at = run_attr_token_all()
    out_at = OUT / "attribute_token_agreement_claude.csv"
    at.to_csv(out_at, index=False)
    print(f"Wrote {out_at} ({len(at)} rows)")


if __name__ == "__main__":
    main()
