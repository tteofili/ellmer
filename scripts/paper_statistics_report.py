#!/usr/bin/env python3
"""
Compile benchmark-paired statistics, concordance bootstrap CIs, and optional run-level
significance (from eval_all_runs) into CSVs and a short Markdown + HTML report.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ellmer.concordance_bootstrap import concordance_bootstrap_long, paired_concordance_two_files
from ellmer.stats_inference import (
    benjamini_hochberg_fdr,
    benchmark_paired_comparisons_dataframe,
    compute_run_pairwise_wilcoxon,
    compute_run_summary,
)

def _load_agg_significance():
    p = _ROOT / "scripts" / "aggregate_significance.py"
    spec = importlib.util.spec_from_file_location("aggregate_significance", p)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load aggregate_significance.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m

DEFAULT_BENCHMARK_METRICS: Tuple[str, ...] = (
    "faithfulness_auc",
    "validity",
    "proximity",
    "sparsity",
    "diversity",
)


def _md_cell(s) -> str:
    if s is None or (isinstance(s, float) and s != s):
        return "nan"
    t = str(s)
    return t.replace("|", "\\|").replace("\n", " ")


def df_to_markdown_table(df: pd.DataFrame, max_rows: int = 500) -> str:
    if df.empty:
        return "_No rows._\n"
    d = df.head(max_rows)
    lines = [
        "| " + " | ".join(_md_cell(c) for c in d.columns) + " |",
        "|" + "|".join("---" for _ in d.columns) + "|",
    ]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(_md_cell(x) for x in row) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_({len(df) - max_rows} more rows omitted)_\n")
    return "\n".join(lines) + "\n"


def df_to_html_table(df: pd.DataFrame, max_rows: int = 500) -> str:
    if df.empty:
        return "<p><em>No rows.</em></p>\n"
    d = df.head(max_rows)
    th = "".join(f"<th>{html.escape(str(c))}</th>" for c in d.columns)
    trs: List[str] = []
    for _, row in d.iterrows():
        tds = "".join(f"<td>{html.escape(str(x)) if x is not None and str(x) != 'nan' else ''}</td>" for x in row)
        trs.append(f"<tr>{tds}</tr>")
    extra = f"<p><em>{len(df) - max_rows} more rows omitted</em></p>\n" if len(df) > max_rows else ""
    return (
        f'<table class="paper-stats" border="1" cellpadding="4" cellspacing="0"><thead><tr>{th}</tr></thead><tbody>\n'
        + "\n".join(trs)
        + f"\n</tbody></table>\n{extra}"
    )


def add_bh_column(df: pd.DataFrame, pcol: str = "p_t_two_sided", out: str = "p_t_bh_fdr_5pct") -> pd.DataFrame:
    if df.empty or pcol not in df.columns:
        return df
    out_df = df.copy()
    p_adj, _ = benjamini_hochberg_fdr(out_df[pcol].tolist(), alpha=0.05)
    out_df[out] = p_adj
    return out_df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build statistical report from metrics CSV, concordance, and/or eval runs."
    )
    ap.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Long-form metrics (e.g. metrics_aggregate_metrics.csv)",
    )
    ap.add_argument(
        "--experiments-root",
        type=Path,
        default=None,
        help="Root with experiments/**/concordance/**/**.csv for bootstrap CIs",
    )
    ap.add_argument("--session-dir", type=Path, default=None, help="Session dir for run_*/eval.csv")
    ap.add_argument("--eval-all-runs", type=Path, default=None, help="Path to eval_all_runs.csv")
    ap.add_argument(
        "--output-dir", type=Path, default=Path("paper_stats_out"), help="Output directory for CSV/MD/HTML"
    )
    ap.add_argument(
        "--output-basename", type=str, default="paper_stats", help="Prefix for written files"
    )
    ap.add_argument(
        "--method-pair",
        nargs=2,
        action="append",
        metavar=("A", "B"),
        help="Explainer keys to compare, e.g. --method-pair zs_sample fs_sample. Repeat for multiple.",
    )
    ap.add_argument(
        "--benchmark-metrics",
        type=str,
        default=",".join(DEFAULT_BENCHMARK_METRICS),
        help="Comma-separated columns in the metrics csv",
    )
    ap.add_argument(
        "--prediction-split",
        type=str,
        default="all",
        help="Filter prediction_split (default all; ignored if column missing).",
    )
    ap.add_argument(
        "--concordance-filter",
        type=str,
        default="all",
        help="all | both_correct | either_incorrect for concordance",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--paired-concordance-a", type=Path, default=None)
    ap.add_argument("--paired-concordance-b", type=Path, default=None)
    ap.add_argument(
        "--paired-concordance-column", type=str, default="kt", help="kt or cos_sim for two-file test"
    )
    ap.add_argument(
        "--no-benchmark-fdr", action="store_true", help="Omit Benjamini–Hochberg on benchmark p-values"
    )
    ap.add_argument("--confidence", type=float, default=0.95, help="CI level (default 0.95)")
    args = ap.parse_args()

    conf = float(args.confidence)
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    base = args.output_basename

    pairs: List[Tuple[str, str]] = list(args.method_pair) if args.method_pair else []
    mcols = [x.strip() for x in str(args.benchmark_metrics).split(",") if x.strip()]

    md_body: List[str] = [
        f"# Paper statistics report\n",
        f"**Confidence level:** {conf * 100:.0f}%\n",
    ]
    h_body: List[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8' /><title>Paper statistics</title></head><body>\n"
        f"<h1>Paper statistics report</h1><p><strong>Confidence</strong> {conf * 100:.0f}%</p>\n"
    ]

    # --- 1) Benchmark-paired
    pbench = outdir / f"{base}_paired_benchmark.csv"
    bench_df = pd.DataFrame()
    if args.metrics_csv and args.metrics_csv.is_file():
        dfm = pd.read_csv(args.metrics_csv)
        if pairs:
            bench_df = benchmark_paired_comparisons_dataframe(
                dfm,
                method_pairs=pairs,
                metrics=tuple(mcols),
                prediction_split=args.prediction_split,
                confidence=conf,
                n_bootstrap=int(args.n_bootstrap),
            )
        if not bench_df.empty and not args.no_benchmark_fdr:
            bench_df = add_bh_column(bench_df, "p_t_two_sided", "p_t_bh_fdr_5pct")
    bench_df.to_csv(pbench, index=False)
    h_body.append(f"<h2>Benchmark-paired (dataset replicates)</h2>\n")
    if not (args.metrics_csv and args.metrics_csv.is_file()):
        h_body.append("<p><em>No --metrics-csv: skipped</em></p>\n")
        md_body.append("## Benchmark-paired (dataset replicates)\n*No --metrics-csv: skipped.*\n")
    elif not pairs:
        h_body.append(
            "<p><em>Pass repeated <code>--method-pair A B</code> to run paired comparisons</em></p>\n"
        )
        h_body.append(df_to_html_table(bench_df))
        md_body.append(
            "## Benchmark-paired (dataset replicates)\n"
            "*(Add `--method-pair explainer_a explainer_b` for paired tests.)*\n"
            + df_to_markdown_table(bench_df)
        )
    else:
        h_body.append(df_to_html_table(bench_df))
        md_body.append("## Benchmark-paired (dataset replicates)\n" + df_to_markdown_table(bench_df))
    h_body.append(f'<p>CSV: <code>{pbench.name}</code></p>')

    # --- 2) Concordance bootstrap
    pcon = outdir / f"{base}_concordance_bootstrap.csv"
    concord_df = pd.DataFrame()
    if args.experiments_root and args.experiments_root.is_dir():
        concord_df = concordance_bootstrap_long(
            args.experiments_root,
            concordance_filter=args.concordance_filter,
            columns=("kt", "cos_sim"),
            confidence=conf,
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.seed),
        )
    concord_df.to_csv(pcon, index=False)
    md_body.append("## Instance-level concordance (bootstrap mean CI)\n" + df_to_markdown_table(concord_df))
    h_body.append(
        f"<h2>Instance-level concordance (bootstrap mean CI), filter={html.escape(str(args.concordance_filter))}</h2>\n"
        + df_to_html_table(concord_df)
        + f'<p>CSV: <code>{pcon.name}</code></p>'
    )

    # --- 3) Optional two concordance files
    p_pair_json = outdir / f"{base}_concordance_paired_two_files.json"
    two: Optional[dict] = None
    if args.paired_concordance_a and args.paired_concordance_b:
        if args.paired_concordance_a.is_file() and args.paired_concordance_b.is_file():
            res = paired_concordance_two_files(
                str(args.paired_concordance_a),
                str(args.paired_concordance_b),
                column=args.paired_concordance_column,
                concordance_filter=args.concordance_filter,
                confidence=conf,
                n_bootstrap=int(args.n_bootstrap),
                seed=int(args.seed),
            )
            if isinstance(res, dict):
                two = res
    if two is not None:
        p_pair_json.write_text(json.dumps(two, indent=2, default=str), encoding="utf-8")
        h_body.append(
            f"<h2>Two concordance files (paired on instances)</h2><pre>{html.escape(str(two))}</pre>\n"
            f"<p>JSON: {p_pair_json.name}</p>"
        )
        md_body.append("## Two concordance files (paired)\n\n" + f"```\n{json.dumps(two, indent=2, default=str)}\n```\n")
        print(f"Wrote {p_pair_json}")

    # --- 4) Run-level
    prun = outdir / f"{base}_run_summary.csv"
    psig = outdir / f"{base}_run_significance.csv"
    rsum = pd.DataFrame()
    rsig = pd.DataFrame()
    if (args.session_dir and args.session_dir.is_dir()) or (args.eval_all_runs and args.eval_all_runs.is_file()):
        _agg = _load_agg_significance()
        dfr = _agg.load_runs(
            session_dir=str(args.session_dir) if args.session_dir else None,
            eval_all_runs_path=str(args.eval_all_runs) if args.eval_all_runs else None,
        )
        metric_cols = [c for c in _agg.METRIC_COLUMNS if c in dfr.columns]
        if not metric_cols:
            for c in dfr.columns:
                if c in ("dataset", "model", "run_id") or c == "Unnamed: 0":
                    continue
                if pd.api.types.is_numeric_dtype(dfr[c]):
                    metric_cols.append(c)
        rsum = compute_run_summary(dfr, metric_cols, confidence=conf)
        rsig = compute_run_pairwise_wilcoxon(dfr, metric_cols, confidence=conf)
    rsum.to_csv(prun, index=False)
    rsig.to_csv(psig, index=False)
    md_body.append("## Run-level (paired across `run_id`)\n" + df_to_markdown_table(rsum))
    md_body.append("### Run-level pairwise (Wilcoxon, paired t CI of diff)\n" + df_to_markdown_table(rsig))
    h_body.append(
        f"<h2>Run-level summary</h2>{df_to_html_table(rsum)}<p>CSV: {prun.name}</p>\n"
        f"<h3>Run-level pairwise</h3>{df_to_html_table(rsig)}<p>CSV: {psig.name}</p>\n"
    )

    p_md = outdir / f"{base}.md"
    p_html = outdir / f"{base}.html"
    p_md.write_text("".join(md_body), encoding="utf-8")
    h_body.append("</body></html>\n")
    p_html.write_text("".join(h_body), encoding="utf-8")

    print(f"Wrote {pbench}")
    print(f"Wrote {pcon}")
    print(f"Wrote {prun}")
    print(f"Wrote {psig}")
    print(f"Wrote {p_md}")
    print(f"Wrote {p_html}")


if __name__ == "__main__":
    main()
