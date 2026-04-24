#!/usr/bin/env python3
"""
Clarify what can (and cannot) be derived from paper_stats.md / paper_stats_paired_benchmark.csv
for faithfulness, vs separate Self vs Post-hoc absolute macro-means with bootstrap CIs.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _read_paired_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _parse_md_benchmark_table(path: Path) -> int:
    """Return row count in benchmark-paired table (for sanity check)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r"##\s*Benchmark-paired.*?\n\s*(\|[^\n]+(?:\n\|[^\n]+)+)", text, re.S | re.I
    )
    if not m:
        raise SystemExit("Could not find 'Benchmark-paired' markdown table.")
    table = m.group(1).strip().splitlines()
    if len(table) < 2:
        raise SystemExit("Table too short.")
    # data rows: skip header and separator
    n = 0
    for ln in table[2:]:
        if not ln.strip().startswith("|"):
            continue
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Explain what faithfulness numbers can be derived from paper_stats outputs."
        )
    )
    ap.add_argument(
        "--paired-csv",
        type=Path,
        default=Path("paper_stats_out/paper_stats_paired_benchmark.csv"),
    )
    ap.add_argument(
        "--md",
        type=Path,
        default=Path("paper_stats_out/paper_stats.md"),
        help="Optional: verify benchmark table can be parsed; compare row count to CSV.",
    )
    ap.add_argument("--out", type=Path, default=None, help="Write text summary here.")
    args = ap.parse_args()

    out_lines: list[str] = []
    w = out_lines.append

    w("=" * 70)
    w("Faithfulness: what `paper_stats.md` can and cannot give you")
    w("=" * 70)
    w("")
    w("CANNOT (from paper_stats benchmark table alone):")
    w("  - Two separate 'Self' and 'Post-hoc' ABSOLUTE macro-means and bootstrap 95% CIs.")
    w("  Reason: the table stores PAIRED DIFFERENCES: mean of (method_a - method_b) per")
    w("  shared dataset, with tests and CIs on that *difference* vector. It does not list")
    w("  the per-dataset values for A or for B, so you cannot recompute mean(A) and mean(B)")
    w("  (bootstrap over datasets) from the md/csv table alone — underdetermined.")
    w("")
    w("  This run's paired export only compares cot vs zs (and other pairs); there is no")
    w("  `certa_sample` row, so you also do not get Self vs post-hoc as a *pair* here.")
    w("")
    w("CAN (from the same columns):")
    w("  - Use mean_diff, ci_bootstrap_lower, ci_bootstrap_upper for the *gap* (A - B) when")
    w("    n_datasets >= 2, e.g. 'CoT minus ZS' faithfulness over N datasets, not the")
    w("    absolute level of either explainer type.")
    w("")

    p = args.paired_csv
    if not p.is_file():
        w(f"(Paired benchmark CSV not found: {p})")
    else:
        rows = _read_paired_csv(p)
        n_certa = sum(
            1
            for r in rows
            if "certa" in (r.get("method_a") or "").lower()
            or "certa" in (r.get("method_b") or "").lower()
        )
        w(f"Paired benchmark rows mentioning certa: {n_certa} (of {len(rows)}). ")

        faith = [r for r in rows if r.get("metric") == "faithfulness_auc"]
        with_nd = [
            r
            for r in faith
            if (r.get("n_datasets") or "").strip()
            and float((r.get("n_datasets") or "0") or 0) >= 2
            and (r.get("mean_diff") or "").strip()
            and (r.get("mean_diff") or "").strip().lower() not in ("nan", "")
        ]
        w("")
        w("Example: faithfulness_auc, n_datasets >= 2, first 3 rows (DIFFERENCE stats):")
        w("")
        for r in with_nd[:3]:
            w(
                f"  {r.get('model_name', '')} | {r.get('method_a')}-{r.get('method_b')} | "
                f"mean_diff={r.get('mean_diff')} n={r.get('n_datasets')} "
                f"boot=({r.get('ci_bootstrap_lower')}, {r.get('ci_bootstrap_upper')})"
            )
        if not with_nd:
            w("  (no rows)")

    w("")
    w("To get:  Self: mean ± half-width(95% bootstrap)  and  Post-hoc: mean ± ...")
    w("  use long metrics (one value per dataset per method), e.g.:")
    w("    out/claude/faithfulness_claude_*.csv")
    w("  and:  python scripts/faithfulness_bootstrap_datasets.py --out <file>")
    w("")

    if args.md.is_file() and p.is_file():
        try:
            n_md = _parse_md_benchmark_table(args.md)
            n_csv = len(_read_paired_csv(p))
            w(
                f"Benchmark table in paper_stats.md: {n_md} data rows; "
                f"paper_stats_paired_benchmark.csv: {n_csv} rows (authoritative, full export)."
            )
            if n_md < n_csv:
                w(
                    f"  (Markdown is capped at 500 rows in `paper_statistics_report.py`.)"
                )
        except SystemExit as e:
            w(f"(MD parse: {e})")
    s = "\n".join(out_lines) + "\n"
    print(s, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(s, encoding="utf-8")
        print("Wrote", args.out, file=sys.__stdout__)


if __name__ == "__main__":
    main()
