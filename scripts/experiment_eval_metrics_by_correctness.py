#!/usr/bin/env python3
"""
Recompute faithfulness and CF aggregate metrics (validity, etc.) per prediction split
(all / correct / incorrect) from existing ``experiments/**/**_results.json`` files.

Does **not** regenerate saliency or counterfactual explanations; only runs
``get_faithfulness`` / ``get_cf_metrics`` on stored ``data[]`` (still requires LLM
evaluation and predict calls).

Infers ``model_type`` and ``model_name`` from each result path via
:func:`ellmer.experiment_paths.parse_results_json_path`. Optional
``--inference-profile-map`` supplies Bedrock inference profile IDs for path prefixes.

**Datasets:** ``--base-dir`` must point to the same layout as ``scripts/eval.py``:
``{base_dir}/{dataset}/tableA.csv``, ``tableB.csv``, ``test.csv``, ``train.csv``.

Writes ``{prefix}_metrics.csv`` and optional ``{prefix}_paired_stats.csv``,
and prints a summary to stdout.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ellmer.experiment_metrics_recompute import default_lemon_kwargs, recompute_metrics_for_file
from ellmer.experiment_paths import parse_results_json_path
from ellmer.paired_metrics_diff import paired_difference_long
from ellmer.stats_inference import one_sample_t_and_wilcoxon_vs_zero


def _load_profile_map(path: Optional[Path]) -> List[Tuple[str, str]]:
    if path is None or not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: could not load inference profile map {path}: {e}", file=sys.stderr)
        return []
    if not isinstance(raw, dict):
        return []
    pairs: List[Tuple[str, str]] = []
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, str):
            pairs.append((k.replace("\\", "/").rstrip("/"), v))
    pairs.sort(key=lambda x: -len(x[0]))
    return pairs


def resolve_deployment_for_path(
    results_path: str,
    experiments_root: Path,
    default_deployment: str,
    profile_map: List[Tuple[str, str]],
) -> str:
    """Longest matching prefix in ``profile_map`` wins; else ``default_deployment``."""
    norm = results_path.replace("\\", "/")
    try:
        rel = str(Path(norm).resolve().relative_to(experiments_root.resolve()))
    except ValueError:
        rel = norm
    rel = rel.replace("\\", "/")
    for prefix, profile_id in profile_map:
        pre = prefix.replace("\\", "/").rstrip("/")
        if rel.startswith(pre) or rel.startswith("experiments/" + pre):
            return profile_id
    return default_deployment


def verify_base_dir_has_datasets(base_dir: Path, dataset_names: Sequence[str]) -> bool:
    """Return True if each dataset folder has the four expected CSV files."""
    ok = True
    for d in dataset_names:
        p = base_dir / d
        for name in ("tableA.csv", "tableB.csv", "test.csv", "train.csv"):
            if not (p / name).is_file():
                print(f"Missing {p / name}", file=sys.stderr)
                ok = False
    return ok


def paired_difference_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Build long-form correct-minus-incorrect diffs (shared with ``paired_metrics_diff``)."""
    return paired_difference_long(df)


def summarize_diffs(diff_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Overall, per explainer, per granularity: mean, std, n, t-test p, Wilcoxon p."""
    out: List[Dict[str, Any]] = []
    if diff_df.empty:
        return out

    def one_block(name: str, mask: pd.Series) -> None:
        for metric in ("faithfulness_auc", "validity"):
            col = "diff_correct_minus_incorrect"
            sub = diff_df.loc[mask & (diff_df["metric"] == metric), col].astype(float).values
            sub = sub[~np.isnan(sub)]
            n = len(sub)
            if n == 0:
                continue
            st = one_sample_t_and_wilcoxon_vs_zero(sub)
            mean = float(np.mean(sub))
            std = float(np.std(sub, ddof=1)) if n > 1 else 0.0
            row: Dict[str, Any] = {
                "block": name,
                "metric": metric,
                "n_pairs": n,
                "mean_diff": mean,
                "std_diff": std,
                "t_statistic": st.get("t_statistic", float("nan")),
                "p_t_two_sided": st.get("p_t_two_sided", float("nan")),
                "p_wilcoxon_two_sided": st.get("p_wilcoxon_two_sided", float("nan")),
            }
            out.append(row)

    one_block("overall", pd.Series(True, index=diff_df.index))

    if "explainer" in diff_df.columns:
        for ex in sorted(diff_df["explainer"].dropna().unique()):
            one_block(f"explainer={ex}", diff_df["explainer"] == ex)

    if "granularity" in diff_df.columns:
        for gr in sorted(diff_df["granularity"].dropna().unique()):
            one_block(f"granularity={gr}", diff_df["granularity"] == gr)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute faithfulness and CF metrics by correctness from experiments/ result JSON."
    )
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Dataset root: {base_dir}/{dataset}/tableA.csv, tableB.csv, test.csv, train.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--output-prefix", type=str, default="experiment_eval_by_correctness")
    parser.add_argument("--dataset", type=str, default=None, help="Only this dataset name")
    parser.add_argument(
        "--granularity",
        type=str,
        default=None,
        metavar="attribute|token",
        help="If set, only include runs with this granularity",
    )
    parser.add_argument(
        "--path-glob",
        type=str,
        default=None,
        help="Unix glob matched against path relative to --experiments-root (e.g. 'bedrock/**')",
    )
    parser.add_argument(
        "--deployment-name",
        type=str,
        default="",
        help="Default Azure/OpenAI deployment; for Bedrock often empty so model ID from path is used. "
        "Per-path overrides via --inference-profile-map.",
    )
    parser.add_argument(
        "--inference-profile-map",
        type=Path,
        default=None,
        help='JSON object: { "path/prefix/from/experiments": "inference-profile-id", ... }',
    )
    parser.add_argument(
        "--model-type-override",
        type=str,
        default=None,
        help="If set, use this model_type for all files instead of inferring from path",
    )
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--num-triangles", type=int, default=10)
    parser.add_argument("--samples", type=int, default=-1)
    parser.add_argument("--tag", type=str, default="sample")
    parser.add_argument("--skip-stats", action="store_true", help="Only write metrics CSV, no paired analysis")
    parser.add_argument(
        "--verify-base-dir",
        action="store_true",
        help="Check that --base-dir contains expected CSVs for each dataset encountered (and exit 1 if not)",
    )
    args = parser.parse_args()
    if args.granularity is not None and args.granularity not in ("attribute", "token"):
        parser.error("--granularity must be 'attribute' or 'token'")

    root = args.experiments_root
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    base_dir = args.base_dir
    if not base_dir.is_dir():
        print(f"Not a directory: {base_dir}", file=sys.stderr)
        sys.exit(1)

    profile_pairs = _load_profile_map(args.inference_profile_map)
    results_paths = sorted(root.rglob("*_results.json"))
    metric_rows: List[Dict[str, Any]] = []
    datasets_seen: set[str] = set()

    for p in results_paths:
        path_str = str(p.resolve())
        rel = str(p.relative_to(root)).replace("\\", "/")
        if args.path_glob and not fnmatch.fnmatch(rel, args.path_glob):
            continue
        info = parse_results_json_path(path_str)
        if info is None:
            continue
        if args.dataset and info.dataset != args.dataset:
            continue
        if args.granularity and info.granularity != args.granularity:
            continue

        datasets_seen.add(info.dataset)
        mt = args.model_type_override or info.model_type
        dep = resolve_deployment_for_path(path_str, root, args.deployment_name, profile_pairs)

        rows = recompute_metrics_for_file(
            path_str,
            info,
            str(base_dir.resolve()),
            args.samples,
            mt,
            dep,
            args.temperature,
            args.num_triangles,
            args.tag,
            default_lemon_kwargs(),
            source="metrics_recompute",
        )
        metric_rows.extend(rows)

    if args.verify_base_dir:
        if not datasets_seen:
            print("No datasets found in matching results; skip verify.", file=sys.stderr)
        elif not verify_base_dir_has_datasets(base_dir, sorted(datasets_seen)):
            sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    df_m = pd.DataFrame(metric_rows)
    metrics_path = args.output_dir / f"{prefix}_metrics.csv"
    df_m.to_csv(metrics_path, index=False)
    print(f"Wrote {metrics_path} ({len(df_m)} rows)")

    if args.skip_stats or df_m.empty:
        return

    diff_df = paired_difference_stats(df_m)
    if not diff_df.empty:
        diff_path = args.output_dir / f"{prefix}_paired_diff_rows.csv"
        diff_df.to_csv(diff_path, index=False)
        print(f"Wrote {diff_path} ({len(diff_df)} rows)")
    stats_rows = summarize_diffs(diff_df)
    df_stats = pd.DataFrame(stats_rows)
    stats_path = args.output_dir / f"{prefix}_paired_stats.csv"
    df_stats.to_csv(stats_path, index=False)
    print(f"Wrote {stats_path} ({len(df_stats)} rows)")

    if diff_df.empty:
        print("No paired correct/incorrect rows with finite metrics for diff analysis.", file=sys.stderr)
        return

    print("\nPaired differences (correct - incorrect):\n")
    if not df_stats.empty:
        with pd.option_context("display.max_rows", None, "display.width", 120):
            print(df_stats.to_string(index=False))
    print(f"\nPer-stratum diff rows: {len(diff_df)} (see {prefix}_paired_diff_rows.csv)", file=sys.stderr)


if __name__ == "__main__":
    main()
