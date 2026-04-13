#!/usr/bin/env python3
"""
Walk experiments/ and aggregate faithfulness, counterfactual metrics, and
inter-explainer concordance (alignment) metrics.

Outputs (default prefix ``metrics_aggregate`` in --output-dir):

- ``{prefix}_metrics.csv`` — long-form rows with grouping keys and ``prediction_split``
  (``all``, ``correct``, ``incorrect``). Fast path reads JSON: top-level ``metrics`` yields
  ``all`` only; if ``metrics_by_prediction_split`` is present (e.g. from eval), three splits
  without recomputation. Otherwise ``correct`` / ``incorrect`` require ``--recompute`` + ``--base-dir``.

- ``{prefix}_alignment.csv`` — concordance CSV aggregates (mean kt, cos_sim, pred agreement).

**Concordance filtering** (``--concordance-filter``):

- ``all``: use every instance row in the concordance CSV.
- ``both_correct``: keep rows where both explainers match the label
  (``int(pred1)==int(label)`` and ``int(pred2)==int(label)``, with label as 0/1).
- ``either_incorrect``: complement of ``both_correct`` among rows with valid preds.

Recomputation calls the same explainer stack as ``scripts/eval.py`` (LLM / API access required).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Repo root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import ellmer.metrics
from ellmer.experiment_paths import parse_results_json_path
from ellmer.metrics_json_rows import (
    pred_int as _pred_int,
    prediction_indices,
    faithfulness_scalar as _faithfulness_scalar,
    fast_rows_from_payload as _fast_rows_from_payload,
)
from ellmer.utils import merge_sources


def _safe_load_results_json(path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a *_results.json file. Returns None if the file is missing, unreadable,
    or not valid JSON (common with interrupted writes or non-JSON artifacts).
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"Warning: could not read {path}: {e}", file=sys.stderr)
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(
            f"Warning: invalid JSON in {path} ({e.msg} at char {e.pos}); skipping",
            file=sys.stderr,
        )
        return None


def _load_eval_script():
    """Load scripts/eval.py as a module (build_self_explainers, _build_few_shot_example, etc.)."""
    path = Path(__file__).resolve().parent / "eval.py"
    spec = importlib.util.spec_from_file_location("ellmer_eval_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _split_concordance_stem(stem: str) -> Tuple[str, str]:
    """
    Split filename stem ``p1_p2`` into two explainer keys. Names always end with ``_sample``
    in this codebase; match longest known suffix first.
    """
    known = sorted(
        [
            "hybrid_lemon_minun_sample",
            "hybrid_sample",
            "certa_sample",
            "cot_sample",
            "fs_sample",
            "zs_sample",
        ],
        key=len,
        reverse=True,
    )
    for p2 in known:
        if stem.endswith(p2) and len(stem) > len(p2):
            prefix = stem[: -len(p2)].rstrip("_")
            p1 = prefix
            if p1:
                return p1, p2
    mid = stem.rfind("_")
    if mid <= 0:
        return stem, ""
    return stem[:mid], stem[mid + 1 :]


def _parse_concordance_path(csv_path: str) -> Optional[Dict[str, Any]]:
    """Infer model_type, model_name, granularity, session, run_id, dataset from concordance CSV path."""
    parts = Path(csv_path).parts
    if "concordance" not in parts:
        return None
    ci = parts.index("concordance")
    dataset = parts[ci + 1] if ci + 1 < len(parts) else ""
    prefix_parts = list(parts[:ci])
    if "experiments" not in prefix_parts:
        return None
    ei = prefix_parts.index("experiments")
    segs = prefix_parts[ei + 1 :]
    run_id = None
    for s in segs:
        if s.startswith("run_") and s[4:].isdigit():
            run_id = int(s[4:])
            break
    if len(segs) >= 5 and run_id is not None:
        model_type, model_name, granularity, session = segs[0], segs[1], segs[2], segs[3]
    elif len(segs) >= 4:
        model_type, model_name, granularity = segs[0], segs[1], segs[2]
        session = segs[3] if len(segs) > 3 else None
    else:
        return None
    stem = Path(csv_path).stem
    p1, p2 = _split_concordance_stem(stem)
    return {
        "model_type": model_type,
        "model_name": model_name,
        "granularity": granularity,
        "session": session,
        "run_id": run_id,
        "dataset": dataset,
        "explainer_a": p1,
        "explainer_b": p2,
        "csv_path": csv_path,
    }


def _filter_concordance_df(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if df.empty or mode == "all":
        return df
    if "pred1" not in df.columns or "pred2" not in df.columns or "label" not in df.columns:
        return df
    p1 = df["pred1"].map(_pred_int)
    p2 = df["pred2"].map(_pred_int)
    lb = df["label"].map(lambda x: _pred_int(x) if not isinstance(x, (int, float)) else int(x))
    both = (p1 == lb) & (p2 == lb)
    if mode == "both_correct":
        return df.loc[both].copy()
    if mode == "either_incorrect":
        return df.loc[~both].copy()
    return df


def aggregate_concordance_csvs(
    experiments_root: Path,
    concordance_filter: str,
) -> pd.DataFrame:
    rows = []
    pattern = str(experiments_root / "**" / "concordance" / "**" / "*.csv")
    import glob

    for csv_path in glob.glob(pattern, recursive=True):
        meta = _parse_concordance_path(csv_path)
        if meta is None:
            continue
        try:
            df = pd.read_csv(csv_path, index_col=0)
        except Exception:
            continue
        df_f = _filter_concordance_df(df, concordance_filter)
        n = len(df_f)
        row: Dict[str, Any] = {
            **{k: v for k, v in meta.items() if k != "csv_path"},
            "concordance_filter": concordance_filter,
            "n_instances": n,
        }
        if "kt" in df_f.columns:
            row["mean_kt"] = float(df_f["kt"].mean()) if n else np.nan
        if "cos_sim" in df_f.columns:
            row["mean_cos_sim"] = float(df_f["cos_sim"].mean()) if n else np.nan
        if "agree" in df_f.columns:
            row["pred_pair_agreement_rate"] = float(df_f["agree"].mean()) if n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _build_ellmers(
    eval_mod,
    base_dir: str,
    dataset_name: str,
    samples: int,
    llm_config: dict,
    temperature: float,
    granularity: str,
    num_triangles: int,
    tag: str,
    lemon_kwargs: dict,
):
    """Return (ellmers dict, test_df) mirroring scripts/eval.py."""
    dataset_dir = "/".join([base_dir, dataset_name])
    lsource = pd.read_csv(dataset_dir + "/tableA.csv")
    rsource = pd.read_csv(dataset_dir + "/tableB.csv")
    test = pd.read_csv(dataset_dir + "/test.csv")
    train = pd.read_csv(dataset_dir + "/train.csv")

    test_df = merge_sources(test, "ltable_", "rtable_", lsource, rsource, ["label"], [], samples=samples)
    train_df = merge_sources(train, "ltable_", "rtable_", lsource, rsource, ["label"], [], samples=samples)

    certa = eval_mod.LLMCertaExplainer(lsource, rsource)
    explainers = eval_mod.build_self_explainers(llm_config, temperature, granularity)
    zeroshot = explainers["zeroshot"]
    cot = explainers["cot"]
    cot_with_why = explainers["cot_with_why"]
    predict_only = explainers["predict_only"]

    few_shot_no = 1
    train_data_matching_df = train_df[train_df["label"] == 1][:few_shot_no]
    train_data_non_matching_df = train_df[train_df["label"] == 0][:few_shot_no]
    data_df = pd.concat([train_data_matching_df, train_data_non_matching_df])
    n_fs = len(data_df)
    examples = []
    for idx in range(n_fs):
        _, ex = eval_mod._build_few_shot_example(cot_with_why, idx, data_df)
        if ex is not None:
            examples.append(ex)

    fs1 = eval_mod.ICLSelfExplainer(
        examples=examples,
        explanation_granularity=granularity,
        deployment_name=llm_config["deployment_name"],
        temperature=temperature,
        model_name=llm_config["model_name"],
        model_type=llm_config["model_type"],
        prompts={"fs": "ellmer/prompts/fs1.txt", "input": "record1:\n{ltuple}\n record2:\n{rtuple}\n"},
    )

    zs_h = zeroshot.fork_for_stats()
    cot_h = cot.fork_for_stats()
    cwhy_h = cot_with_why.fork_for_stats()
    cot_lemon = cot.fork_for_stats()

    ellmers = {
        "zs_" + tag: zeroshot,
        "cot_" + tag: cot_with_why,
        "fs_" + tag: fs1,
        "certa_" + tag: eval_mod.FullCerta(granularity, predict_only, certa, num_triangles),
        "hybrid_" + tag: eval_mod.HybridCerta(
            granularity,
            cot_h,
            certa,
            [zs_h, cot_h, cwhy_h],
            num_triangles=num_triangles,
        ),
        "hybrid_lemon_minun_" + tag: eval_mod.HybridLemonMinun(
            granularity,
            cot_lemon,
            certa,
            num_triangles=num_triangles,
            **lemon_kwargs,
        ),
    }
    return ellmers, test_df


def _recompute_metrics_for_file(
    path: str,
    info,
    base_dir: str,
    samples: int,
    model_type: str,
    deployment_name: str,
    temperature: float,
    num_triangles: int,
    tag: str,
    lemon_kwargs: dict,
) -> List[Dict[str, Any]]:
    eval_mod = _load_eval_script()
    llm_config = {
        "model_type": model_type,
        "model_name": info.model_name,
        "deployment_name": deployment_name,
        "tag": tag,
    }
    ellmers, test_df = _build_ellmers(
        eval_mod,
        base_dir,
        info.dataset,
        samples,
        llm_config,
        temperature,
        info.granularity,
        num_triangles,
        tag,
        lemon_kwargs,
    )
    payload = _safe_load_results_json(Path(path))
    if payload is None:
        return []
    data = payload.get("data") or []
    expdir = os.path.dirname(path) + "/"
    test_data_df = test_df if samples < 0 else test_df[:samples]
    key = info.explainer_key
    if key not in ellmers:
        return []
    llm = ellmers[key]
    _, correct_idx, incorrect_idx = prediction_indices(data)
    out_rows = []
    for split_name, idx in (
        ("all", None),
        ("correct", correct_idx),
        ("incorrect", incorrect_idx),
    ):
        if idx is not None and len(idx) == 0:
            continue
        faith = ellmer.metrics.get_faithfulness(
            [key],
            llm.evaluation,
            expdir,
            test_data_df,
            results_by_name={key: {"data": data}},
            row_indices=idx,
        )
        cf = ellmer.metrics.get_cf_metrics(
            [key],
            llm.predict,
            expdir,
            test_data_df,
            results_by_name={key: {"data": data}},
            row_indices=idx,
        )
        f_sc = _faithfulness_scalar(faith, key)
        cf_row = cf.get(key, {})
        out_rows.append(
            {
                "dataset": info.dataset,
                "model_name": info.model_name,
                "model_type": model_type,
                "granularity": info.granularity,
                "session": info.session,
                "run_id": info.run_id,
                "explainer": key,
                "prediction_split": split_name,
                "faithfulness_auc": f_sc,
                "validity": cf_row.get("validity"),
                "proximity": cf_row.get("proximity"),
                "sparsity": cf_row.get("sparsity"),
                "diversity": cf_row.get("diversity"),
                "source": "recompute",
            }
        )
    return out_rows


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics from experiments/.")
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Dataset root (required with --recompute or --prediction-splits)",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Emit prediction_split rows all, correct, and incorrect by recomputing faithfulness and CF "
        "metrics from raw result data using the same explainers as eval (requires --base-dir; LLM/API).",
    )
    parser.add_argument(
        "--prediction-splits",
        action="store_true",
        dest="prediction_splits",
        help="Alias for --recompute: same behavior (emit rows for all, correct, incorrect).",
    )
    parser.add_argument("--model-type", type=str, default=None, help="Override model type for recompute")
    parser.add_argument("--deployment-name", type=str, default="gpt-35-turbo")
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--num-triangles", type=int, default=10)
    parser.add_argument("--samples", type=int, default=-1)
    parser.add_argument("--tag", type=str, default="sample", help="Explainer tag suffix (zs_<tag>)")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--output-prefix", type=str, default="metrics_aggregate")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset name")
    parser.add_argument("--granularity", type=str, default=None, help="Filter attribute|token")
    parser.add_argument(
        "--concordance-filter",
        choices=["all", "both_correct", "either_incorrect", "each"],
        default="each",
        help="Filter concordance rows: 'each' writes all three modes (all, both_correct, either_incorrect); "
        "otherwise a single mode.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--verbose-json",
        action="store_true",
        help="With skipped JSON files, print path for each (default: one summary line only)",
    )
    args = parser.parse_args()

    do_recompute = bool(args.recompute or args.prediction_splits)
    if do_recompute and not args.base_dir:
        parser.error("--recompute / --prediction-splits requires --base-dir")
    if do_recompute and args.workers != 1:
        print("Note: using --workers 1 for --recompute to avoid overlapping LLM calls", file=sys.stderr)
        args.workers = 1

    root: Path = args.experiments_root
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    results_paths = sorted(root.rglob("*_results.json"))
    metric_rows: List[Dict[str, Any]] = []
    skipped_json_paths: List[str] = []

    def process_one(p: Path) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Returns (rows, skipped_path_if_invalid_json)."""
        path_str = str(p)
        info = parse_results_json_path(path_str)
        if info is None:
            return [], None
        if args.dataset and info.dataset != args.dataset:
            return [], None
        if args.granularity and info.granularity != args.granularity:
            return [], None
        if do_recompute:
            if not args.base_dir:
                print("--recompute requires --base-dir", file=sys.stderr)
                return [], None
            mt = args.model_type or info.model_type
            lemon_kwargs = {
                "lem_num_samples": None,
                "cf_max_evals": 800,
                "max_iterations": 10,
                "cf_method": "greedy",
                "lime_cap_samples": True,
            }
            rows = _recompute_metrics_for_file(
                path_str,
                info,
                args.base_dir,
                args.samples,
                mt,
                args.deployment_name,
                args.temperature,
                args.num_triangles,
                args.tag,
                lemon_kwargs,
            )
            return rows, None
        payload = _safe_load_results_json(p)
        if payload is None:
            return [], path_str
        if payload.get("metrics"):
            return _fast_rows_from_payload(payload, info), None
        return [], None

    if args.workers <= 1:
        for p in results_paths:
            rows, skip = process_one(p)
            metric_rows.extend(rows)
            if skip:
                skipped_json_paths.append(skip)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_one, p) for p in results_paths]
            for fut in as_completed(futs):
                rows, skip = fut.result()
                metric_rows.extend(rows)
                if skip:
                    skipped_json_paths.append(skip)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    df_m = pd.DataFrame(metric_rows)
    df_m.to_csv(args.output_dir / f"{prefix}_metrics.csv", index=False)

    if args.concordance_filter == "each":
        df_a = pd.concat(
            [aggregate_concordance_csvs(root, m) for m in ("all", "both_correct", "either_incorrect")],
            ignore_index=True,
        )
    else:
        df_a = aggregate_concordance_csvs(root, args.concordance_filter)
    df_a.to_csv(args.output_dir / f"{prefix}_alignment.csv", index=False)

    print(f"Wrote {args.output_dir / (prefix + '_metrics.csv')} ({len(df_m)} rows)")
    print(f"Wrote {args.output_dir / (prefix + '_alignment.csv')} ({len(df_a)} rows)")
    if not do_recompute:
        print(
            "Note: metrics rows use only JSON in each result file (fast path). "
            "For all/correct/incorrect without pre-stored splits, use --recompute --base-dir <datasets>, "
            "or produce result JSON with eval --split-metrics-by-correctness.",
            file=sys.stderr,
        )
    if skipped_json_paths:
        print(
            f"Skipped {len(skipped_json_paths)} result file(s) with invalid or unreadable JSON "
            f"(see warnings above).",
            file=sys.stderr,
        )
        if args.verbose_json:
            for sp in skipped_json_paths:
                print(f"  skipped: {sp}", file=sys.stderr)


if __name__ == "__main__":
    main()
