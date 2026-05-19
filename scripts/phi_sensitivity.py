"""
Run phi sensitivity for HybridCerta and HybridLemonMinun (phi in {0.25, 0.5, 0.75} by default).

Reuses the same dataset setup and ``run_explainer`` as ``scripts/eval.py``, but only runs the two hybrid
explainers, once per phi value, with distinct result keys so files do not overwrite.

By default, all phi hybrid explainers for a dataset run concurrently, and multiple ``--datasets`` are
processed concurrently (one thread per dataset). Use ``--no-parallel-explainers`` or
``--no-parallel-datasets`` to limit peak API load.

Example::

    python scripts/phi_sensitivity.py --base_dir /path/to/data --model_type azure_openai \\
        --datasets books --samples 20 --tag exp1 --output_dir ./experiments/phi_sweep/run_a
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, List, Tuple

import langchain
import pandas as pd
from langchain_core.caches import InMemoryCache
from langchain_community.cache import SQLiteCache

from ellmer.hybrid import HybridCerta
from ellmer.hybrid_lemon_minun import HybridLemonMinun
from ellmer.phi_sensitivity import aggregate_results_directory, phi_tag
from ellmer.post_hoc.certa_explain import LLMCertaExplainer
from ellmer.utils import merge_sources

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("ellmer_eval_script", os.path.join(_EVAL_DIR, "eval.py"))
_eval_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_eval_mod)
build_self_explainers = _eval_mod.build_self_explainers
run_explainer = _eval_mod.run_explainer


def _run_explainers_for_dataset(
    ellmers: dict,
    test_df,
    samples: int,
    expdir: str,
    quantitative: bool,
    dataset_name: str,
    workers: int,
    parallel_explainers: bool,
) -> List[Any]:
    rows: List[Any] = []
    if parallel_explainers:
        n = max(1, len(ellmers))
        by_key: dict = {}
        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = [
                pool.submit(run_explainer, key, llm, test_df, samples, expdir, quantitative, dataset_name, workers)
                for key, llm in ellmers.items()
            ]
            for fut in as_completed(futs):
                key, output_file_path, eval_row = fut.result()
                print(f"[{dataset_name}] wrote {output_file_path}")
                by_key[key] = eval_row
        rows = [by_key[k] for k in sorted(by_key)]
    else:
        for key, llm in ellmers.items():
            _, output_file_path, eval_row = run_explainer(
                key, llm, test_df, samples, expdir, quantitative, dataset_name, workers=workers
            )
            print(f"[{dataset_name}] wrote {output_file_path}")
            rows.append(eval_row)
    return rows


def _process_one_dataset(
    d: str,
    session_base: str,
    base_dir: str,
    samples: int,
    num_triangles: int,
    explanation_granularity: str,
    quantitative: bool,
    tag: str,
    phi_values: list,
    zeroshot,
    cot,
    cot_with_why,
    workers: int,
    parallel_explainers: bool,
    lemon_minun_lem_num_samples=None,
    lemon_minun_cf_max_evals=800,
    lemon_minun_max_iterations=10,
    lemon_minun_cf_method="greedy",
    lemon_minun_lime_cap_samples=True,
) -> Tuple[str, List[Any], str]:
    """Load one dataset, run all phi explainers, return (dataset_name, eval_rows, expdir)."""
    expdir = os.path.join(session_base, d, "")
    print(f"dataset {d} -> {expdir}")
    dataset_dir = "/".join([base_dir, d])
    lsource = pd.read_csv(dataset_dir + "/tableA.csv")
    rsource = pd.read_csv(dataset_dir + "/tableB.csv")
    test = pd.read_csv(dataset_dir + "/test.csv")

    test_df = merge_sources(test, "ltable_", "rtable_", lsource, rsource, ["label"], [], samples=samples)
    certa = LLMCertaExplainer(lsource, rsource)

    ellmers = {}
    for phi in phi_values:
        pt = phi_tag(phi)
        zs_h = zeroshot.fork_for_stats()
        cot_h = cot.fork_for_stats()
        cwhy_h = cot_with_why.fork_for_stats()
        cot_lemon = cot.fork_for_stats()
        ellmers[f"hybrid_certa_phi{pt}_{tag}"] = HybridCerta(
            explanation_granularity,
            cot_h,
            certa,
            [zs_h, cot_h, cwhy_h],
            num_triangles=num_triangles,
            phi=phi,
        )
        ellmers[f"hybrid_lemon_minun_phi{pt}_{tag}"] = HybridLemonMinun(
            explanation_granularity,
            cot_lemon,
            certa,
            num_triangles=num_triangles,
            phi=phi,
            lem_num_samples=lemon_minun_lem_num_samples,
            cf_max_evals=lemon_minun_cf_max_evals,
            max_iterations=lemon_minun_max_iterations,
            cf_method=lemon_minun_cf_method,
            lime_cap_samples=lemon_minun_lime_cap_samples,
        )

    eval_rows = _run_explainers_for_dataset(
        ellmers, test_df, samples, expdir, quantitative, d, workers, parallel_explainers
    )
    return d, eval_rows, expdir


def run_phi_sensitivity(
    cache: str,
    samples: int,
    num_triangles: int,
    explanation_granularity: str,
    quantitative: bool,
    base_dir: str,
    dataset_names: list,
    model_type: str,
    model_name: str,
    deployment_name: str,
    tag: str,
    temperature: float,
    phi_values: list,
    output_dir: str | None,
    workers: int = 1,
    parallel_explainers: bool = True,
    parallel_datasets: bool = True,
    max_parallel_datasets: int | None = None,
    lemon_minun_lem_num_samples=None,
    lemon_minun_cf_max_evals=800,
    lemon_minun_max_iterations=10,
    lemon_minun_cf_method="greedy",
    lemon_minun_lime_cap_samples=True,
):
    if cache == "memory":
        langchain.llm_cache = InMemoryCache()
    elif cache == "sqlite":
        langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

    llm_config = {"model_type": model_type, "model_name": model_name, "deployment_name": deployment_name, "tag": tag}
    explainers_common = build_self_explainers(llm_config, temperature, explanation_granularity)
    zeroshot = explainers_common["zeroshot"]
    cot = explainers_common["cot"]
    cot_with_why = explainers_common["cot_with_why"]

    stamp = f"{datetime.now():%Y%m%d}_{datetime.now():%H_%M}"
    session_base = output_dir or f"./experiments/{model_type}/{model_name}/{explanation_granularity}/phi_sensitivity/{stamp}/"
    os.makedirs(session_base, exist_ok=True)

    all_eval_rows: List[Any] = []
    use_parallel_ds = parallel_datasets and len(dataset_names) > 1
    ds_workers = len(dataset_names) if max_parallel_datasets is None else min(len(dataset_names), max_parallel_datasets)
    ds_workers = max(1, ds_workers)

    if use_parallel_ds:
        with ThreadPoolExecutor(max_workers=ds_workers) as pool:
            futs = {
                pool.submit(
                    _process_one_dataset,
                    d,
                    session_base,
                    base_dir,
                    samples,
                    num_triangles,
                    explanation_granularity,
                    quantitative,
                    tag,
                    phi_values,
                    zeroshot,
                    cot,
                    cot_with_why,
                    workers,
                    parallel_explainers,
                    lemon_minun_lem_num_samples,
                    lemon_minun_cf_max_evals,
                    lemon_minun_max_iterations,
                    lemon_minun_cf_method,
                    lemon_minun_lime_cap_samples,
                ): d
                for d in dataset_names
            }
            by_dataset: dict = {}
            for fut in as_completed(futs):
                d, eval_rows, expdir = fut.result()
                by_dataset[d] = (eval_rows, expdir)
            for d in dataset_names:
                eval_rows, expdir = by_dataset[d]
                all_eval_rows.extend(eval_rows)
                agg_path = os.path.join(session_base, f"phi_table_{d}.csv")
                aggregate_results_directory(expdir).to_csv(agg_path, index=False)
                print(f"wrote {agg_path}")
    else:
        for d in dataset_names:
            _, eval_rows, expdir = _process_one_dataset(
                d,
                session_base,
                base_dir,
                samples,
                num_triangles,
                explanation_granularity,
                quantitative,
                tag,
                phi_values,
                zeroshot,
                cot,
                cot_with_why,
                workers,
                parallel_explainers,
                lemon_minun_lem_num_samples,
                lemon_minun_cf_max_evals,
                lemon_minun_max_iterations,
                lemon_minun_cf_method,
                lemon_minun_lime_cap_samples,
            )
            all_eval_rows.extend(eval_rows)
            agg_path = os.path.join(session_base, f"phi_table_{d}.csv")
            aggregate_results_directory(expdir).to_csv(agg_path, index=False)
            print(f"wrote {agg_path}")

    eval_df = pd.DataFrame(all_eval_rows)
    combined_path = os.path.join(session_base, "eval_phi_sensitivity.csv")
    eval_df.to_csv(combined_path)
    print(f"wrote {combined_path}")

    return session_base


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phi sensitivity for hybrid CERTA and Lemon/Minun.")
    parser.add_argument("--base_dir", type=str, required=True, help="datasets base directory")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["azure_openai", "falcon", "llama2", "hf", "bedrock"],
        required=True,
    )
    parser.add_argument("--datasets", type=str, nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=-1)
    parser.add_argument("--cache", type=str, choices=["", "sqlite", "memory"], default="")
    parser.add_argument("--num_triangles", type=int, default=50)
    parser.add_argument("--granularity", type=str, default="attribute", choices=["attribute", "token"])
    parser.add_argument("--quantitative", action="store_true", default=True)
    parser.add_argument("--no-quantitative", dest="quantitative", action="store_false")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--deployment_name", type=str, default="gpt-35-turbo")
    parser.add_argument("--tag", type=str, default="sample")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--phi-values",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1],
        help="phi thresholds (default: 0.25 0.5 0.75 1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="session directory (default: experiments/.../phi_sensitivity/timestamp/)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--parallel-explainers",
        dest="parallel_explainers",
        action="store_true",
        help="run all phi hybrid explainers for a dataset concurrently (default)",
    )
    parser.add_argument(
        "--no-parallel-explainers",
        dest="parallel_explainers",
        action="store_false",
        help="run explainers for each dataset one after another",
    )
    parser.add_argument(
        "--parallel-datasets",
        dest="parallel_datasets",
        action="store_true",
        help="when multiple --datasets are given, process each dataset in parallel (default)",
    )
    parser.add_argument(
        "--no-parallel-datasets",
        dest="parallel_datasets",
        action="store_false",
        help="process datasets sequentially even when multiple are listed",
    )
    parser.add_argument(
        "--max-parallel-datasets",
        type=int,
        default=None,
        metavar="N",
        help="cap concurrent dataset workers (default: one worker per dataset)",
    )
    parser.add_argument(
        "--lemon-minun-lem-num-samples",
        type=int,
        default=100,
        metavar="N",
        help="Hybrid Lemon/Minun: LIME perturbation count (default: adaptive per masked n)",
    )
    parser.add_argument(
        "--lemon-minun-cf-max-evals",
        type=int,
        default=100,
        metavar="N",
        help="Hybrid Lemon/Minun: max CF predictor evaluations per iteration (default: 800)",
    )
    parser.add_argument(
        "--lemon-minun-max-iterations",
        type=int,
        default=10,
        metavar="N",
        help="Hybrid Lemon/Minun: max outer iterations (default: 10)",
    )
    parser.add_argument(
        "--lemon-minun-cf-method",
        type=str,
        choices=["greedy", "binary"],
        default="greedy",
        help="Hybrid Lemon/Minun: Minun search strategy (default: greedy)",
    )
    parser.add_argument(
        "--lemon-minun-no-lime-cap",
        dest="lemon_minun_lime_cap_samples",
        action="store_false",
        help="Hybrid Lemon/Minun: disable capping lem_num_samples to adaptive budget",
    )
    parser.set_defaults(parallel_explainers=True, parallel_datasets=True, lemon_minun_lime_cap_samples=True)

    args = parser.parse_args()

    run_phi_sensitivity(
        cache=args.cache,
        samples=args.samples,
        num_triangles=args.num_triangles,
        explanation_granularity=args.granularity,
        quantitative=args.quantitative,
        base_dir=args.base_dir,
        dataset_names=args.datasets,
        model_type=args.model_type,
        model_name=args.model_name,
        deployment_name=args.deployment_name,
        tag=args.tag,
        temperature=args.temperature,
        phi_values=args.phi_values,
        output_dir=args.output_dir,
        workers=args.workers,
        parallel_explainers=args.parallel_explainers,
        parallel_datasets=args.parallel_datasets,
        max_parallel_datasets=args.max_parallel_datasets,
        lemon_minun_lem_num_samples=args.lemon_minun_lem_num_samples,
        lemon_minun_cf_max_evals=args.lemon_minun_cf_max_evals,
        lemon_minun_max_iterations=args.lemon_minun_max_iterations,
        lemon_minun_cf_method=args.lemon_minun_cf_method,
        lemon_minun_lime_cap_samples=args.lemon_minun_lime_cap_samples,
    )
