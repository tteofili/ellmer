import argparse
import itertools
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from time import sleep, time

import langchain
import pandas as pd
from langchain_core.caches import InMemoryCache
from langchain_community.cache import SQLiteCache
from tqdm import tqdm

import ellmer.metrics
from ellmer.metrics_json_rows import prediction_indices
from ellmer.full_certa import FullCerta
from ellmer.hybrid import HybridCerta
from ellmer.hybrid_lemon_minun import HybridLemonMinun
from ellmer.post_hoc.certa_explain import LLMCertaExplainer
from ellmer.selfexplainer import SelfExplainer, ICLSelfExplainer
from ellmer.utils import merge_sources


def _json_serializable_default(obj):
    """Convert numpy/pandas scalars to native Python for JSON serialization."""
    if hasattr(obj, 'item'):
        return obj.item()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def _nonempty_mapping(obj):
    return isinstance(obj, dict) and len(obj) > 0


def _explanation_presence_summary(rows):
    """Aggregate saliency/cf presence from result rows (self-explainers set flags or use non-empty dicts)."""
    if not rows:
        return {"n": 0, "saliency_count": 0, "cf_count": 0, "saliency_rate": 0.0, "cf_rate": 0.0}
    n = len(rows)
    sal_ok = 0
    cf_ok = 0
    for r in rows:
        if r.get("saliency_present") is not None:
            sal_ok += 1 if r["saliency_present"] else 0
        elif _nonempty_mapping(r.get("saliency")):
            sal_ok += 1
        if r.get("cf_present") is not None:
            cf_ok += 1 if r["cf_present"] else 0
        else:
            cfs = r.get("cfs") or []
            cf0 = cfs[0] if cfs else {}
            if _nonempty_mapping(cf0):
                cf_ok += 1
    return {
        "n": n,
        "saliency_count": sal_ok,
        "cf_count": cf_ok,
        "saliency_rate": sal_ok / n,
        "cf_rate": cf_ok / n,
    }


def _build_few_shot_example(cot_with_why, idx, data_df):
    """Build one few-shot example for ICL; returns (idx, example_dict) or (idx, None) on failure."""
    try:
        rand_row = data_df.iloc[[idx]]
        ltuple, rtuple = ellmer.utils.get_tuples(rand_row)
        answer_dictionary = cot_with_why.predict_and_explain(ltuple, rtuple)
        prediction = answer_dictionary['prediction']
        saliency_explanation = answer_dictionary['saliency']
        cf_explanation = answer_dictionary['cf']
        ex = {
            "input": f"record1:\n{ltuple}\n record2:\n{rtuple}\n",
            "prediction": prediction,
            "saliency": saliency_explanation,
            "cf": cf_explanation,
        }
        return idx, ex
    except Exception:
        traceback.print_exc()
        print('error while finding few shot samples')
        return idx, None


def _concordance_one_pair(pair):
    """Compute concordance for one explainer pair; used with ThreadPoolExecutor."""
    p1_name, p1_file = pair[0]
    p2_name, p2_file = pair[1]
    observations = ellmer.metrics.get_concordance(p1_file, p2_file)
    return p1_name, p2_name, observations


def build_self_explainers(llm_config, temperature, granularity):
    """Build zeroshot, cot (no why), cot_with_why, and predict_only explainers."""
    common = dict(
        explanation_granularity=granularity,
        deployment_name=llm_config['deployment_name'],
        temperature=temperature,
        model_name=llm_config['model_name'],
        model_type=llm_config['model_type'],
    )
    pase_prompt = (
        "ellmer/prompts/constrained16_attribute.txt"
        if granularity == "attribute"
        else "ellmer/prompts/constrained16.txt"
    )
    return {
        "zeroshot": SelfExplainer(
            **common,
            prompts={"pase": pase_prompt},
        ),
        "cot": SelfExplainer(
            **common,
            prompts={
                "ptse_staged": {
                    "er": "ellmer/prompts/cot_staged_er.txt",
                    "saliency": "ellmer/prompts/cot_staged_saliency.txt",
                    "cf": "ellmer/prompts/cot_staged_cf.txt",
                }
            },
        ),
        "cot_with_why": SelfExplainer(
            **common,
            prompts={
                "ptse_staged": {
                    "er": "ellmer/prompts/cot_staged_er.txt",
                    "why": "ellmer/prompts/cot_staged_why.txt",
                    "saliency": "ellmer/prompts/cot_staged_saliency.txt",
                    "cf": "ellmer/prompts/cot_staged_cf.txt",
                }
            },
        ),
        "predict_only": SelfExplainer(
            **common,
            prompts={"ptse": {"er": "ellmer/prompts/er.txt"}},
        ),
    }


def _nan_cf_row():
    return {"validity": float("nan"), "proximity": float("nan"), "sparsity": float("nan"), "diversity": float("nan")}


def _run_one_sample(idx, llm, test_df):
    """Run predict_and_explain for one test row. Returns (idx, row_dict) or (idx, None) on error."""
    try:
        rand_row = test_df.iloc[[idx]]
        ltuple, rtuple = ellmer.utils.get_tuples(rand_row)
        ptime = time()
        answer_dictionary = llm.predict_and_explain(ltuple, rtuple)
        ptime = time() - ptime
        prediction = answer_dictionary['prediction']
        saliency = answer_dictionary['saliency']
        cfs = [answer_dictionary['cf']]
        saliency_present = answer_dictionary.get("saliency_present")
        if saliency_present is None:
            saliency_present = _nonempty_mapping(saliency)
        cf_present = answer_dictionary.get("cf_present")
        if cf_present is None:
            cf_present = _nonempty_mapping(answer_dictionary.get("cf"))
        conversation = answer_dictionary.get('conversation', '')
        # JSON has no tuple type; convert list of (role, content) tuples to list of lists
        if conversation:
            conversation = [list(t) for t in conversation]
        row_dict = {
            "id": idx, "ltuple": ltuple, "rtuple": rtuple, "prediction": prediction,
            "label": rand_row['label'].values[0], "saliency": saliency, "cfs": cfs,
            "saliency_present": bool(saliency_present), "cf_present": bool(cf_present),
            "latency": ptime, "conversation": conversation,
        }
        if "llm_time" in answer_dictionary:
            row_dict["llm_time"] = answer_dictionary["llm_time"]
        if "filter_features" in answer_dictionary:
            row_dict["filter_features"] = answer_dictionary["filter_features"]
        return (idx, row_dict)
    except Exception:
        traceback.print_exc()
        print('error, waiting...')
        sleep(10)
        return (idx, None)


def run_explainer(
    key,
    llm,
    test_df,
    samples,
    expdir,
    quantitative,
    dataset_name,
    workers=1,
    split_metrics_by_correctness=False,
):
    """Run one explainer on the test set, optionally compute metrics, write results.

    Returns (key, output_file_path, eval_rows) where ``eval_rows`` is a list of one or more
    :class:`pandas.Series` (one row per ``prediction_split`` when ``split_metrics_by_correctness``).
    """
    print(f'{key} on {dataset_name}')
    test_data_df = test_df[:samples]
    indices = list(range(len(test_data_df)))

    tokens_before = llm.count_tokens()
    preds_before = llm.count_predictions()
    start_time = time()

    if workers <= 1:
        curr_llm_results = []
        for idx in tqdm(indices, disable=False):
            _, row_dict = _run_one_sample(idx, llm, test_df)
            if row_dict is not None:
                curr_llm_results.append(row_dict)
    else:
        curr_llm_results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_one_sample, idx, llm, test_df) for idx in indices]
            for f in tqdm(as_completed(futures), total=len(futures), disable=False):
                idx, row_dict = f.result()
                if row_dict is not None:
                    curr_llm_results.append(row_dict)
        curr_llm_results.sort(key=lambda r: r["id"])

    total_time = time() - start_time
    # Local time excludes remote LLM execution only; use for stable timing across runs.
    n = len(curr_llm_results)
    presence = _explanation_presence_summary(curr_llm_results)
    print(
        f"{key} explanation_presence: saliency {presence['saliency_count']}/{presence['n']} "
        f"({100.0 * presence['saliency_rate']:.1f}%), "
        f"cf {presence['cf_count']}/{presence['n']} ({100.0 * presence['cf_rate']:.1f}%)"
    )
    total_llm_time = sum(r.get("llm_time", r["latency"]) for r in curr_llm_results) if n else 0.0
    local_time = max(0.0, total_time - total_llm_time)
    avg_latency_llm = total_llm_time / n if n else 0.0
    avg_latency_local = local_time / samples if samples else 0.0

    os.makedirs(expdir, exist_ok=True)
    faithfulness = 'nan'
    cf_metrics = {}

    metrics_by_prediction_split = None
    if quantitative:
        faithfulness = ellmer.metrics.get_faithfulness(
            [key], llm.evaluation, expdir, test_data_df,
            results_by_name={key: {"data": curr_llm_results}})
        print(f'{key} faithfulness({key}):{faithfulness}')
        cf_metrics = ellmer.metrics.get_cf_metrics(
            [key], llm.predict, expdir, test_data_df,
            results_by_name={key: {"data": curr_llm_results}})
        print(f'{key} cf_metrics({key}):{cf_metrics}')

        if split_metrics_by_correctness:
            _, correct_idx, incorrect_idx = prediction_indices(curr_llm_results)
            metrics_by_prediction_split = {
                "all": {"faithfulness": faithfulness, "counterfactual_metrics": cf_metrics},
            }
            for split_name, idx in (("correct", correct_idx), ("incorrect", incorrect_idx)):
                if idx:
                    f_s = ellmer.metrics.get_faithfulness(
                        [key], llm.evaluation, expdir, test_data_df,
                        results_by_name={key: {"data": curr_llm_results}},
                        row_indices=idx,
                    )
                    cf_s = ellmer.metrics.get_cf_metrics(
                        [key], llm.predict, expdir, test_data_df,
                        results_by_name={key: {"data": curr_llm_results}},
                        row_indices=idx,
                    )
                else:
                    f_s = {key: float("nan")}
                    cf_s = {key: _nan_cf_row()}
                metrics_by_prediction_split[split_name] = {
                    "faithfulness": f_s,
                    "counterfactual_metrics": cf_s,
                }
                print(f'{key} faithfulness[{split_name}]({key}):{f_s}')
                print(f'{key} cf_metrics[{split_name}]({key}):{cf_s}')

    tokens_delta = llm.count_tokens() - tokens_before
    preds_delta = llm.count_predictions() - preds_before
    denom = float(samples) if samples else 0.0
    count_tokens_samples = (tokens_delta / denom) if denom else 0.0
    predictions_samples = (preds_delta / denom) if denom else 0.0

    metrics_results = {"faithfulness": faithfulness, "counterfactual_metrics": cf_metrics} if quantitative else {}
    llm_results = {
        "data": curr_llm_results,
        "total_time": total_time,
        "total_llm_time": total_llm_time,
        "total_local_time": local_time,
        "tokens": count_tokens_samples,
        "tokens_total_run": tokens_delta,
        "predictions": predictions_samples,
        "predictions_total_run": preds_delta,
        "avg_latency": total_time / samples if samples else 0.0,
        "avg_latency_llm": avg_latency_llm,
        "avg_latency_local": avg_latency_local,
        "explanation_presence": presence,
    }
    if quantitative:
        llm_results["metrics"] = metrics_results
        if metrics_by_prediction_split is not None:
            llm_results["metrics_by_prediction_split"] = metrics_by_prediction_split
    print(llm_results)

    output_file_path = expdir + key + '_results.json'
    with open(output_file_path, 'w') as fout:
        json.dump(llm_results, fout, default=_json_serializable_default)

    timing_fields = {
        "total_time": total_time,
        "total_llm_time": total_llm_time,
        "total_local_time": local_time,
        "avg_latency_llm": avg_latency_llm,
        "avg_latency_local": avg_latency_local,
        "tokens": count_tokens_samples,
        "predictions": predictions_samples,
    }
    eval_rows = []
    if quantitative and split_metrics_by_correctness and metrics_by_prediction_split is not None:
        for split_name in ("all", "correct", "incorrect"):
            block = metrics_by_prediction_split[split_name]
            faith_s = block["faithfulness"]
            cf_s = block["counterfactual_metrics"]
            row_dict = {
                **timing_fields,
                "faithfulness": faith_s,
                "model": key,
                "dataset": dataset_name,
                "prediction_split": split_name,
            }
            for cfk, cfv in cf_s.items():
                row_dict[cfk] = cfv
            eval_rows.append(pd.Series(row_dict))
    else:
        row_dict = {
            **timing_fields,
            "faithfulness": faithfulness,
            "model": key,
            "dataset": dataset_name,
        }
        for cfk, cfv in cf_metrics.items():
            row_dict[cfk] = cfv
        eval_rows.append(pd.Series(row_dict))
    print(f'{key} data generated in {total_time}s (local: {local_time}s)')
    return key, output_file_path, eval_rows


def eval(
    cache,
    samples,
    num_triangles,
    explanation_granularity,
    quantitative,
    base_dir,
    dataset_names,
    model_type,
    model_name,
    deployment_name,
    tag,
    temperature,
    run_id=None,
    session_dir=None,
    multi_run=False,
    workers=1,
    parallel_explainers=False,
    lemon_minun_lem_num_samples=None,
    lemon_minun_cf_max_evals=800,
    lemon_minun_max_iterations=10,
    lemon_minun_cf_method="greedy",
    lemon_minun_lime_cap_samples=True,
    split_metrics_by_correctness=False,
):
    if not multi_run:
        if cache == "memory":
            langchain.llm_cache = InMemoryCache()
        elif cache == "sqlite":
            langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

    llm_config = {"model_type": model_type, "model_name": model_name, "deployment_name": deployment_name, "tag": tag}
    explainers = build_self_explainers(llm_config, temperature, explanation_granularity)
    zeroshot = explainers["zeroshot"]
    cot = explainers["cot"]
    cot_with_why = explainers["cot_with_why"]
    predict_only = explainers["predict_only"]

    evals = []

    for d in dataset_names:
        if session_dir is not None and run_id is not None:
            expdir = f'{session_dir}/run_{run_id}/{d}/'
            obs_dir = f'{session_dir}/run_{run_id}/concordance/{d}/'
        else:
            expdir = f'./experiments/{model_type}/{model_name}/{explanation_granularity}/{d}/{datetime.now():%Y%m%d}/{datetime.now():%H_%M}/'
            obs_dir = f'experiments/{model_type}/{model_name}/{explanation_granularity}/concordance/{d}//{datetime.now():%Y%m%d}/{datetime.now():%H_%M}'

        print(f'using dataset {d}')
        dataset_dir = '/'.join([base_dir, d])
        lsource = pd.read_csv(dataset_dir + '/tableA.csv')
        rsource = pd.read_csv(dataset_dir + '/tableB.csv')
        test = pd.read_csv(dataset_dir + '/test.csv')
        train = pd.read_csv(dataset_dir + '/train.csv')

        test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'],
                                [], samples=samples)
        train_df = merge_sources(train, 'ltable_', 'rtable_', lsource, rsource, ['label'],
                                 [], samples=samples)

        certa = LLMCertaExplainer(lsource, rsource)

        # generate predictions and explanations for few-shot examples
        few_shot_no = 1
        train_data_matching_df = train_df[train_df['label'] == 1][:few_shot_no]
        train_data_non_matching_df = train_df[train_df['label'] == 0][:few_shot_no]
        data_df = pd.concat([train_data_matching_df, train_data_non_matching_df])
        n_fs = len(data_df)
        fs_workers = min(max(1, n_fs), (os.cpu_count() or 8) * 2)
        if fs_workers <= 1:
            examples = []
            for idx in tqdm(range(n_fs), disable=False):
                _, ex = _build_few_shot_example(cot_with_why, idx, data_df)
                if ex is not None:
                    examples.append(ex)
        else:
            examples_with_idx = []
            with ThreadPoolExecutor(max_workers=fs_workers) as fs_pool:
                futs = [fs_pool.submit(_build_few_shot_example, cot_with_why, idx, data_df) for idx in range(n_fs)]
                for fut in tqdm(as_completed(futs), total=len(futs), disable=False, desc='few-shot'):
                    idx, ex = fut.result()
                    if ex is not None:
                        examples_with_idx.append((idx, ex))
            examples_with_idx.sort(key=lambda t: t[0])
            examples = [ex for _, ex in examples_with_idx]

        fs1 = ICLSelfExplainer(examples=examples,
                               explanation_granularity=explanation_granularity,
                               deployment_name=llm_config['deployment_name'],
                               temperature=temperature,
                               model_name=llm_config['model_name'],
                               model_type=llm_config['model_type'],
                               prompts={"fs": "ellmer/prompts/fs1.txt", "input":
                                   "record1:\n{ltuple}\n record2:\n{rtuple}\n"})

        # Isolated usage counters per hybrid (shared ``cot`` would mix token/pred stats across explainers).
        zs_h = zeroshot.fork_for_stats()
        cot_h = cot.fork_for_stats()
        cwhy_h = cot_with_why.fork_for_stats()
        cot_lemon = cot.fork_for_stats()

        ellmers = {
            "zs_" + llm_config['tag']: zeroshot,
            "cot_" + llm_config['tag']: cot_with_why,
            "fs_" + llm_config['tag']: fs1,
            "certa_" + llm_config['tag']: FullCerta(explanation_granularity, predict_only, certa, num_triangles),
            "hybrid_" + llm_config['tag']: HybridCerta(
                explanation_granularity, cot_h, certa,
                [zs_h, cot_h, cwhy_h],
                num_triangles=num_triangles,
            ),
            "hybrid_lemon_minun_" + llm_config['tag']: HybridLemonMinun(
                explanation_granularity,
                cot_lemon,
                certa,
                num_triangles=num_triangles,
                lem_num_samples=lemon_minun_lem_num_samples,
                cf_max_evals=lemon_minun_cf_max_evals,
                max_iterations=lemon_minun_max_iterations,
                cf_method=lemon_minun_cf_method,
                lime_cap_samples=lemon_minun_lime_cap_samples,
            ),
        }

        result_files = []
        if parallel_explainers:
            explainer_results = []
            n_exp = len(ellmers)
            exp_pool_workers = max(1, n_exp)
            with ThreadPoolExecutor(max_workers=exp_pool_workers) as executor:
                futures = {
                    executor.submit(
                        run_explainer,
                        key,
                        llm,
                        test_df,
                        samples,
                        expdir,
                        quantitative,
                        d,
                        workers,
                        split_metrics_by_correctness,
                    ): key
                    for key, llm in ellmers.items()
                }
                for future in tqdm(as_completed(futures), total=len(futures), disable=False, desc='explainers'):
                    _, output_file_path, eval_rows = future.result()
                    key = futures[future]
                    explainer_results.append((key, output_file_path, eval_rows))
            explainer_results.sort(key=lambda x: x[0])
            result_files = [(k, p) for k, p, _ in explainer_results]
            for _, _, eval_rows in explainer_results:
                evals.extend(eval_rows)
        else:
            for key, llm in ellmers.items():
                _, output_file_path, eval_rows = run_explainer(
                    key,
                    llm,
                    test_df,
                    samples,
                    expdir,
                    quantitative,
                    d,
                    workers=workers,
                    split_metrics_by_correctness=split_metrics_by_correctness,
                )
                result_files.append((key, output_file_path))
                evals.extend(eval_rows)

        # generate concordance statistics for each pair of results (parallel when many pairs)
        pairs = list(itertools.combinations(result_files, 2))
        n_pairs = len(pairs)
        os.makedirs(obs_dir, exist_ok=True)
        c_workers = min(max(1, n_pairs), (os.cpu_count() or 8) * 4)
        if n_pairs == 0:
            pass
        elif c_workers <= 1:
            for pair in pairs:
                p1_name, p2_name, observations = _concordance_one_pair(pair)
                print(f'concordance statistics for {p1_name} - {p2_name}')
                print(f'{observations}')
                observations.to_csv(f'{obs_dir}/{p1_name}_{p2_name}.csv')
        else:
            with ThreadPoolExecutor(max_workers=c_workers) as c_pool:
                cfuts = {c_pool.submit(_concordance_one_pair, p): p for p in pairs}
                for fut in tqdm(as_completed(cfuts), total=n_pairs, disable=False, desc='concordance'):
                    p1_name, p2_name, observations = fut.result()
                    print(f'concordance statistics for {p1_name} - {p2_name}')
                    print(f'{observations}')
                    observations.to_csv(f'{obs_dir}/{p1_name}_{p2_name}.csv')

    eval_df = pd.DataFrame(evals)
    if session_dir is not None and run_id is not None:
        eval_expdir = f'{session_dir}/run_{run_id}/'
    else:
        eval_expdir = f'./experiments/{model_type}/{model_name}/{explanation_granularity}/{datetime.now():%Y%m%d}/{datetime.now():%H_%M}/'
    os.makedirs(eval_expdir, exist_ok=True)
    eval_df.to_csv(eval_expdir + "eval.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run saliency experiments.')
    parser.add_argument('--base_dir', metavar='b', type=str, help='the datasets base directory',
                        required=True)
    parser.add_argument('--model_type', metavar='m', type=str, help='the LLM type to evaluate',
                        choices=['azure_openai', 'falcon', 'llama2', 'hf', 'bedrock'], required=True)
    parser.add_argument('--datasets', metavar='d', type=str, nargs='+', required=True,
                        help='the dataset(s) to be used for the evaluation')
    parser.add_argument('--samples', metavar='s', type=int, default=-1,
                        help='no. of samples from the test set used for the evaluation')
    parser.add_argument('--cache', metavar='c', type=str, choices=['', 'sqlite', 'memory'], default='',
                        help='LLM prediction caching mechanism')
    parser.add_argument('--num_triangles', metavar='t', type=int, default=10,
                        help='no. of open triangles used to generate CERTA explanations')
    parser.add_argument('--granularity', metavar='tk', type=str, default='attribute',
                        choices=['attribute', 'token'], help='explanation granularity')
    parser.add_argument('--quantitative', action='store_true',
                        help='generate quantitative explanation evaluation results', default=True)
    parser.add_argument('--model_name', metavar='mn', type=str, help='model name/identifier',
                        default="gpt-3.5-turbo")
    parser.add_argument('--deployment_name', metavar='dn', type=str,
                        help='Azure deployment name; for Bedrock, optional inference profile ID (Azure defaults like gpt-35-turbo are ignored)',
                        default="gpt-35-turbo")
    parser.add_argument('--tag', metavar='tg', type=str, help='run tag', default="sample")
    parser.add_argument('--temperature', metavar='tp', type=float, help='LLM temperature', default=0.01)
    parser.add_argument('--runs', metavar='n', type=int, default=1,
                        help='number of runs for significance testing (when > 1, uses run-scoped dirs and per-run or no cache)')
    parser.add_argument('--run_output_dir', metavar='o', type=str, default=None,
                        help='output directory for multi-run session (default: experiments/.../YYYYMMDD_HH_MM/)')
    parser.add_argument('--workers', type=int, default=4,
                        help='parallel workers for test-sample predict_and_explain calls per explainer (default: 4)')
    parser.add_argument(
        '--parallel-explainers', '--parallel_explainers',
        dest='parallel_explainers',
        action='store_true',
        help='run all explainers (zs, cot, fs, certa, hybrid, hybrid_lemon_minun) concurrently (default)',
    )
    parser.add_argument(
        '--no-parallel-explainers',
        dest='parallel_explainers',
        action='store_false',
        help='run explainers one after another (lower peak memory / API load)',
    )
    parser.set_defaults(parallel_explainers=True)
    parser.add_argument(
        '--lemon-minun-lem-num-samples',
        type=int,
        default=None,
        metavar='N',
        help='Hybrid Lemon/Minun: LIME perturbation count (default: adaptive max(min(30*n,3000),500) per masked n)',
    )
    parser.add_argument(
        '--lemon-minun-cf-max-evals',
        type=int,
        default=800,
        metavar='N',
        help='Hybrid Lemon/Minun: max counterfactual predictor evaluations per iteration (default: 800)',
    )
    parser.add_argument(
        '--lemon-minun-max-iterations',
        type=int,
        default=10,
        metavar='N',
        help='Hybrid Lemon/Minun: max outer mask-refinement iterations (default: 10)',
    )
    parser.add_argument(
        '--lemon-minun-cf-method',
        type=str,
        choices=['greedy', 'binary'],
        default='greedy',
        help='Hybrid Lemon/Minun: Minun search strategy (default: greedy)',
    )
    parser.add_argument(
        '--lemon-minun-no-lime-cap',
        dest='lemon_minun_lime_cap_samples',
        action='store_false',
        help='Hybrid Lemon/Minun: do not cap lem_num_samples to adaptive budget for current mask size',
    )
    parser.set_defaults(lemon_minun_lime_cap_samples=True)
    parser.add_argument(
        "--split-metrics-by-correctness",
        action="store_true",
        help="With --quantitative, also compute faithfulness and CF metrics on correct vs incorrect "
        "predictions (extra LLM evaluation calls). Adds prediction_split to eval.csv and "
        "metrics_by_prediction_split to each result JSON.",
    )

    args = parser.parse_args()
    if getattr(args, "split_metrics_by_correctness", False) and not args.quantitative:
        parser.error("--split-metrics-by-correctness requires quantitative metrics (use --quantitative, default True)")
    base_datadir = args.base_dir
    samples = args.samples
    num_triangles = args.num_triangles
    temperature = args.temperature

    cache = args.cache
    explanation_granularity = args.granularity
    quantitative = args.quantitative
    dataset_names = args.datasets
    base_dir = args.base_dir

    model_type = args.model_type
    model_name = args.model_name
    deployment_name = args.deployment_name
    tag = args.tag
    runs = args.runs
    run_output_dir = args.run_output_dir
    workers = args.workers
    parallel_explainers = args.parallel_explainers

    if runs > 1:
        session_dir = run_output_dir or (
            f'./experiments/{model_type}/{model_name}/{explanation_granularity}/'
            f'{datetime.now():%Y%m%d}_{datetime.now():%H_%M}/'
        )
        os.makedirs(session_dir, exist_ok=True)
        all_evals = []
        for run_id in range(runs):
            if cache == "memory":
                langchain.llm_cache = InMemoryCache()
            elif cache == "sqlite":
                langchain.llm_cache = SQLiteCache(database_path=f".langchain_run_{run_id}.db")
            else:
                langchain.llm_cache = None
            print(f'--- Run {run_id + 1}/{runs} ---')
            eval(
                cache,
                samples,
                num_triangles,
                explanation_granularity,
                quantitative,
                base_dir,
                dataset_names,
                model_type,
                model_name,
                deployment_name,
                tag,
                temperature,
                run_id=run_id,
                session_dir=session_dir,
                multi_run=True,
                workers=workers,
                parallel_explainers=parallel_explainers,
                lemon_minun_lem_num_samples=args.lemon_minun_lem_num_samples,
                lemon_minun_cf_max_evals=args.lemon_minun_cf_max_evals,
                lemon_minun_max_iterations=args.lemon_minun_max_iterations,
                lemon_minun_cf_method=args.lemon_minun_cf_method,
                lemon_minun_lime_cap_samples=args.lemon_minun_lime_cap_samples,
                split_metrics_by_correctness=args.split_metrics_by_correctness,
            )
            run_eval_path = os.path.join(session_dir, f'run_{run_id}', 'eval.csv')
            if os.path.isfile(run_eval_path):
                run_eval_df = pd.read_csv(run_eval_path, index_col=0)
                run_eval_df['run_id'] = run_id
                all_evals.append(run_eval_df)
        if all_evals:
            eval_all_runs_df = pd.concat(all_evals, ignore_index=True)
            eval_all_runs_path = os.path.join(session_dir, 'eval_all_runs.csv')
            eval_all_runs_df.to_csv(eval_all_runs_path)
            print(f'Wrote {eval_all_runs_path}')
    else:
        eval(
            cache,
            samples,
            num_triangles,
            explanation_granularity,
            quantitative,
            base_dir,
            dataset_names,
            model_type,
            model_name,
            deployment_name,
            tag,
            temperature,
            workers=workers,
            parallel_explainers=parallel_explainers,
            lemon_minun_lem_num_samples=args.lemon_minun_lem_num_samples,
            lemon_minun_cf_max_evals=args.lemon_minun_cf_max_evals,
            lemon_minun_max_iterations=args.lemon_minun_max_iterations,
            lemon_minun_cf_method=args.lemon_minun_cf_method,
            lemon_minun_lime_cap_samples=args.lemon_minun_lime_cap_samples,
            split_metrics_by_correctness=args.split_metrics_by_correctness,
        )
