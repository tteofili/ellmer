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
from langchain.cache import InMemoryCache, SQLiteCache
from tqdm import tqdm

import ellmer.metrics
from ellmer.full_certa import FullCerta
from ellmer.hybrid import HybridCerta
from ellmer.post_hoc.certa_explain import LLMCertaExplainer
from ellmer.selfexplainer import SelfExplainer, ICLSelfExplainer
from ellmer.utils import merge_sources


def build_self_explainers(llm_config, temperature, granularity):
    """Build zeroshot, cot (no why), cot_with_why, and predict_only explainers."""
    common = dict(
        explanation_granularity=granularity,
        deployment_name=llm_config['deployment_name'],
        temperature=temperature,
        model_name=llm_config['model_name'],
        model_type=llm_config['model_type'],
    )
    return {
        "zeroshot": SelfExplainer(
            **common,
            prompts={"pase": "ellmer/prompts/constrained16.txt"},
        ),
        "cot": SelfExplainer(
            **common,
            prompts={"ptse": {
                "er": "ellmer/prompts/er.txt",
                "saliency": "ellmer/prompts/er-saliency-lc.txt",
                "cf": "ellmer/prompts/er-cf-lc.txt",
            }},
        ),
        "cot_with_why": SelfExplainer(
            **common,
            prompts={"ptse": {
                "er": "ellmer/prompts/er.txt",
                "why": "ellmer/prompts/er-why.txt",
                "saliency": "ellmer/prompts/er-saliency-lc.txt",
                "cf": "ellmer/prompts/er-cf-lc.txt",
            }},
        ),
        "predict_only": SelfExplainer(
            **common,
            prompts={"ptse": {"er": "ellmer/prompts/er.txt"}},
        ),
    }


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
        conversation = answer_dictionary.get('conversation', '')
        row_dict = {
            "id": idx, "ltuple": ltuple, "rtuple": rtuple, "prediction": prediction,
            "label": rand_row['label'].values[0], "saliency": saliency, "cfs": cfs,
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


def run_explainer(key, llm, test_df, samples, expdir, quantitative, dataset_name, workers=1):
    """Run one explainer on the test set, optionally compute metrics, write results. Returns (key, output_file_path, eval_row)."""
    print(f'{key} on {dataset_name}')
    start_time = time()
    test_data_df = test_df[:samples]
    indices = list(range(len(test_data_df)))

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
    total_llm_time = sum(r.get("llm_time", r["latency"]) for r in curr_llm_results) if n else 0.0
    local_time = max(0.0, total_time - total_llm_time)
    avg_latency_llm = total_llm_time / n if n else 0.0
    avg_latency_local = local_time / samples if samples else 0.0

    os.makedirs(expdir, exist_ok=True)
    count_tokens_samples = llm.count_tokens() / samples
    predictions_samples = llm.count_predictions() / samples
    faithfulness = 'nan'
    cf_metrics = {}

    if quantitative:
        faithfulness = ellmer.metrics.get_faithfulness(
            [key], llm.evaluation, expdir, test_data_df,
            results_by_name={key: {"data": curr_llm_results}})
        print(f'{key} faithfulness({key}):{faithfulness}')
        cf_metrics = ellmer.metrics.get_cf_metrics(
            [key], llm.predict, expdir, test_data_df,
            results_by_name={key: {"data": curr_llm_results}})
        print(f'{key} cf_metrics({key}):{cf_metrics}')

    metrics_results = {"faithfulness": faithfulness, "counterfactual_metrics": cf_metrics} if quantitative else {}
    llm_results = {
        "data": curr_llm_results,
        "total_time": total_time,
        "total_llm_time": total_llm_time,
        "total_local_time": local_time,
        "tokens": count_tokens_samples,
        "predictions": predictions_samples,
        "avg_latency": total_time / samples,
        "avg_latency_llm": avg_latency_llm,
        "avg_latency_local": avg_latency_local,
    }
    if quantitative:
        llm_results["metrics"] = metrics_results

    output_file_path = expdir + key + '_results.json'
    with open(output_file_path, 'w') as fout:
        json.dump(llm_results, fout)

    row_dict = {
        "total_time": total_time,
        "total_llm_time": total_llm_time,
        "total_local_time": local_time,
        "avg_latency_llm": avg_latency_llm,
        "avg_latency_local": avg_latency_local,
        "tokens": count_tokens_samples,
        "predictions": predictions_samples,
        "faithfulness": faithfulness,
        "model": key,
        "dataset": dataset_name,
    }
    for cfk, cfv in cf_metrics.items():
        row_dict[cfk] = cfv
    eval_row = pd.Series(row_dict)
    print(f'{key} data generated in {total_time}s (local: {local_time}s)')
    return key, output_file_path, eval_row


def eval(cache, samples, num_triangles, explanation_granularity, quantitative, base_dir, dataset_names, model_type,
         model_name, deployment_name, tag, temperature, run_id=None, session_dir=None, multi_run=False, workers=1,
         parallel_explainers=False):
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

        examples = []

        # generate predictions and explanations for few-shot examples
        few_shot_no = 1
        train_data_matching_df = train_df[train_df['label'] == 1][:few_shot_no]
        train_data_non_matching_df = train_df[train_df['label'] == 0][:few_shot_no]
        data_df = pd.concat([train_data_matching_df, train_data_non_matching_df])
        ranged = range(len(data_df))
        for idx in tqdm(ranged, disable=False):
            try:
                rand_row = data_df.iloc[[idx]]
                ltuple, rtuple = ellmer.utils.get_tuples(rand_row)
                answer_dictionary = cot_with_why.predict_and_explain(ltuple, rtuple)
                prediction = answer_dictionary['prediction']
                saliency_explanation = answer_dictionary['saliency']
                cf_explanation = answer_dictionary['cf']

                examples.append({"input": f"record1:\n{ltuple}\n record2:\n{rtuple}\n",
                                 "prediction": prediction, "saliency": saliency_explanation,
                                 "cf": cf_explanation})
            except Exception:
                traceback.print_exc()
                print(f'error while finding few shot samples')

        fs1 = ICLSelfExplainer(examples=examples,
                               explanation_granularity=explanation_granularity,
                               deployment_name=llm_config['deployment_name'],
                               temperature=temperature,
                               model_name=llm_config['model_name'],
                               model_type=llm_config['model_type'],
                               prompts={"fs": "ellmer/prompts/fs1.txt", "input":
                                   "record1:\n{ltuple}\n record2:\n{rtuple}\n"})

        ellmers = {
            "zs_" + llm_config['tag']: zeroshot,
            "cot_" + llm_config['tag']: cot_with_why,
            "fs_" + llm_config['tag']: fs1,
            "certa_" + llm_config['tag']: FullCerta(explanation_granularity, predict_only, certa, num_triangles),
            "hybrid_" + llm_config['tag']: HybridCerta(
                explanation_granularity, cot, certa,
                [zeroshot, cot, cot_with_why],
                num_triangles=num_triangles,
            ),
        }

        result_files = []
        if parallel_explainers:
            explainer_results = []
            with ThreadPoolExecutor(max_workers=len(ellmers)) as executor:
                futures = {
                    executor.submit(
                        run_explainer, key, llm, test_df, samples, expdir, quantitative, d, workers
                    ): key
                    for key, llm in ellmers.items()
                }
                for future in as_completed(futures):
                    _, output_file_path, eval_row = future.result()
                    key = futures[future]
                    explainer_results.append((key, output_file_path, eval_row))
            explainer_results.sort(key=lambda x: x[0])
            result_files = [(k, p) for k, p, _ in explainer_results]
            evals.extend(r for _, _, r in explainer_results)
        else:
            for key, llm in ellmers.items():
                _, output_file_path, eval_row = run_explainer(
                    key, llm, test_df, samples, expdir, quantitative, d, workers=workers)
                result_files.append((key, output_file_path))
                evals.append(eval_row)

        # generate concordance statistics for each pair of results
        for pair in itertools.combinations(result_files, 2):
            p1 = pair[0]
            p1_name = p1[0]
            p1_file = p1[1]
            p2 = pair[1]
            p2_name = p2[0]
            p2_file = p2[1]
            print(f'concordance statistics for {p1_name} - {p2_name}')
            observations = ellmer.metrics.get_concordance(p1_file, p2_file)
            print(f'{observations}')
            os.makedirs(obs_dir, exist_ok=True)
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
                        choices=['azure_openai', 'falcon', 'llama2', 'hf'], required=True)
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
                        help='generate quantitative explanation evaluation results')
    parser.add_argument('--model_name', metavar='mn', type=str, help='model name/identifier',
                        default="gpt-3.5-turbo")
    parser.add_argument('--deployment_name', metavar='dn', type=str, help='deployment name',
                        default="gpt-35-turbo")
    parser.add_argument('--tag', metavar='tg', type=str, help='run tag', default="sample")
    parser.add_argument('--temperature', metavar='tp', type=float, help='LLM temperature', default=0.01)
    parser.add_argument('--runs', metavar='n', type=int, default=1,
                        help='number of runs for significance testing (when > 1, uses run-scoped dirs and per-run or no cache)')
    parser.add_argument('--run_output_dir', metavar='o', type=str, default=None,
                        help='output directory for multi-run session (default: experiments/.../YYYYMMDD_HH_MM/)')
    parser.add_argument('--workers', type=int, default=1,
                        help='number of parallel workers for test-sample LLM calls (default: 1)')
    parser.add_argument('--parallel_explainers', action='store_true',
                        help='run the five explainers (zs, cot, fs, certa, hybrid) in parallel')

    args = parser.parse_args()
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
            eval(cache, samples, num_triangles, explanation_granularity, quantitative, base_dir, dataset_names,
                 model_type, model_name, deployment_name, tag, temperature,
                 run_id=run_id, session_dir=session_dir, multi_run=True, workers=workers,
                 parallel_explainers=parallel_explainers)
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
        eval(cache, samples, num_triangles, explanation_granularity, quantitative, base_dir, dataset_names,
             model_type, model_name, deployment_name, tag, temperature, workers=workers,
             parallel_explainers=parallel_explainers)
