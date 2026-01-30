import argparse
import itertools
import json
import os
import traceback
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


def eval(cache, samples, num_triangles, explanation_granularity, quantitative, base_dir, dataset_names, model_type,
         model_name, deployment_name, tag, temperature):
    if cache == "memory":
        langchain.llm_cache = InMemoryCache()
    elif cache == "sqlite":
        langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

    llm_config = {"model_type": model_type, "model_name": model_name, "deployment_name": deployment_name, "tag": tag}

    zeroshot = SelfExplainer(explanation_granularity=explanation_granularity,
                             deployment_name=llm_config['deployment_name'], temperature=temperature,
                             model_name=llm_config['model_name'], model_type=llm_config['model_type'],
                             prompts={"pase": "ellmer/prompts/constrained16.txt"})

    cot = SelfExplainer(explanation_granularity=explanation_granularity,
                        deployment_name=llm_config['deployment_name'], temperature=temperature,
                        model_name=llm_config['model_name'], model_type=llm_config['model_type'],
                        prompts={"ptse": {"er": "ellmer/prompts/er.txt",
                                          "saliency": "ellmer/prompts/er-saliency-lc.txt",
                                          "cf": "ellmer/prompts/er-cf-lc.txt"}})

    cot2 = SelfExplainer(explanation_granularity=explanation_granularity,
                         deployment_name=llm_config['deployment_name'], temperature=temperature,
                         model_name=llm_config['model_name'], model_type=llm_config['model_type'],
                         prompts={
                             "ptse": {"er": "ellmer/prompts/er.txt",
                                      "why": "ellmer/prompts/er-why.txt",
                                      "saliency": "ellmer/prompts/er-saliency-lc.txt",
                                      "cf": "ellmer/prompts/er-cf-lc.txt"}})

    predict_only = SelfExplainer(explanation_granularity=explanation_granularity,
                                 deployment_name=llm_config['deployment_name'], temperature=temperature,
                                 model_name=llm_config['model_name'], model_type=llm_config['model_type'],
                                 prompts={"ptse": {"er": "ellmer/prompts/er.txt"}})

    evals = []

    for d in dataset_names:
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

        # generate predictions and explanations
        few_shot_no = 1
        train_data_matching_df = train_df[train_df['label'] == 1][:few_shot_no]
        train_data_non_matching_df = train_df[train_df['label'] == 0][:few_shot_no]
        data_df = pd.concat([train_data_matching_df, train_data_non_matching_df])
        ranged = range(len(data_df))
        for idx in tqdm(ranged, disable=False):
            try:
                rand_row = data_df.iloc[[idx]]
                ltuple, rtuple = ellmer.utils.get_tuples(rand_row)
                answer_dictionary = cot.predict_and_explain(ltuple, rtuple)
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
            "cot_" + llm_config['tag']: cot2,
            "fs_" + llm_config['tag']: fs1,
            "certa_" + llm_config['tag']: FullCerta(explanation_granularity, predict_only, certa,
                                                    num_triangles),
            "hybrid_" + llm_config['tag']: HybridCerta(explanation_granularity, cot, certa,
                                                       [zeroshot, cot, cot2],
                                                       num_triangles=num_triangles),
        }

        result_files = []
        for key, llm in ellmers.items():
            print(f'{key} on {d}')
            curr_llm_results = []
            start_time = time()

            # generate predictions and explanations
            test_data_df = test_df[:samples]
            ranged = range(len(test_data_df))
            for idx in tqdm(ranged, disable=False):
                try:
                    rand_row = test_df.iloc[[idx]]
                    ltuple, rtuple = ellmer.utils.get_tuples(rand_row)
                    ptime = time()
                    answer_dictionary = llm.predict_and_explain(ltuple, rtuple)
                    ptime = time() - ptime
                    prediction = answer_dictionary['prediction']
                    saliency = answer_dictionary['saliency']
                    cfs = [answer_dictionary['cf']]
                    if 'conversation' in answer_dictionary:
                        conversation = answer_dictionary['conversation']
                    else:
                        conversation = ''
                    row_dict = {"id": idx, "ltuple": ltuple, "rtuple": rtuple, "prediction": prediction,
                                "label": rand_row['label'].values[0], "saliency": saliency, "cfs": cfs,
                                "latency": ptime, "conversation": conversation}
                    if "filter_features" in answer_dictionary:
                        row_dict["filter_features"] = answer_dictionary["filter_features"]
                    curr_llm_results.append(row_dict)
                except Exception:
                    traceback.print_exc()
                    print(f'error, waiting...')
                    sleep(10)
                    start_time += 10

            total_time = time() - start_time

            os.makedirs(expdir, exist_ok=True)
            count_tokens_samples = llm.count_tokens() / samples
            predictions_samples = llm.count_predictions() / samples
            llm_results = {"data": curr_llm_results, "total_time": total_time, "tokens": count_tokens_samples,
                           "predictions": predictions_samples,
                           "avg_latency": total_time / samples}

            output_file_path = expdir + key + '_results.json'
            with open(output_file_path, 'w') as fout:
                json.dump(llm_results, fout)

            faithfulness = 'nan'
            cf_metrics = {}
            count_tokens_samples = 'nan'
            predictions_samples = 'nan'

            if quantitative:
                # generate quantitative explainability metrics for each set of generated explanations

                # generate saliency metrics
                faithfulness = ellmer.metrics.get_faithfulness([key], llm.evaluation, expdir, test_data_df)
                print(f'{key} faithfulness({key}):{faithfulness}')

                # generate counterfactual metrics
                cf_metrics = ellmer.metrics.get_cf_metrics([key], llm.predict, expdir, test_data_df)
                print(f'{key} cf_metrics({key}):{cf_metrics}')

                metrics_results = {"faithfulness": faithfulness, "counterfactual_metrics": cf_metrics}

                llm_results = {"data": curr_llm_results, "total_time": total_time, "metrics": metrics_results,
                               "tokens": count_tokens_samples, "predictions": predictions_samples,
                               "avg_latency": total_time / samples}

                output_file_path = expdir + key + '_results.json'
                with open(output_file_path, 'w') as fout:
                    json.dump(llm_results, fout)

            result_files.append((key, output_file_path))
            print(f'{key} data generated in {total_time}s')

            row_dict = {"total_time": total_time, "tokens": count_tokens_samples, "predictions": predictions_samples,
                        "faithfulness": faithfulness, "model": key, "dataset": d}
            for cfk, cfv in cf_metrics.items():
                row_dict[cfk] = cfv
            eval_row = pd.Series(row_dict)
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
    parser.add_argument('--quantitative', metavar='q', type=bool, default=False,
                        help='whether to generate quantitative explanation evaluation results')
    parser.add_argument('--model_name', metavar='mn', type=str, help='model name/identifier',
                        default="gpt-3.5-turbo")
    parser.add_argument('--deployment_name', metavar='dn', type=str, help='deployment name',
                        default="gpt-35-turbo")
    parser.add_argument('--tag', metavar='tg', type=str, help='run tag', default="sample")
    parser.add_argument('--temperature', metavar='tp', type=float, help='LLM temperature', default=0.01)

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

    eval(cache, samples, num_triangles, explanation_granularity, quantitative, base_dir, dataset_names, model_type,
         model_name, deployment_name, tag, temperature)
