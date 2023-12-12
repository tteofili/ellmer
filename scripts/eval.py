import itertools
from langchain.cache import InMemoryCache, SQLiteCache
import langchain
import pandas as pd
from certa.utils import merge_sources
from certa.explain import CertaExplainer
from datetime import datetime
import os
import ellmer.models
import ellmer.metrics
from time import sleep, time
import json
import traceback
from tqdm import tqdm
import argparse


def eval(cache, samples, num_triangles, explanation_granularity, quantitative, base_dir, dataset_names, model_type,
         model_name, deployment_name, tag, temperature):
    if cache == "memory":
        langchain.llm_cache = InMemoryCache()
    elif cache == "sqlite":
        langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

    llm_config = {"model_type": model_type, "model_name": model_name, "deployment_name": deployment_name, "tag": tag}

    pase = ellmer.models.GenericEllmer(explanation_granularity=explanation_granularity,
                                       deployment_name=llm_config['deployment_name'], temperature=temperature,
                                       model_name=llm_config['model_name'], model_type=llm_config['model_type'],
                                       prompts={"pase": "ellmer/prompts/constrained14.txt"})

    ptse = ellmer.models.GenericEllmer(explanation_granularity=explanation_granularity,
                                       deployment_name=llm_config['deployment_name'], temperature=temperature,
                                       model_name=llm_config['model_name'], model_type=llm_config['model_type'],
                                       prompts={"ptse": {"er": "ellmer/prompts/er.txt",
                                                         "saliency": "ellmer/prompts/er-saliency-lc.txt",
                                                         "cf": "ellmer/prompts/er-cf-lc.txt"}})

    ptsew = ellmer.models.GenericEllmer(explanation_granularity=explanation_granularity,
                                        deployment_name=llm_config['deployment_name'], temperature=temperature,
                                        model_name=llm_config['model_name'], model_type=llm_config['model_type'],
                                        prompts={
                                            "ptse": {"er": "ellmer/prompts/er.txt",
                                                     "why": "ellmer/prompts/er-why.txt",
                                                     "saliency": "ellmer/prompts/er-saliency-lc.txt",
                                                     "cf": "ellmer/prompts/er-cf-lc.txt"}})

    ptn = ellmer.models.GenericEllmer(explanation_granularity=explanation_granularity,
                                      deployment_name=llm_config['deployment_name'], temperature=temperature,
                                      model_name=llm_config['model_name'], model_type=llm_config['model_type'],
                                      prompts={"ptse": {"er": "ellmer/prompts/er.txt"}})

    for d in dataset_names:
        print(f'using dataset {d}')
        dataset_dir = '/'.join([base_dir, d])
        lsource = pd.read_csv(dataset_dir + '/tableA.csv')
        rsource = pd.read_csv(dataset_dir + '/tableB.csv')
        test = pd.read_csv(dataset_dir + '/test.csv')
        test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'],
                                [])

        certa = CertaExplainer(lsource, rsource)

        ellmers = {
            "pase_" + llm_config['tag']: pase,
            "ptse_" + llm_config['tag']: ptse,
            "ptsew_" + llm_config['tag']: ptsew,
            "certa(ptse)_" + llm_config['tag']: ellmer.models.CertaEllmer(explanation_granularity, ptn, certa,
                                                                          num_triangles),
            "certa(pase)_" + llm_config['tag']: ellmer.models.CertaEllmer(explanation_granularity, pase, certa,
                                                                          num_triangles),
            "uncerta(pase)_" + llm_config['tag']: ellmer.models.UnCertaEllmer(explanation_granularity, pase, certa,
                                                                              [pase, ptse, ptsew],
                                                                              num_triangles=num_triangles),
            "uncerta(ptse)_" + llm_config['tag']: ellmer.models.UnCertaEllmer(explanation_granularity, ptse, certa,
                                                                              [pase, ptse, ptsew],
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
                    row_dict = {"id": idx, "ltuple": ltuple, "rtuple": rtuple, "prediction": prediction,
                              "label": rand_row['label'].values[0], "saliency": saliency, "cfs": cfs,
                              "latency": ptime}
                    if "filter_features" in answer_dictionary:
                        row_dict["filter_features"] = answer_dictionary["filter_features"]
                    curr_llm_results.append(row_dict)
                except Exception:
                    traceback.print_exc()
                    print(f'error, waiting...')
                    sleep(10)
                    start_time += 10

            total_time = time() - start_time

            expdir = f'./experiments/{explanation_granularity}/{d}/{datetime.now():%Y%m%d}/{datetime.now():%H_%M}/'
            os.makedirs(expdir, exist_ok=True)
            llm_results = {"data": curr_llm_results, "total_time": total_time}

            output_file_path = expdir + key + '_results.json'
            with open(output_file_path, 'w') as fout:
                json.dump(llm_results, fout)

            if quantitative:
                # generate quantitative explainability metrics for each set of generated explanations

                # generate saliency metrics
                faithfulness = ellmer.utils.get_faithfulness([key], llm.evaluation, expdir, test_data_df)
                print(f'{key} faithfulness({key}):{faithfulness}')

                # generate counterfactual metrics
                cf_metrics = ellmer.utils.get_cf_metrics([key], llm.predict, expdir, test_data_df)
                print(f'{key} cf_metrics({key}):{cf_metrics}')

                metrics_results = {"faithfulness": faithfulness, "counterfactual_metrics": cf_metrics}

                llm_results = {"data": curr_llm_results, "total_time": total_time, "metrics": metrics_results}

                output_file_path = expdir + key + '_results.json'
                with open(output_file_path, 'w') as fout:
                    json.dump(llm_results, fout)

            result_files.append((key, output_file_path))
            print(f'{key} data generated in {total_time}s')

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
            obs_dir = f'experiments/{explanation_granularity}/concordance/{d}//{datetime.now():%Y%m%d}/{datetime.now():%H_%M}'
            os.makedirs(obs_dir, exist_ok=True)
            observations.to_csv(f'{obs_dir}/{p1_name}_{p2_name}.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run saliency experiments.')
    parser.add_argument('--base_dir', metavar='b', type=str, help='the datasets base directory',
                        required=True)
    parser.add_argument('--model_type', metavar='m', type=str, help='the LLM type to evaluate',
                        choices=['azure_openai', 'falcon', 'llama2'], required=True)
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
    parser.add_argument('--quantitative', metavar='q', type=bool, default=True,
                        help='whether to generate quantitative explanation evaluation results')
    parser.add_argument('--model_name', metavar='mn', type=str, help='model name/identifier',
                        default="gpt-3.5-turbo")
    parser.add_argument('--deployment_name', metavar='dn', type=str, help='deployment name',
                        default="gpt-35-turbo")
    parser.add_argument('--tag', metavar='tg', type=str, help='run tag', default="run tag")
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
