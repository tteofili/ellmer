import itertools

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

samples = 4
num_triangles = 10
explanation_granularity = "attribute"
quantitative = False

pase = llm = ellmer.models.GenericEllmer(explanation_granularity=explanation_granularity,
                                         deployment_name="gpt-35-turbo",
                                         model_name="gpt-3.5-turbo", model_type="azure_openai", temperature=0.01,
                                         prompts={"pase": "ellmer/prompts/constrained7.txt"})
ptse = ellmer.models.GenericEllmer(explanation_granularity=explanation_granularity, deployment_name="gpt-35-turbo",
                                   model_name="gpt-3.5-turbo", model_type="azure_openai", temperature=0.01,
                                   prompts={"ptse": {"er": "ellmer/prompts/er.txt",
                                                     "saliency": "ellmer/prompts/er-saliency-lc.txt",
                                                     "cf": "ellmer/prompts/er-cf-lc.txt"}})
ptsew = ellmer.models.GenericEllmer(explanation_granularity=explanation_granularity, deployment_name="gpt-35-turbo",
                                    model_name="gpt-3.5-turbo", model_type="azure_openai", temperature=0.01,
                                    prompts={"ptse": {"er": "ellmer/prompts/er.txt", "why": "ellmer/prompts/er-why.txt",
                                                      "saliency": "ellmer/prompts/er-saliency-lc.txt",
                                                      "cf": "ellmer/prompts/er-cf-lc.txt"}})

# for each dataset in deepmatcher datasets
dataset_names = ['beers', 'abt_buy']
base_dir = '/Users/tteofili/dev/cheapER/datasets/'

for d in dataset_names:
    print(f'using dataset {d}')
    dataset_dir = '/'.join([base_dir, d])
    lsource = pd.read_csv(dataset_dir + '/tableA.csv')
    rsource = pd.read_csv(dataset_dir + '/tableB.csv')
    gt = pd.read_csv(dataset_dir + '/train.csv')
    valid = pd.read_csv(dataset_dir + '/valid.csv')
    test = pd.read_csv(dataset_dir + '/test.csv')
    test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'], [])

    certa = CertaExplainer(lsource, rsource)

    ellmers = {
        "pase": pase,
        "ptse": ptse,
        "ptsew": ptsew,
    }

    result_files = []
    all_llm_results = dict()
    for key, llm in ellmers.items():
        curr_llm_results = []
        start_time = time()

        # generate predictions and explanations
        test_data_df = test_df[:samples]
        for idx in range(len(test_data_df)):
            try:
                rand_row = test_df.iloc[[idx]]
                ltuple, rtuple = ellmer.utils.get_tuples(rand_row)
                answer_dictionary = llm.predict_and_explain(ltuple, rtuple)
                prediction = answer_dictionary['prediction']
                saliency = answer_dictionary['saliency']
                cfs = [answer_dictionary['cf']]
                curr_llm_results.append({"id": idx, "ltuple": ltuple, "rtuple": rtuple, "prediction": prediction,
                                         "label": rand_row['label'].values[0], "saliency": saliency, "cfs": cfs})
            except Exception:
                traceback.print_exc()
                print(f'error, waiting...')
                sleep(10)
                start_time += 10

        total_time = time() - start_time

        all_llm_results[key] = {"data": curr_llm_results, "total_time": total_time}

        expdir = f'./experiments/{d}/{datetime.now():%Y%m%d}/{datetime.now():%H:%M}/'
        os.makedirs(expdir, exist_ok=True)
        output_file_path = expdir + key + '_results.json'
        with open(output_file_path, 'w') as fout:
            json.dump(curr_llm_results, fout)

        result_files.append((key, output_file_path))
        print(f'{key} data generated in {total_time}s')

        if quantitative:
            # generate quantitative explainability metrics for each set of generated explanations:

            # generate saliency metrics
            faithfulness = ellmer.utils.get_faithfulness([key], llm.evaluation, expdir, test_data_df)
            print(f'{key} faithfulness({key}):{faithfulness}')

            # generate counterfactual metrics
            cf_metrics = ellmer.utils.get_cf_metrics([key], llm.predict, expdir, test_data_df)
            print(f'{key} cf_metrics({key}):{cf_metrics}')

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
        observations.to_csv(f'{d}_{p1_name}_{p2_name}.csv')
