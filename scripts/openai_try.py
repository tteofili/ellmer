import os
from datetime import datetime
import json
from langchain.cache import InMemoryCache, SQLiteCache
import langchain
import pandas as pd
from certa.utils import merge_sources
import ellmer.models
import ellmer.metrics
from time import sleep, time
import traceback
from tqdm import tqdm

cache = "sqlite"
samples = 2
explanation_granularity = "attribute"

# setup langchain cache
if cache == "memory":
    langchain.llm_cache = InMemoryCache()
elif cache == "sqlite":
    langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

llm_configs = [
    {"model_type": "azure_openai", "model_name": "gpt-3.5-turbo", "deployment_name": "gpt-35-turbo",
     "tag": "azure_openai"},
]
for llm_config in llm_configs:
    pase = llm = ellmer.models.GenericEllmer(explanation_granularity=explanation_granularity, verbose=True,
                                             deployment_name=llm_config['deployment_name'], temperature=0.01,
                                             model_name=llm_config['model_name'], model_type=llm_config['model_type'],
                                             prompts={"pase": "ellmer/prompts/constrained14.txt"})

    ptsew = ellmer.models.GenericEllmer(explanation_granularity=explanation_granularity, verbose=True,
                                        deployment_name=llm_config['deployment_name'], temperature=0.01,
                                        model_name=llm_config['model_name'], model_type=llm_config['model_type'],
                                        prompts={
                                            "ptse": {"er": "ellmer/prompts/er.txt", "why": "ellmer/prompts/er-why.txt",
                                                     "saliency": "ellmer/prompts/er-saliency-lc.txt",
                                                     "cf": "ellmer/prompts/er-cf-lc.txt"}})

    # for each dataset in deepmatcher datasets
    dataset_names = ['abt_buy', 'fodo_zaga', 'walmart_amazon']
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

        ellmers = {
            "ptsew_" + llm_config['tag']: ptsew,
            "pase_" + llm_config['tag']: pase,
        }

        result_files = []
        all_llm_results = dict()
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
                    print(f'ltuple:\n{ltuple}\nrtuple:\n{rtuple}')
                    answer_dictionary = llm.predict_and_explain(ltuple, rtuple)
                    print(answer_dictionary)
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

            expdir = f'./experiments/{d}/{datetime.now():%Y%m%d}/{datetime.now():%H_%M}/'
            os.makedirs(expdir, exist_ok=True)
            all_llm_results[key] = {"data": curr_llm_results, "total_time": total_time}

            output_file_path = expdir + key + '_results.json'
            with open(output_file_path, 'w') as fout:
                json.dump(all_llm_results, fout)