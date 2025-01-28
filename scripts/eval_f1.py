from langchain.cache import InMemoryCache, SQLiteCache
import langchain
import pandas as pd
from certa.utils import merge_sources
from datetime import datetime
import os
import ellmer.models
import ellmer.metrics
from time import sleep
import math
import traceback
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score

def eval(cache, samples, base_dir, dataset_names, model_type,
         model_name, deployment_name, tag, temperature):
    if cache == "memory":
        langchain.llm_cache = InMemoryCache()
    elif cache == "sqlite":
        langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

    llm_config = {"model_type": model_type, "model_name": model_name, "deployment_name": deployment_name, "tag": tag}

    predict_only = ellmer.models.SelfExplainer(deployment_name=llm_config['deployment_name'], temperature=temperature,
                                      model_name=llm_config['model_name'], model_type=llm_config['model_type'],
                                      prompts={"ptse": {"er": "ellmer/prompts/er.txt"}})

    evals = []

    for d in dataset_names:
        print(f'using dataset {d}')
        dataset_dir = '/'.join([base_dir, d])
        lsource = pd.read_csv(dataset_dir + '/tableA.csv')
        rsource = pd.read_csv(dataset_dir + '/tableB.csv')
        test = pd.read_csv(dataset_dir + '/test.csv')
        test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'],
                                [])

        labels = []
        predictions = []
        # generate predictions
        test_data_df = test_df[:samples]
        ranged = range(len(test_data_df))
        predictions_df = pd.DataFrame()
        for idx in tqdm(ranged, disable=False):
            try:
                rand_row = test_df.iloc[[idx]]
                answer_dictionary = predict_only.predict(rand_row.drop(['ltable_id', 'rtable_id', 'label'], axis=1))
                label = rand_row['label'].values[0]
                if math.isnan(label):
                    continue
                prediction = answer_dictionary['match_score'].values[0]
                if math.isnan(prediction):
                    continue
                rand_row['prediction'] = prediction
                print(f'{prediction}-{label}')
                predictions.append(prediction)
                labels.append(label)
                predictions_df = pd.concat([predictions_df, pd.DataFrame(rand_row[['prediction', 'label', 'ltable_id', 'rtable_id']])], axis=0)
            except Exception:
                traceback.print_exc()
                print(f'error, waiting...')
                sleep(10)

        f1 = f1_score(y_true=labels, y_pred=predictions)
        print(f'f1 for {d}: {f1}')
        predictions_df.to_csv(f'predictions_{d}.csv', index=False)

        evals.append(f'{d}:{f1}')


    eval_df = pd.DataFrame(evals)
    eval_expdir = f'./experiments/{model_type}/{model_name}/{datetime.now():%Y%m%d}/{datetime.now():%H_%M}/'
    os.makedirs(eval_expdir, exist_ok=True)
    eval_df.to_csv(eval_expdir + "eval_f1.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run predictions.')
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
    parser.add_argument('--model_name', metavar='mn', type=str, help='model name/identifier',
                        default="gpt-3.5-turbo")
    parser.add_argument('--deployment_name', metavar='dn', type=str, help='deployment name',
                        default="gpt-35-turbo")
    parser.add_argument('--tag', metavar='tg', type=str, help='run tag', default="sample")
    parser.add_argument('--temperature', metavar='tp', type=float, help='LLM temperature', default=0.01)

    args = parser.parse_args()
    base_datadir = args.base_dir
    samples = args.samples
    temperature = args.temperature

    cache = args.cache
    dataset_names = args.datasets
    base_dir = args.base_dir

    model_type = args.model_type
    model_name = args.model_name
    deployment_name = args.deployment_name
    tag = args.tag

    eval(cache, samples, base_dir, dataset_names, model_type, model_name, deployment_name, tag, temperature)
