from langchain.cache import InMemoryCache, SQLiteCache
import langchain
import pandas as pd
from certa.utils import merge_sources
from datetime import datetime
import os
import ellmer.metrics
from ellmer.hybrid import HybridCerta
from ellmer.selfexplainer import SelfExplainer
from ellmer.post_hoc.certa_explain import LLMCertaExplainer
import math
import traceback
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score
import operator

def compare(cache, samples, base_dir, dataset_names, model_type, model_name, deployment_name, tag, temperature, explanation_granularity):
    if cache == "memory":
        langchain.llm_cache = InMemoryCache()
    elif cache == "sqlite":
        langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

    llm_config = {"model_type": model_type, "model_name": model_name, "deployment_name": deployment_name, "tag": tag}

    zeroshot = SelfExplainer(explanation_granularity=explanation_granularity,
                                           deployment_name=llm_config['deployment_name'], temperature=temperature,
                                           model_name=llm_config['model_name'], model_type=llm_config['model_type'],
                                           prompts={"pase": "ellmer/prompts/constrained7.txt"})

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

    evals = []

    for d in dataset_names:
        print(f'using dataset {d}')
        dataset_dir = '/'.join([base_dir, d])
        lsource = pd.read_csv(dataset_dir + '/tableA.csv')
        rsource = pd.read_csv(dataset_dir + '/tableB.csv')
        test = pd.read_csv(dataset_dir + '/test.csv')
        test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'],
                                [])

        certa = LLMCertaExplainer(lsource, rsource)

        ellmer_explainer = HybridCerta(explanation_granularity, cot, certa,[zeroshot, cot, cot2],
                                                                                num_triangles=10)
        labels = []
        predictions = []
        # generate predictions
        test_data_df = test_df[:samples]
        ranged = range(len(test_data_df))
        predictions_df = pd.DataFrame()
        explanation_examples_size = 10
        bad_too = False
        examples = []
        for idx in tqdm(ranged, disable=False):
            try:
                rand_row = test_df.iloc[[idx]]
                label = rand_row['label'].values[0]
                if len(examples) < explanation_examples_size and not bad_too:
                    ltuple, rtuple = ellmer.utils.get_tuples(rand_row)
                    answer_dictionary = ellmer_explainer.predict_and_explain(ltuple, rtuple)
                    prediction = answer_dictionary['prediction']
                    saliency = answer_dictionary['saliency']
                    fix = ''
                    if label != prediction:
                        fix = 'less'
                        bad_too = True
                    focus_dict = sorted(saliency.items(), key=operator.itemgetter(1), reverse=True)
                    focus_list = [k for k, v in focus_dict if v[0] > 0]
                    examples.append(f"When predicting a pair like {ltuple} and {rtuple}, pay {fix} attention to the following {explanation_granularity}s: {focus_list}")
                else:
                    prediction_response = ellmer_explainer.predict(
                        rand_row.drop(['ltable_id', 'rtable_id', 'label'], axis=1))
                    prediction = prediction_response
                if math.isnan(label):
                    continue
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

        f1 = f1_score(y_true=labels, y_pred=predictions)
        print(f'baseline f1 for {d}: {f1}')
        predictions_df.to_csv(f'predictions_{d}.csv', index=False)
        evals.append(f'{d} baseline:{f1}')

        labels = []
        predictions = []
        explanations_enhanced_llm = SelfExplainer(explanation_granularity=explanation_granularity,
                                               deployment_name=llm_config['deployment_name'], temperature=temperature,
                                               model_name=llm_config['model_name'], model_type=llm_config['model_type'],
                                               prompts={"ptse": {"er": "ellmer/prompts/er.txt"}})

        extended_prompt = []
        for s in examples[-explanation_examples_size:]:
            extended_prompt.append(('user', s.replace("'",'').replace('{','[').replace('}',']')))
            extended_prompt.append(('ai', 'ok'))

        print(extended_prompt)

        for idx in tqdm(ranged, disable=False):
            try:
                rand_row = test_df.iloc[[idx]]
                label = rand_row['label'].values[0]
                ltuple, rtuple = ellmer.utils.get_tuples(rand_row)
                prediction_response = explanations_enhanced_llm.predict_tuples(ltuple, rtuple, append_conversation=extended_prompt)
                prediction = prediction_response
                if math.isnan(label):
                    continue
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
        f1 = f1_score(y_true=labels, y_pred=predictions)
        print(f'enhanced f1 for {d}: {f1}')
        evals.append(f'{d} enhanced:{f1}')

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
    parser.add_argument('--granularity', metavar='tk', type=str, default='attribute',
                        choices=['attribute', 'token'], help='explanation granularity')

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
    explanation_granularity = args.granularity

    compare(cache, samples, base_dir, dataset_names, model_type, model_name, deployment_name, tag, temperature, explanation_granularity)
