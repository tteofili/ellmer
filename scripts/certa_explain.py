import pandas as pd
from certa.utils import merge_sources
from datetime import datetime
import os
import ellmer.models
import ellmer.utils
from time import sleep, time
from certa.explain import CertaExplainer
import json
import traceback

lprefix = 'ltable_'
rprefix = 'rtable_'

dataset_name = 'abt_buy'
datadir = '/Users/tteofili/dev/cheapER/datasets/' + dataset_name
lsource = pd.read_csv(datadir + '/tableA.csv')
rsource = pd.read_csv(datadir + '/tableB.csv')
gt = pd.read_csv(datadir + '/train.csv')
valid = pd.read_csv(datadir + '/valid.csv')
test = pd.read_csv(datadir + '/test.csv')
test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'], [])

explanation_granularity = 'attribute'
temperature = 0.01
num_triangles = 10
model_type = "delegate"
hf_repo = None
verbose = False
max_length = 180
fake = False
samples = 50

delegate = ellmer.models.PredictThenSelfExplainER(explanation_granularity=explanation_granularity, why=True)

params = {"model_type": model_type, "hf_repo": hf_repo, "verbose": verbose, "max_length": max_length, "fake": fake,
          "temperature": temperature, "num_triangles": num_triangles, "samples": samples,
          "explanation_granularity": explanation_granularity, "delegate": str(delegate)}

results = [params]

certa_explainer = CertaExplainer(lsource, rsource)
llm = ellmer.models.LLMERModel(temperature=temperature, model_type=model_type, hf_repo=hf_repo, verbose=verbose,
                               max_length=max_length, fake=fake, delegate=delegate)


def predict_fn(x):
    return llm.predict(x)


start_time = time()

for idx in range(len(test_df[:samples])):
    try:
        rand_row = test_df.iloc[[idx]]
        ltuple, rtuple = ellmer.utils.get_tuples(rand_row)
        match = str(llm.predict(rand_row)['match_score'].values[0])

        l_id = int(rand_row[lprefix + 'id'])
        ltuple_series = lsource.iloc[l_id]
        r_id = int(rand_row[rprefix + 'id'])
        rtuple_series = rsource.iloc[r_id]

        saliency_df, cf_summary, cfs, tri, _ = certa_explainer.explain(ltuple_series, rtuple_series, predict_fn,
                                                                       token="token" == explanation_granularity,
                                                                       num_triangles=num_triangles)
        answer = dict()
        answer['id'] = idx
        answer['ltuple'] = ltuple
        answer['rtuple'] = rtuple
        answer['prediction'] = match
        answer['saliency_exp'] = saliency_df.to_dict('list')
        if len(cfs) > 0:
            answer['cf_exp'] = cfs.drop(
                ['alteredAttributes', 'droppedValues', 'copiedValues', 'triangle', 'attr_count'], axis=1).T.to_dict()
        answer['cf_summary'] = cf_summary.to_dict()
        answer['triangles'] = len(tri)
        results.append(answer)
        print(f'{answer}')
    except Exception:
        traceback.print_exc()
        print(f'error, waiting...')
        sleep(10)
        start_time += 10

total_time = time() - start_time

results.append({"total_time": total_time})

expdir = f'./experiments/{dataset_name}/{datetime.now():%Y%m%d}/{datetime.now():%H:%M}/'
os.makedirs(expdir, exist_ok=True)
with open(expdir + 'certa_results.json', 'w') as fout:
    json.dump(results, fout)
