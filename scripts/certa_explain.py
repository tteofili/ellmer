import pandas as pd
from certa.utils import merge_sources
from datetime import datetime
import os
import ellmer.models
import ellmer.utils
from time import sleep
from certa.explain import CertaExplainer
import json

lprefix = 'ltable_'
rprefix = 'rtable_'

dataset_name = 'beers'
datadir = '/Users/tteofili/dev/cheapER/datasets/' + dataset_name
lsource = pd.read_csv(datadir + '/tableA.csv')
rsource = pd.read_csv(datadir + '/tableB.csv')
gt = pd.read_csv(datadir + '/train.csv')
valid = pd.read_csv(datadir + '/valid.csv')
test = pd.read_csv(datadir + '/test.csv')
test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'], [])

results = []

certa_explainer = CertaExplainer(lsource, rsource)
llm = ellmer.models.AzureOpenAIERModel(temperature=0.01)


def predict_fn(x):
    return llm.predict(x)


for idx in range(len(test_df[:50])):
    try:
        rand_row = test_df.iloc[[idx]]
        ltuple, rtuple = ellmer.utils.get_tuples(rand_row)
        question = "record1:\n" + str(ltuple) + "\n record2:\n" + str(rtuple) + "\n"
        _, prediction = llm(question, er=True)

        l_id = int(rand_row['ltable_id'])
        ltuple_series = lsource.iloc[l_id]
        r_id = int(rand_row['rtable_id'])
        rtuple_series = rsource.iloc[r_id]

        saliency_df, cf_summary, counterfactual_examples, triangles, _ = certa_explainer.explain(ltuple_series,
                                                                                                 rtuple_series,
                                                                                                 predict_fn,
                                                                                                 token=False,
                                                                                                 num_triangles=10)
        answer = dict()
        answer['ltuple'] = ltuple
        answer['rtuple'] = rtuple
        answer['prediction'] = prediction
        answer['saliency_exp'] = saliency_df.to_dict()
        answer['cf_exp'] = counterfactual_examples.drop(
            ['alteredAttributes', 'droppedValues', 'copiedValues', 'triangle', 'attr_count'], axis=1).T.to_dict()
        answer['cf_summary'] = cf_summary.to_dict()
        answer['triangles'] = len(triangles)
        results.append(answer)
        print(f'{answer}')
    except Exception:
        print(f'error, waiting...')
        sleep(10)

expdir = f'./experiments/{datetime.now():%Y%m%d}/{datetime.now():%H:%M}/'
os.makedirs(expdir, exist_ok=True)
with open(expdir + 'certa_results.json', 'w') as fout:
    json.dump(results, fout)
