import pandas as pd
from certa.utils import merge_sources
from datetime import datetime
import os
import ellmer.models
import ellmer.utils
from time import sleep, time
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

explanation_granularity = 'attribute'
temperature = 0.01
why = False

params = {"temperature": temperature, "why": why, "explanation_granularity": explanation_granularity}

llm = ellmer.models.PredictThenSelfExplainER(explanation_granularity=explanation_granularity)

results = [params]


start_time = time()

for idx in range(len(test_df[:50])):
    try:
        rand_row = test_df.iloc[[idx]]
        ltuple, rtuple = ellmer.utils.get_tuples(rand_row)
        prediction = llm.er(str(ltuple), str(rtuple), temperature=temperature)
        answer = llm.explain(str(ltuple), str(rtuple), prediction['prediction'], temperature=temperature, why=why)
        try:
            saliency = answer['saliency_exp'].split('```')[1]
            saliency_dict = json.loads(saliency)
            answer['saliency_exp'] = saliency_dict
        except:
            pass
        try:
            cf = answer['cf_exp'].split('```')[1]
            cf_dict = json.loads(cf)
            answer['cf_exp'] = cf_dict
        except:
            pass
        answer['ltuple'] = ltuple
        answer['rtuple'] = rtuple
        answer['label'] = rand_row['label'].values[0]
        results.append(answer)
        print(f'{ltuple}\n{rtuple}\n{answer}')
    except Exception:
        print(f'error, waiting...')
        sleep(10)
        start_time += 10

total_time = time() - start_time

results.append({"total_time": total_time})

expdir = f'./experiments/{dataset_name}/{datetime.now():%Y%m%d}/{datetime.now():%H:%M}/'
os.makedirs(expdir, exist_ok=True)
with open(expdir + 'ptse_results.json', 'w') as fout:
    json.dump(results, fout)
