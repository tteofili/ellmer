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

dataset_name = 'abt_buy'
datadir = '/Users/tteofili/dev/cheapER/datasets/' + dataset_name
lsource = pd.read_csv(datadir + '/tableA.csv')
rsource = pd.read_csv(datadir + '/tableB.csv')
gt = pd.read_csv(datadir + '/train.csv')
valid = pd.read_csv(datadir + '/valid.csv')
test = pd.read_csv(datadir + '/test.csv')
test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'], [])

samples = 50
explanation_granularity = 'attribute'
temperature = 0.01
params = {"temperature": temperature, "explanation_granularity": explanation_granularity, "samples": samples}
results = [params]

llm = ellmer.models.PredictAndSelfExplainER(explanation_granularity=explanation_granularity)

start_time = time()

for idx in range(len(test_df[:samples])):
    try:
        rand_row = test_df.iloc[[idx]]
        ltuple, rtuple = ellmer.utils.get_tuples(rand_row)
        answer = llm.er(str(ltuple), str(rtuple), temperature=temperature)
        try:
            answer = answer.split('```')[1]
            answer = json.loads(answer)
        except:
            pass
        results.append({"index": idx, "ltuple": ltuple,
                        "rtuple": rtuple, "answer": answer,
                        "label": rand_row['label'].values[0]})
        print(f'{ltuple}\n{rtuple}\n{answer}')
    except Exception:
        print(f'error, waiting...')
        sleep(10)
        start_time += 10

total_time = time() - start_time

results.append({"total_time": total_time})

expdir = f'./experiments/{dataset_name}/{datetime.now():%Y%m%d}/{datetime.now():%H:%M}/'
os.makedirs(expdir, exist_ok=True)
with open(expdir + 'pase_results.json', 'w') as fout:
    json.dump(results, fout)
