import pandas as pd
from certa.utils import merge_sources
from datetime import datetime
import os
import ellmer.models
import ellmer.utils
from time import sleep
import openai

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

llm = ellmer.models.PredictAndSelfExplainER(explanation_granularity='attribute')

for idx in range(len(test_df[:50])):
    try:
        rand_row = test_df.iloc[[idx]]
        ltuple, rtuple = ellmer.utils.get_tuples(rand_row)
        answer = llm.er(ltuple, rtuple, temperature=0.5)
        results.append((ltuple, rtuple, answer))
        print(f'{ltuple}\n{rtuple}\n{answer}')
        sleep(4)
    except openai.error.RateLimitError:
        print(f'rate-limit error, waiting...')
        sleep(10)

results_df = pd.DataFrame(columns=['left', 'right', 'answer'], data=results)
expdir = f'./experiments/{datetime.now():%Y%m%d}/{datetime.now():%H:%M}/'
os.makedirs(expdir, exist_ok=True)
results_df.to_csv(expdir + 'pase_results.csv')
