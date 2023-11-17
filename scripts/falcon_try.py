import traceback

from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import ellmer.utils
from tqdm import tqdm
from certa.utils import merge_sources
from ellmer.models import falcon_pipeline

prompts = {"pase": "ellmer/prompts/constrained13.txt"}
samples = 2
conversation = []
prompt = prompts['pase']
for prompt_message in ellmer.utils.read_prompt(prompt):
    conversation.append((prompt_message[0], prompt_message[1]))
question = "record1:\n{ltuple}\n record2:\n{rtuple}\n"
conversation.append(("user", question))
template = ChatPromptTemplate.from_messages(conversation)

dataset_names = ['beers', 'abt_buy', 'fodo_zaga', 'walmart_amazon']
base_dir = '/Users/tteofili/dev/cheapER/datasets/'

d = dataset_names[0]
print(f'using dataset {d}')
dataset_dir = '/'.join([base_dir, d])
lsource = pd.read_csv(dataset_dir + '/tableA.csv')
rsource = pd.read_csv(dataset_dir + '/tableB.csv')
gt = pd.read_csv(dataset_dir + '/train.csv')
valid = pd.read_csv(dataset_dir + '/valid.csv')
test = pd.read_csv(dataset_dir + '/test.csv')
test_df = merge_sources(test, 'ltable_', 'rtable_', lsource, rsource, ['label'], [])

result_files = []
all_llm_results = dict()
curr_llm_results = []

# generate predictions and explanations
test_data_df = test_df[:samples]
ranged = range(len(test_data_df))


falcon_llm = falcon_pipeline()
print("******")
print("******")
print("******")
for idx in tqdm(ranged, disable=False):
    try:
        rand_row = test_df.iloc[[idx]]
        ltuple, rtuple = ellmer.utils.get_tuples(rand_row)
        print('---')
        print(ltuple)
        print('---')
        print(rtuple)
        print('---')
        llm_chain = LLMChain(llm=falcon_llm, prompt=template)
        answer = llm_chain.predict(ltuple=str(ltuple), rtuple=str(rtuple), feature='attribute')
        print(answer)
    except:
        traceback.print_exc()
print("******")
print("******")
print("******")
