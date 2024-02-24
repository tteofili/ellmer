import traceback

from langchain.prompts import ChatPromptTemplate
from langchain import PromptTemplate, HuggingFaceHub, OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
import random
from certa.models.ermodel import ERModel
import numpy as np
import pandas as pd
import ellmer.utils
import openai
import os
import json
import ast
from sklearn.metrics import f1_score
from tqdm import tqdm
from certa.utils import merge_sources
from langchain.llms import HuggingFacePipeline
import ellmer.models

from huggingface_hub.hf_api import HfFolder

HfFolder.save_token(os.getenv('HUGGINGFACEHUB_API_TOKEN'))

# prompts = {"pase": "ellmer/prompts/constrained8.txt"}
prompts = {"pase": "ellmer/prompts/constrained12.txt"}
samples = 2
conversation = []
prompt = prompts['pase']
for prompt_message in ellmer.utils.read_prompt(prompt):
    conversation.append((prompt_message[0], prompt_message[1]))
question = "record1:\n{ltuple}\n record2:\n{rtuple}\n"
conversation.append(("user", question))
template = ChatPromptTemplate.from_messages(conversation)
# new_conversation = conversation + [('ai', 'Ok, I can generate the two tables and return them in the form of a JSON'), ('user', 'ok, return the json')]

dataset_names = ['beers', 'abt_buy', 'fodo_zaga', 'walmart_amazon']
base_dir = '/Users/tommasoteofili/dev/cheapER/datasets'

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

explanation_granularity = 'attribute'
temperature = 0.01
verbose = True

for repo_name in [
    'HuggingFaceH4/zephyr-7b-beta',
    'tiiuae/falcon-7b-instruct',
    "tiiuae/falcon-7b",
]:

    pase = ellmer.models.GenericEllmer(explanation_granularity=explanation_granularity,
                                       deployment_name='', temperature=temperature,
                                       model_name=repo_name, model_type='hf',
                                       prompts={"pase": "ellmer/prompts/constrained13.txt"})

    ptse = ellmer.models.GenericEllmer(explanation_granularity=explanation_granularity,
                                       deployment_name='', temperature=temperature,
                                       model_name=repo_name, model_type='hf',
                                       prompts={"ptse": {"er": "ellmer/prompts/er.txt",
                                                         "saliency": "ellmer/prompts/er-saliency-lc.txt",
                                                         "cf": "ellmer/prompts/er-cf-lc.txt"}})

    ptsew = ellmer.models.GenericEllmer(explanation_granularity=explanation_granularity,
                                        deployment_name='', temperature=temperature,
                                        model_name=repo_name, model_type='hf', verbose=verbose,
                                        prompts={
                                            "ptse": {"er": "ellmer/prompts/er.txt",
                                                     "why": "ellmer/prompts/er-why.txt",
                                                     "saliency": "ellmer/prompts/er-saliency-lc.txt",
                                                     "cf": "ellmer/prompts/er-cf-lc.txt"}})
    for idx in tqdm(ranged, disable=False):
        try:
            rand_row = test_df.iloc[[idx]]
            ltuple, rtuple = ellmer.utils.get_tuples(rand_row)

            '''llm = HuggingFacePipeline.from_model_id(
                model_id="bigscience/bloom-1b7",
                task="text-generation",
                model_kwargs={"temperature": 0, "max_length": 64},
            )
            prompt = ChatPromptTemplate.from_messages(new_conversation)

            answer = llm.predict_messages(messages=prompt.messages, ltuple=ltuple, rtuple=rtuple)'''
            print('---')
            print('---')
            print('---')
            print(ltuple)
            print('---')
            print(rtuple)
            print('---')
            answer_dictionary = pase.predict_and_explain(ltuple, rtuple)
            print('---')
            print(answer_dictionary)
            print('---')
            print('---')
            print('---')
        except:
            traceback.print_exc()
    print("******")
    print("******")
    print("******")
