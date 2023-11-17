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

prompts = {"pase": "ellmer/prompts/constrained13.txt"}
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

for repo_name in [
    "tiiuae/falcon-7b",
    'tiiuae/falcon-7b-instruct',
    "Writer/camel-5b-hf",
    "google/flan-t5-xxl",
    "internlm/internlm-chat-7b",
    "Qwen/Qwen-7B"
    "databricks/dolly-v2-3b",
    #"tiiuae/falcon-40b",
    #'EleutherAI/gpt-neox-20b',
]:

    hub = HuggingFaceHub(repo_id=repo_name, model_kwargs={'temperature': 0.01, 'max_length': 128})
    print("******")
    print("******")
    print("******")
    print(repo_name)
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
            print(ltuple)
            print('---')
            print(rtuple)
            print('---')
            llm_chain = LLMChain(llm=hub, prompt=template)
            #answer1 = llm_chain.apply([{"ltuple": str(ltuple), "rtuple": str(rtuple), "feature": 'attribute'}])
            #print(answer1)
            answer2 = llm_chain.predict(ltuple=str(ltuple), rtuple=str(rtuple), feature='attribute')
            print(answer2)
        except:
            traceback.print_exc()
    print("******")
    print("******")
    print("******")
