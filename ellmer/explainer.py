import traceback
import operator
from collections import Counter
from langchain.prompts import ChatPromptTemplate
from langchain import PromptTemplate, HuggingFaceHub, OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import re
import certa.utils
import math

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
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from time import time

openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")


class BaseLLMExplainer:

    pred_count = 0
    tokens = 0

    def predict_and_explain(self, ltuple, rtuple):
        prediction = self.predict_tuples(ltuple, rtuple)
        saliency, cf = self.explain(ltuple, rtuple, prediction)
        return {"prediction": prediction, "saliency_explanation": saliency, "counterfactual_explanation": cf}

    def predict_tuples(self, ltuple, rtuple):
        return False

    def predict(self, x, mojito=False):
        xcs = []
        ranged = range(len(x))
        for idx in tqdm(ranged, disable=False):
            xc = x.iloc[[idx]].copy()
            ltuple, rtuple = ellmer.utils.get_tuples(xc)
            matching = self.predict_tuples(ltuple, rtuple)
            if matching:
                xc['nomatch_score'] = 0
                xc['match_score'] = 1
            else:
                xc['nomatch_score'] = 1
                xc['match_score'] = 0
            if mojito:
                full_df = np.dstack((xc['nomatch_score'], xc['match_score'])).squeeze()
                xc = full_df
            xcs.append(xc)
            self.pred_count += len(x)
        return pd.concat(xcs, axis=0)

    def explain(self, ltuple, rtuple, prediction):
        return None, None

    def evaluation(self, data_df):
        predictions = self.predict(data_df)
        predictions = predictions['match_score'].astype(int).values
        labels = data_df['label'].astype(int).values
        return f1_score(y_true=labels, y_pred=predictions)

    def count_predictions(self):
        return self.pred_count

    def count_tokens(self):
        return self.tokens


def falcon_pipeline(model_id="vilsonrodrigues/falcon-7b-instruct-sharded", quantized: bool = False):
    if quantized:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_4bit = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        model = model_4bit
    else:
        model = model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    fpip = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=296,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=fpip)


def llama2_llm(verbose=True, quantized_model_path='ggml-model-q4_0.gguf', temperature=0.0, top_p=1, n_ctx=6000):
    llama2_llm = LlamaCpp(
        model_path=quantized_model_path,
        temperature=temperature,
        top_p=top_p,
        n_ctx=n_ctx,
        verbose=verbose,
    )
    return llama2_llm
