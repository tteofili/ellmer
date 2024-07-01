import traceback
import operator
from langchain.prompts import ChatPromptTemplate
from langchain import PromptTemplate, HuggingFaceHub, OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain_community.chat_models.huggingface import ChatHuggingFace
import re
import certa.utils
from lime.lime_text import LimeTextExplainer

import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from itertools import product, combinations

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

hf_models = ['EleutherAI/gpt-neox-20b', 'tiiuae/falcon-7b-instruct', "Writer/camel-5b-hf", "databricks/dolly-v2-3b",
             "google/flan-t5-xxl", "tiiuae/falcon-40b", "tiiuae/falcon-7b", "internlm/internlm-chat-7b", "Qwen/Qwen-7B",
             "meta-llama/Llama-2-7b-chat-hf"]

openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")


class BaseLLMExplainer:

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


class FullCerta(BaseLLMExplainer):

    def __init__(self, explanation_granularity, delegate, certa, num_triangles=10):
        self.explanation_granularity = explanation_granularity
        self.certa = certa
        self.num_triangles = num_triangles
        self.delegate = delegate
        self.predict_fn = lambda x: self.delegate.predict(x)

    def predict_and_explain(self, ltuple, rtuple, max_predict: int = -1, verbose: bool = False):
        pae = self.delegate.predict_tuples(ltuple, rtuple)
        prediction = pae
        saliency_explanation = None
        cf_explanation = None
        if pae is not None:
            ltuple_series = self.get_row(ltuple, self.certa.lsource, prefix="ltable_")
            rtuple_series = self.get_row(rtuple, self.certa.rsource, prefix="rtable_")
            saliency_df, cf_summary, cfs, _, _, _ = self.certa.explain(ltuple_series, rtuple_series, self.predict_fn,
                                                                      token="token" == self.explanation_granularity,
                                                                      num_triangles=self.num_triangles,
                                                                      max_predict=max_predict)
            saliency_explanation = saliency_df.to_dict('list')
            if len(cfs) > 0:
                cf_explanation = cfs.drop(
                    ['alteredAttributes', 'droppedValues', 'copiedValues', 'triangle', 'attr_count'],
                    axis=1).iloc[0].T.to_dict()
            else:
                cf_explanation = {}
        return {"prediction": prediction, "saliency": saliency_explanation, "cf": cf_explanation}

    def get_row(self, row, df, prefix=''):
        rc = dict()
        for k, v in row.items():
            new_k = k.replace(prefix, "")
            rc[new_k] = [v]
        result = df[df.drop(['id'], axis=1).isin(rc).all(axis=1)]
        if len(result) == 0:
            result = df.copy()
            for k, v in rc.items():
                result_new = result[result[k] == rc[k][0]]
                if len(result_new) == 1:
                    break
                if len(result_new) == 0:
                    continue
                else:
                    result = result_new
            if len(result) > 1:
                print(f'warning: found more than 1 item!({len(result)})')
        return result.iloc[0]

    def count_predictions(self):
        return self.delegate.count_predictions()

    def count_tokens(self):
        return self.delegate.count_tokens()


class HybridCerta(FullCerta):
    def __init__(self, explanation_granularity, pred_delegate, certa, ellmers, num_draws=1, num_triangles=10,
                 combine : str = 'freq', top_k: int = -1):
        self.explanation_granularity = explanation_granularity
        self.certa = certa
        self.num_triangles = num_triangles
        self.delegate = pred_delegate
        self.ellmers = ellmers
        self.predict_fn = lambda x: self.delegate.predict(x)
        self.num_draws = num_draws
        self.combine = combine
        self.top_k = top_k

    def predict_and_explain(self, ltuple, rtuple, max_predict: int = -1, verbose: bool = False):
        satisfied = False
        saliency_explanation = {}
        cf_explanation = {}
        filter_features = []
        tri = []

        if 'attribute' == self.explanation_granularity:
            no_features = len(ltuple) + len(rtuple)
        elif 'token' == self.explanation_granularity:
            no_features = len(str(ltuple).split(' ')) + len(str(rtuple).split(' '))
        else:
            raise ValueError('invalid explanation granularity')
        if self.top_k < 0:
            top_k = int(no_features * 0.25)
        else:
            top_k = self.top_k
        its = 0
        support_samples = None
        pae_dicts = []
        for e in self.ellmers:
            for _ in range(self.num_draws):
                pae_dict = e.predict_and_explain(ltuple, rtuple)['saliency']
                pae_dicts.append(pae_dict)

        pae = self.delegate.predict_tuples(ltuple, rtuple)
        prediction = pae
        ltuple_series = self.get_row(ltuple, self.certa.lsource, prefix="ltable_")
        rtuple_series = self.get_row(rtuple, self.certa.rsource, prefix="rtable_")
        num_triangles = self.num_triangles

        while not satisfied:
            if self.combine == 'freq':
                # get most frequent features from the self-explanations
                filter_features = []
                for se in pae_dicts:
                    fse = {k: v for k, v in se.items() if v > 0}
                    if len(fse) > 0:
                        sorted_attributes_dict = sorted(fse.items(), key=operator.itemgetter(1), reverse=True)[:top_k]
                        top_features = [f[0] for f in sorted_attributes_dict]
                        filter_features = filter_features + top_features
                fc = {}
                for f in filter_features:
                    if f in fc:
                        fc[f] = fc[f] + 1
                    else:
                        fc[f] = 1
                sorted_fc = sorted(fc.items(), key=operator.itemgetter(1), reverse=True)[:top_k]
                filter_features = [sfc[0] for sfc in sorted_fc]
            elif self.combine == 'union':
                # get all features from the self-explanations
                filter_features = []
                for se in pae_dicts:
                    sorted_attributes_dict = sorted(se.items(), key=operator.itemgetter(1), reverse=True)[:top_k]
                    top_features = [f[0] for f in sorted_attributes_dict]
                    for tf in top_features:
                        if tf not in filter_features:
                            filter_features.append(tf)
            elif self.combine == 'intersection':
                # get recurring features only from the self-explanations
                filter_features = set()
                for se in pae_dicts:
                    sorted_attributes_dict = sorted(se.items(), key=operator.itemgetter(1), reverse=True)[:top_k]
                    top_features = set([f[0] for f in sorted_attributes_dict])
                    if len(filter_features) == 0:
                        filter_features = top_features
                    filter_features = filter_features.intersection(top_features)
                filter_features = list(filter_features)
            else:
                raise ValueError("Unknown combination method")
            if len(filter_features) > 0:
                if 'token' == self.explanation_granularity:
                    token_attributes_filtered = []
                    for ff in filter_features:
                        if ff.startswith('ltable_'):
                            continue
                            #for token in ltuple[ff].split(' '):
                            #    token_attributes_filtered.append(ff+'__'+token)
                        if ff.startswith('rtable_'):
                            continue
                            #for token in rtuple[ff].split(' '):
                            #    token_attributes_filtered.append(ff+'__'+ token)
                        else:
                            for ic in ltuple.keys():
                                if ff in ltuple[ic].split(' '):
                                    token_attributes_filtered.append(ic+'__' + ff)
                            for ic in rtuple.keys():
                                if ff in rtuple[ic].split(' '):
                                    token_attributes_filtered.append(ic+'__' + ff)
                    filter_features = token_attributes_filtered
                # regenerate support_samples, when empty
                if support_samples is not None and len(support_samples) == 0:
                    support_samples = None
                    num_triangles *= 2
                saliency_df, cf_summary, cfs, tri, _, support_samples = self.certa.explain(ltuple_series, rtuple_series, self.predict_fn,
                                                                          token="token" == self.explanation_granularity,
                                                                          num_triangles=num_triangles,
                                                                          max_predict=max_predict,
                                                                          filter_features=filter_features, support_samples=support_samples)
                if len(saliency_df) > 0 and len(cf_summary) > 0:
                    saliency_explanation = saliency_df.to_dict('list')
                    if len(cfs) > 0:
                        cf_explanation = cfs.drop(
                            ['alteredAttributes', 'droppedValues', 'copiedValues', 'triangle', 'attr_count'],
                            axis=1).iloc[0].T.to_dict()
                    print(saliency_explanation)
                    aggregated_pn = 0
                    for sev in saliency_explanation.values():
                        if type(sev) == list:
                            aggregated_pn += sev[0]
                        else:
                            aggregated_pn += sev
                    if aggregated_pn >= 0.1 and max(cf_summary.to_dict().values()) > 0:
                        satisfied = True
            if 'attribute' == self.explanation_granularity:
                top_k += 1
            elif 'token' == self.explanation_granularity:
                top_k += 5
            its +=1
            if satisfied or top_k == no_features or its == 10:
                top_k -= 1
                break
        return {"prediction": prediction, "saliency": saliency_explanation, "cf": cf_explanation,
                "filter_features": filter_features, "self_explanations": pae_dicts, "top_k": top_k, "iterations": its,
                "triangles": len(tri)}

    def count_predictions(self):
        return self.delegate.count_predictions()

    def count_tokens(self):
        return self.delegate.count_tokens()


class SelfExplainer(BaseLLMExplainer):

    def __init__(self, model_type='azure_openai', temperature=0.01, max_length=512, fake=False, model_name="",
                 verbose=False, delegate=None, explanation_granularity="attribute", explainer_fn="self", prompts=None,
                 deployment_name="", model_version="2023-05-15"):
        self.fake = fake
        self.model_type = model_type
        if model_type == 'hf':
            llm = HuggingFaceHub(repo_id=model_name, task="text-generation",
                                 model_kwargs={'temperature': temperature, 'max_length': max_length,
                                               'max_new_tokens': 1024})
            self.llm = ChatHuggingFace(llm=llm, token=True)
        elif model_type == 'openai':
            self.llm = OpenAI(temperature=temperature, model_name=model_name)
        elif model_type == 'azure_openai':
            self.llm = AzureChatOpenAI(deployment_name=deployment_name, model_name=model_name, request_timeout=30,
                                       openai_api_version=model_version, temperature=temperature)
        elif model_type == 'delegate':
            self.llm = delegate
        elif model_type == 'falcon':
            self.llm = falcon_pipeline(model_id=model_name)
        elif model_type == 'llama2':
            self.llm = llama2_llm(verbose=verbose, temperature=temperature, quantized_model_path=model_name)
        self.verbose = verbose
        self.explanation_granularity = explanation_granularity
        if "self" == explainer_fn:
            self.explainer_fn = "self"
        else:
            self.explainer_fn = explainer_fn
        self.prompts = prompts
        self.pred_count = 0
        self.tokens = 0

    def predict_tuples(self, ltuple, rtuple):
        conversation = []
        if "ptse" in self.prompts:
            ptse_prompts = self.prompts["ptse"]
            er_prompt = ptse_prompts['er']
            for prompt_message in ellmer.utils.read_prompt(er_prompt):
                conversation.append((prompt_message[0], prompt_message[1]))
            question = "record1:\n{ltuple}\n record2:\n{rtuple}\n"
            conversation.append(("user", question))
            template = ChatPromptTemplate.from_messages(conversation)
            if self.model_type in ['falcon', 'llama2']:
                chain = LLMChain(llm=self.llm, prompt=template)
                er_answer = chain.predict(ltuple=ltuple, rtuple=rtuple)
            elif self.model_type in ['hf']:
                messages = template.format_messages(ltuple=ltuple, rtuple=rtuple)
                raw_content = self.llm.invoke(messages)
                try:
                    er_answer = raw_content.content.split('[/INST]')[-1]
                except:
                    er_answer = raw_content.content
            else:
                messages = template.format_messages(feature=self.explanation_granularity, ltuple=ltuple, rtuple=rtuple)
                answer = self.llm(messages)
                er_answer = answer.content

            # parse answer into prediction
            _, prediction = ellmer.utils.text_to_match(er_answer, self.llm)
            self.pred_count += 1
            self.tokens += sum([len(m[1].split(' ')) for m in conversation])  # input tokens
            self.tokens += len(er_answer.split(' '))  # output tokens
        else:
            prediction = self.predict_and_explain(ltuple, rtuple)['prediction']
        if self.verbose:
            print(ltuple)
            print(rtuple)
            print(prediction)
        return prediction

    def predict_and_explain(self, ltuple, rtuple):
        conversation = []
        if "pase" in self.prompts:
            if self.verbose:
                prep_t = time()
            prompt = self.prompts['pase']
            for prompt_message in ellmer.utils.read_prompt(prompt):
                conversation.append((prompt_message[0], prompt_message[1]))
            question = "record1:\n{ltuple}\n record2:\n{rtuple}\n"
            conversation.append(("user", question))
            template = ChatPromptTemplate.from_messages(conversation)
            if self.verbose:
                prep_t = time() - prep_t
                print(f'prep_time:{prep_t}')
            if self.model_type in ['falcon', 'llama2']:
                if self.verbose:
                    pre_pred_t = time()
                chain = LLMChain(llm=self.llm, prompt=template)
                if self.verbose:
                    pre_pred_t = time() - pre_pred_t
                    print(f'pre_prep_time:{pre_pred_t}')
                    pred_t = time()
                content = chain.predict(ltuple=ltuple, rtuple=rtuple, feature=self.explanation_granularity)
                if self.verbose:
                    pred_t = time() - pred_t
                    print(f'pred_time:{pred_t}')
                    print(f'content:{content}')
            elif self.model_type in ['hf']:
                messages = template.format_messages(feature=self.explanation_granularity, ltuple=ltuple, rtuple=rtuple)
                raw_content = self.llm.invoke(messages)
                try:
                    content = raw_content.content.split('[/INST]')[-1]
                except:
                    content = raw_content.content
            else:
                if self.verbose:
                    pre_pred_t = time()
                messages = template.format_messages(feature=self.explanation_granularity, ltuple=ltuple, rtuple=rtuple)
                if self.verbose:
                    pre_pred_t = time() - pre_pred_t
                    print(f'pre_prep_time:{pre_pred_t}')
                    pred_t = time()
                answer = self.llm(messages)
                if self.verbose:
                    pred_t = time() - pred_t
                    print(f'pred_time:{pred_t}')
                content = answer.content
            if self.verbose:
                print(content)
                parse_t = time()
            conversation.append(("assistant", content))
            prediction, saliency_explanation, cf_explanation = parse_pase_answer(content, self.llm)
            if self.verbose:
                parse_t = time() - parse_t
                print(f'parse_time:{parse_t}')
            if prediction is None:
                print(f'empty prediction!\nquestion{question}\nconversation{conversation}')
            self.pred_count += 1
            self.tokens += sum([len(m[1].split(' ')) for m in conversation])  # input tokens
            self.tokens += len(content.split(' '))  # output tokens
            try:
                saliency_explanation = dict([(x[0], x[1]['saliency']) for x in list(saliency_explanation.items())])
            except:
                try:
                    saliency_explanation = dict([(x[0], x[1]['saliency_score']) for x in list(saliency_explanation.items())])
                except:
                    pass
            return {"prediction": prediction, "saliency": saliency_explanation, "cf": cf_explanation,
                    "conversation": conversation}
        elif "ptse" in self.prompts:
            if self.verbose:
                prep_t = time()
            ptse_prompts = self.prompts["ptse"]
            er_prompt = ptse_prompts['er']
            for prompt_message in ellmer.utils.read_prompt(er_prompt):
                conversation.append((prompt_message[0], prompt_message[1]))
            question = "record1:\n{ltuple}\n record2:\n{rtuple}\n"
            conversation.append(("user", question))
            template = ChatPromptTemplate.from_messages(conversation)
            if self.verbose:
                prep_t = time() - prep_t
                print(f'er_prep_time:{prep_t}')
            if self.model_type in ['falcon', 'llama2']:
                if self.verbose:
                    pre_pred_t = time()
                chain = LLMChain(llm=self.llm, prompt=template)
                if self.verbose:
                    pre_pred_t = time() - pre_pred_t
                    print(f'er_pre_pred_time:{pre_pred_t}')
                    pred_t = time()
                er_answer = chain.predict(ltuple=ltuple, rtuple=rtuple)
                if self.verbose:
                    pred_t = time() - pred_t
                    print(f'er_pred_time:{pred_t}')
                    print(f'er_answer:{er_answer}')
            elif self.model_type in ['hf']:
                messages = template.format_messages(feature=self.explanation_granularity, ltuple=ltuple, rtuple=rtuple)
                er_answer = self.llm.invoke(messages).content
                try:
                    er_answer = er_answer.split('[/INST]')[2]
                except:
                    pass
            else:
                if self.verbose:
                    pre_pred_t = time()
                messages = template.format_messages(feature=self.explanation_granularity, ltuple=ltuple, rtuple=rtuple)
                if self.verbose:
                    pre_pred_t = time() - pre_pred_t
                    print(f'er_pre_pred_time:{pre_pred_t}')
                    pred_t = time()
                answer = self.llm(messages)
                er_answer = answer.content
                if self.verbose:
                    pred_t = time() - pred_t
                    print(f'er_pred_time:{pred_t}')
            if self.verbose:
                print(er_answer)
            if self.verbose:
                parse_t = time()
            # parse answer into prediction
            _, prediction = ellmer.utils.text_to_match(er_answer, self.llm)
            if self.verbose:
                parse_t = time() - parse_t
                print(f'er_parse_time:{parse_t}')

            self.pred_count += 1

            conversation.append(("assistant", er_answer))

            why = None
            saliency_explanation = None
            cf_explanation = None

            # get explanations
            if "why" in ptse_prompts:
                if self.verbose:
                    prep_t = time()
                for prompt_message in ellmer.utils.read_prompt(ptse_prompts["why"]):
                    conversation.append((prompt_message[0], prompt_message[1]))
                template = ChatPromptTemplate.from_messages(conversation)
                if self.verbose:
                    prep_t = time() - prep_t
                    print(f'why_prep_time:{prep_t}')
                if self.model_type in ['falcon', 'llama2']:
                    if self.verbose:
                        pre_pred_t = time()
                    chain = LLMChain(llm=self.llm, prompt=template)
                    if self.verbose:
                        pre_pred_t = time() - pre_pred_t
                        print(f'why_pre_pred_time:{pre_pred_t}')
                        pred_t = time()
                    why_answer = chain.predict(ltuple=ltuple, rtuple=rtuple, prediction=prediction)
                    if self.verbose:
                        pred_t = time() - pred_t
                        print(f'why_pred_time:{pred_t}')
                        print(f'why_answer:{why_answer}')
                elif self.model_type in ['hf']:
                    messages = template.format_messages(feature=self.explanation_granularity, ltuple=ltuple,
                                                        rtuple=rtuple)
                    why_answer = self.llm.invoke(messages).content
                    try:
                        why_answer = why_answer.split('[/INST]')[-1]
                    except:
                        pass
                else:
                    if self.verbose:
                        pre_pred_t = time()
                    messages = template.format_messages(feature=self.explanation_granularity, ltuple=ltuple,
                                                        rtuple=rtuple, prediction=prediction)
                    if self.verbose:
                        pre_pred_t = time() - pre_pred_t
                        print(f'why_pre_pred_time:{pre_pred_t}')
                        pred_t = time()
                    answer = self.llm(messages)
                    if self.verbose:
                        pred_t = time() - pred_t
                        print(f'why_pred_time:{pred_t}')
                    why_answer = answer.content
                if self.verbose:
                    print(why_answer)
                why = why_answer
                conversation.append(("assistant", why_answer))
                self.pred_count += 1

            # saliency explanation
            if "saliency" in ptse_prompts:
                if self.verbose:
                    prep_t = time()
                for prompt_message in ellmer.utils.read_prompt(ptse_prompts["saliency"]):
                    conversation.append((prompt_message[0], prompt_message[1]))
                template = ChatPromptTemplate.from_messages(conversation)
                if self.verbose:
                    prep_t = time() - prep_t
                    print(f'saliency_prep_time:{prep_t}')
                if self.model_type in ['falcon', 'llama2']:
                    if self.verbose:
                        pre_pred_t = time()
                    chain = LLMChain(llm=self.llm, prompt=template)
                    if self.verbose:
                        pre_pred_t = time() - pre_pred_t
                        print(f'saliency_pre_pred_time:{pre_pred_t}')
                        pred_t = time()
                    saliency_answer = chain.predict(ltuple=ltuple, rtuple=rtuple, prediction=prediction,
                                                    feature=self.explanation_granularity)
                    if self.verbose:
                        pred_t = time() - pred_t
                        print(f'saliency_pred_time:{pred_t}')
                        print(f'saliency_answer:{saliency_answer}')
                elif self.model_type in ['hf']:
                    messages = template.format_messages(ltuple=ltuple, rtuple=rtuple, prediction=prediction,
                                                        feature=self.explanation_granularity)
                    saliency_answer = self.llm.invoke(messages).content
                    try:
                        saliency_answer = saliency_answer.split('[/INST]')[-1]
                    except:
                        pass
                else:
                    if self.verbose:
                        pre_pred_t = time()
                    messages = template.format_messages(feature=self.explanation_granularity, ltuple=ltuple,
                                                        rtuple=rtuple,
                                                        prediction=prediction)
                    if self.verbose:
                        pre_pred_t = time() - pre_pred_t
                        print(f'saliency_pre_pred_time:{pre_pred_t}')
                        pred_t = time()
                    answer = self.llm(messages)
                    if self.verbose:
                        pred_t = time() - pred_t
                        print(f'saliency_pred_time:{pred_t}')
                    saliency_answer = answer.content
                if self.verbose:
                    print(saliency_answer)
                    parse_t = time()
                saliency_explanation = dict()
                try:
                    saliency_content = saliency_answer
                    try:
                        saliency_content = saliency_answer.split('```')[1].replace('`', '').replace('´', '')
                    except:
                        if '```' in saliency_answer:
                            start_index = saliency_answer.index('```')
                            saliency_content = saliency_answer[start_index + 3:saliency_answer.index('```', start_index + 3)]
                    saliency_dict = json.loads(saliency_content)
                    saliency_explanation = saliency_dict
                except:
                    try:
                        saliency = saliency_answer[saliency_answer.index("{"):saliency_answer.rfind("}") + 1]
                        saliency_dict = json.loads(saliency)
                        saliency_explanation = saliency_dict
                    except:
                        pass
                if self.verbose:
                    parse_t = time() - parse_t
                    print(f'saliency_parse_time:{parse_t}')
                conversation.append(("assistant", json.dumps(saliency_explanation).replace('{','').replace('}','')))
                self.pred_count += 1

            # counterfactual explanation
            if "cf" in ptse_prompts:
                if self.verbose:
                    prep_t = time()
                for prompt_message in ellmer.utils.read_prompt(ptse_prompts["cf"]):
                    conversation.append((prompt_message[0], prompt_message[1]))
                template = ChatPromptTemplate.from_messages(conversation)
                if self.verbose:
                    prep_t = time() - prep_t
                    print(f'cf_prep_time:{prep_t}')
                if self.model_type in ['falcon', 'llama2']:
                    if self.verbose:
                        pre_pred_t = time()
                    chain = LLMChain(llm=self.llm, prompt=template)
                    if self.verbose:
                        pre_pred_t = time() - pre_pred_t
                        print(f'cf_pre_pred_time:{pre_pred_t}')
                        pred_t = time()
                    cf_answer = chain.predict(ltuple=ltuple, rtuple=rtuple, prediction=prediction,
                                              feature=self.explanation_granularity)
                    if self.verbose:
                        pred_t = time() - pred_t
                        print(f'cf_pred_time:{pred_t}')
                        print(f'cf_answer:{cf_answer}')
                elif self.model_type in ['hf']:
                    messages = template.format_messages(feature=self.explanation_granularity, ltuple=ltuple,
                                                        rtuple=rtuple, prediction=prediction)
                    cf_answer = self.llm.invoke(messages).content
                    try:
                        cf_answer = cf_answer.split('[/INST]')[-1]
                    except:
                        pass
                else:
                    if self.verbose:
                        pre_pred_t = time()
                    messages = template.format_messages(feature=self.explanation_granularity, ltuple=ltuple,
                                                        rtuple=rtuple,
                                                        prediction=prediction)
                    if self.verbose:
                        pre_pred_t = time() - pre_pred_t
                        print(f'cf_pre_pred_time:{pre_pred_t}')
                        pred_t = time()
                    answer = self.llm(messages)
                    if self.verbose:
                        pred_t = time() - pred_t
                        print(f'cf_pred_time:{pred_t}')
                    cf_answer = answer.content
                if self.verbose:
                    print(cf_answer)
                    parse_t = time()
                cf_explanation = dict()
                try:
                    cf_answer_content = cf_answer.replace('`', '').replace('´', '')
                    if '```' in cf_answer_content:
                        cf_answer_json = cf_answer_content.split('```')[1]
                    elif cf_answer_content.startswith("{"):
                        cf_answer_json = cf_answer_content
                    elif "{" in cf_answer_content and "}" in cf_answer_content:
                        cf_answer_json = cf_answer_content[
                                         cf_answer_content.index("{"):cf_answer_content.rfind("}") + 1]
                        # cf_answer_json = ''.join(cf_answer_content.split("{")[1].split("}")[0])
                    else:
                        cf_answer_json = cf_answer_content
                    cf_dict = ast.literal_eval(cf_answer_json)
                    keys = cf_dict.keys()
                    if "record_after" in keys:
                        cf_explanation = cf_dict["record_after"]
                        if list(cf_explanation.keys())[0].startswith('rtable_'):
                            cf_explanation = cf_explanation | ast.literal_eval(ltuple)
                        else:
                            cf_explanation = cf_explanation | ast.literal_eval(rtuple)
                    elif "counterfactual_record" in keys:
                        cf_explanation = cf_dict["counterfactual_record"]
                    elif "counterfactual" in keys:
                        cf_explanation = cf_dict['counterfactual']
                    elif "record1" in keys and "record2" in keys:
                        for k in cf_dict['record1'].keys():
                            if not k.startswith('ltable_'):
                                cf_dict['record1']['ltable_' + k] = cf_dict['record1'][k]
                                cf_dict['record1'].pop(k)

                        for k in cf_dict['record2'].keys():
                            if not k.startswith('rtable_'):
                                cf_dict['record2']['rtable_' + k] = cf_dict['record2'][k]
                                cf_dict['record2'].pop(k)
                        cf_explanation = cf_dict['record1'] | cf_dict['record2']
                    else:
                        cf_explanation = cf_dict
                except:
                    pass
                if self.verbose:
                    parse_t = time() - parse_t
                    print(f'cf_parse_time:{parse_t}')
                conversation.append(("assistant", str(cf_explanation)))
                self.pred_count += 1

            self.tokens += sum([len(m[1].split(' ')) for m in conversation])
            try:
                saliency_explanation = dict([(x[0], x[1]['saliency']) for x in list(saliency_explanation.items())])
            except:
                try:
                    saliency_explanation = dict([(x[0], x[1]['saliency_score']) for x in list(saliency_explanation.items())])
                except:
                    pass
            return {"prediction": prediction, "why": why, "saliency": saliency_explanation, "cf": cf_explanation,
                    "conversation": conversation}


def parse_pase_answer(answer, llm):
    matching = 0
    saliency = dict()
    cf = dict()

    try:
        # find the json content
        split = answer.split('```')
        if len(split) > 1:
            answer = split[1]
            if answer.startswith('json'):
                answer = answer[4:]
        elif "\n\n{" in answer and "}\n\n" in answer:
            answer = '{' + ''.join(answer.split("\n\n{")[1].split("}\n\n")[0]) + '}'

        # decode the json content
        try:
            answer = answer.replace('´', '').replace('`', '')
            answer = json.loads(answer)
            if 'answers' in answer:
                answer = answer['answers']
            if "matching" in answer.keys():
                prediction = answer['matching']
            elif "matching_prediction" in answer.keys():
                prediction = answer['matching_prediction']
            elif "match" in answer.keys():
                prediction = answer['match']
            elif "prediction" in answer.keys():
                prediction = answer['prediction']
            elif 'same_entity' in answer.keys():
                prediction = answer['same_entity']
            elif 'entity_resolution' in answer.keys():
                prediction = answer['entity_resolution']
            elif '1' in answer.keys():
                prediction = answer['1']
            else:
                print(f"cannot find 'matching' key in {answer}")
                prediction = None
            if prediction:
                matching = 1
            elif not prediction:
                matching = 0
            else:
                _, matching = ellmer.utils.text_to_match(prediction, llm.__call__)
            try:
                if "saliency_explanation" in answer.keys():
                    saliency = answer['saliency_explanation']
                elif "saliency_explanation_table" in answer.keys():
                    saliency = answer['saliency_explanation_table']
                elif "2" in answer.keys():
                    saliency = answer['2']
            except:
                pass
            try:
                if "counterfactual_explanation" in answer.keys():
                    cf = answer['counterfactual_explanation']
                elif "counterfactual_explanation_table" in answer.keys():
                    cf = answer['counterfactual_explanation_table']
                elif "attribute_counterfactual" in answer.keys():
                    cf = answer['attribute_counterfactual']
                elif "token_counterfactual" in answer.keys():
                    cf = answer['token_counterfactual']
                elif "3" in answer.keys():
                    cf = answer['3']
            except:
                pass
        except Exception as d:
            print(f"{d}: cannot decode json: {answer}")
            pass
    except Exception as e:
        print(f"{e}: cannot find json in: {answer}")
        pass
    if matching is None:
        _, matching = ellmer.utils.text_to_match(answer, llm.__call__)
    return matching, saliency, cf


# deprecated
class Certa(BaseLLMExplainer):

    def __init__(self, explanation_granularity, delegate, certa, num_triangles=10):
        self.explanation_granularity = explanation_granularity
        self.llm = delegate
        self.predict_fn = lambda x: self.llm.predict(x)
        self.certa = certa
        self.num_triangles = num_triangles

    def get_row(self, row, df, prefix=''):
        rc = dict()
        for k, v in row.items():
            new_k = k.replace(prefix, "")
            rc[new_k] = [v]
        result = df[df.drop(['id'], axis=1).isin(rc).all(axis=1)]
        return result.iloc[0]

    def predict_and_explain(self, ltuple, rtuple):
        matching, _, _ = self.llm.predict_and_explain(ltuple, rtuple)

        ltuple_series = self.get_row(ltuple, self.certa.lsource, prefix="ltable_")
        rtuple_series = self.get_row(rtuple, self.certa.rsource, prefix="rtable_")

        saliency_df, cf_summary, cfs, _, _, _ = self.certa.explain(ltuple_series, rtuple_series, self.predict_fn,
                                                                  token="token" == self.explanation_granularity,
                                                                  num_triangles=self.num_triangles, max_predict=100)
        saliency = saliency_df.to_dict('list')
        if len(cfs) > 0:
            counterfactuals = [cfs.drop(
                ['alteredAttributes', 'droppedValues', 'copiedValues', 'triangle', 'attr_count'], axis=1).T.to_dict()]
        else:
            counterfactuals = [{}]
        return matching, saliency, counterfactuals


# deprecated
class PASE(BaseLLMExplainer):

    def __init__(self, explanation_granularity, temperature):
        self.llm = PredictAndSelfExplainER(explanation_granularity=explanation_granularity,
                                           temperature=temperature)

    def predict_and_explain(self, ltuple, rtuple):
        answer = self.llm.er(str(ltuple), str(rtuple))
        matching = 0
        saliency = dict()
        cf = dict()

        try:
            split = answer.split('```')
            if len(split) > 1:
                answer = split[1]
                answer = json.loads(answer)
                if "matching" in answer.keys():
                    prediction = answer['matching']
                elif "matching_prediction" in answer.keys():
                    prediction = answer['matching_prediction']
                else:
                    raise Exception("cannot find 'matching' key")
                if prediction:
                    matching = 1
                elif not prediction:
                    matching = 0
                else:
                    _, matching = ellmer.utils.text_to_match(prediction, self.llm.__call__)
                try:
                    saliency = answer['saliency_explanation']
                except:
                    pass
                try:
                    cf = answer['counterfactual_explanation']
                except:
                    pass
            else:
                _, matching = ellmer.utils.text_to_match(answer, self.llm.__call__)
        except:
            traceback.print_exc()
            pass

        return {"prediction": matching, "saliency_explanation": saliency, "cf": cf}


# deprecated
class PTSE(BaseLLMExplainer):

    def __init__(self, explanation_granularity, why, temperature):
        self.llm = PredictThenSelfExplainER(explanation_granularity=explanation_granularity, why=why,
                                            temperature=temperature)

    def predict_and_explain(self, ltuple, rtuple):
        prediction_answer = self.llm.er(str(ltuple), str(rtuple))
        prediction = prediction_answer['prediction']
        if type(prediction) == dict:
            prediction = str(prediction)
        _, matching = ellmer.utils.text_to_match(prediction, self.llm.__call__)
        explain_answer = self.llm.explain(str(ltuple), str(rtuple), prediction)

        saliency = dict()
        try:
            saliency = explain_answer['saliency_exp'].split('```')[1]
            saliency_dict = json.loads(saliency)
            saliency = saliency_dict
        except:
            pass
        cf = dict()
        try:
            cf_answer = explain_answer['cf_exp'].split('```')[1]
            cf_dict = json.loads(cf_answer)
            keys = cf_dict.keys()
            if "record_after" in keys:
                cf = cf_dict["record_after"]
            elif "counterfactual_record" in keys:
                cf = cf_dict["counterfactual_record"]
            elif "counterfactual" in keys:
                cf = cf_dict['counterfactual']
            elif "record1" in keys and "record2" in keys:
                for k in cf_dict['record1'].keys():
                    if not k.startswith('ltable_'):
                        cf_dict['record1']['ltable_' + k] = cf_dict['record1'][k]
                        cf_dict['record1'].pop(k)

                for k in cf_dict['record2'].keys():
                    if not k.startswith('rtable_'):
                        cf_dict['record2']['rtable_' + k] = cf_dict['record2'][k]
                        cf_dict['record2'].pop(k)
                cf = cf_dict['record1'] | cf_dict['record2']
        except:
            pass
        return matching, saliency, [cf]


# deprecated
class LLMERModel(ERModel):
    count = 0
    idks = 0
    summarized = 0
    llm = None
    fake = False
    verbose = False

    def __init__(self, model_type='azure_openai', temperature=0.01, max_length=512, fake=False, hf_repo=hf_models[0],
                 verbose=False, delegate=None):
        template = "given the record:\n{ltuple}\n and the record:\n{rtuple}\n do they refer to the same entity in the real world?\nreply yes or no"
        # ellmer.utils.read_prompt("ellmer/prompts/er2.txt")
        self.prompt = PromptTemplate(
            input_variables=["ltuple", "rtuple"],
            template=template,
        )
        self.fake = fake
        if model_type == 'hf':
            self.llm = HuggingFaceHub(repo_id=hf_repo,
                                      model_kwargs={'temperature': temperature, 'max_length': max_length})
        elif model_type == 'openai':
            self.llm = OpenAI(temperature=temperature, model_name='gpt-3.5-turbo')
        elif model_type == 'azure_openai':
            self.llm = AzureOpenAIERModel(temperature=temperature)
        elif model_type == 'delegate':
            self.llm = delegate
        self.verbose = verbose

    def predict(self, x, mojito=False):
        xcs = []
        for idx in range(len(x)):
            xc = x.iloc[[idx]].copy()
            if self.fake:
                if random.choice([True, False]):
                    match_score = 1
                    nomatch_score = 0
                else:
                    match_score = 0
                    nomatch_score = 1
            else:
                elt = []
                ert = []
                for c in xc.columns:
                    if c in ['ltable_id', 'rtable_id']:
                        continue
                    if c.startswith('ltable_'):
                        elt.append(str(c) + ':' + xc[c].astype(str).values[0])
                    if c.startswith('rtable_'):
                        ert.append(str(c) + ':' + xc[c].astype(str).values[0])
                question = self.prompt.format(ltuple='\n'.join(elt), rtuple='\n'.join(ert))
                answer = self.llm(question, er=True)
                if self.verbose:
                    print(question)
                    print(answer)
                nomatch_score, match_score = self.text_to_match(answer)
            xc['nomatch_score'] = nomatch_score
            xc['match_score'] = match_score
            self.count += 1
            if mojito:
                full_df = np.dstack((xc['nomatch_score'], xc['match_score'])).squeeze()
                xc = full_df
            xcs.append(xc)
            if self.verbose:
                print(f'{self.count},{self.summarized},{self.idks}')
        return pd.concat(xcs, axis=0)

    def text_to_match(self, answer, n=0):
        if type(answer) == dict:
            answer = answer['prediction']
        else:
            json_answers = answer.split('```')
            if len(json_answers) > 1:
                json_answer = json.loads(answer[1].replace('´', '').replace('`', ''))
                if "matching_prediction" in json_answer:
                    answer = json_answer['matching_prediction']
                else:
                    raise Exception

        no_match_score = 0
        match_score = 0
        answer_lc = answer.lower()
        if answer_lc.startswith("true") or answer_lc.startswith("yes"):
            match_score = 1
        elif answer_lc.startswith("no"):
            no_match_score = 1
        else:
            if "yes".casefold() in answer.casefold():
                match_score = 1
            elif "no".casefold() in answer.casefold():
                no_match_score = 1
            elif n == 0:
                template = "summarize {response} as yes or no"
                summarize_prompt = PromptTemplate(
                    input_variables=["response"],
                    template=template,
                )
                summarized_answer = self.llm(summarize_prompt.format(response=answer))
                self.summarized += 1
                snms, sms = self.text_to_match(summarized_answer, n=1)
                if snms == 0 and sms == 0:
                    self.idks += 1
                    no_match_score = 1
        return no_match_score, match_score


# deprecated
class PredictThenSelfExplainER:
    def __init__(self, explanation_granularity: str = "attribute", why: bool = False, temperature=0):
        self.explanation_granularity = explanation_granularity
        self.why = why
        self.temperature = temperature

    def er(self, ltuple: str, rtuple: str):
        question = "record1:\n" + ltuple + "\n record2:\n" + rtuple + "\n"
        return self.__call__(question, er=True, explanation=False)

    def explain(self, ltuple: str, rtuple: str, prediction: str, *args, **kwargs):
        question = "record1:\n" + ltuple + "\n record2:\n" + rtuple + "\n"
        return self.__call__(question, er=True, explanation=True, prediction=prediction, temperature=self.temperature,
                             *args, **kwargs)

    def predict_and_explain(self, ltuple, rtuple):
        question = "record1:\n" + str(ltuple) + "\n record2:\n" + str(rtuple) + "\n"
        return self.__call__(question, er=True, saliency=True, cf=True, explanation=True, ltuple=ltuple, rtuple=rtuple)

    def __call__(self, question, er: bool = False, saliency: bool = False, cf: bool = False,
                 explanation=False, prediction=None, ltuple='', rtuple='', *args, **kwargs):
        answers = dict()
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        conversation = []
        if er:
            for prompt_message in ellmer.utils.read_prompt('ellmer/prompts/er.txt'):
                conversation.append(
                    {"role": prompt_message[0],
                     "content": prompt_message[1].replace("feature", self.explanation_granularity)})
        conversation.append({"role": "user", "content": question})
        if prediction is None:
            response = ellmer.utils.completion_with_backoff(
                deployment_id="gpt-35-turbo", model="gpt-3.5-turbo",
                messages=conversation, temperature=self.temperature
            )
            prediction = response["choices"][0]["message"]
            if "content" in prediction:
                prediction_content = prediction["content"]
                _, prediction = ellmer.utils.text_to_match(prediction_content, self.__call__)
        answers['prediction'] = prediction

        if explanation:
            conversation.append({"role": "assistant", "content": prediction})

            # natural language explanation
            if self.why:
                for prompt_message in ellmer.utils.read_prompt('ellmer/prompts/er-why.txt'):
                    conversation.append(
                        {"role": prompt_message[0],
                         "content": prompt_message[1].replace("feature", self.explanation_granularity)})
                nl_exp = ellmer.utils.completion_with_backoff(
                    deployment_id="gpt-35-turbo", model="gpt-3.5-turbo",
                    messages=conversation, temperature=self.temperature
                )["choices"][0]["message"]
                if "content" in nl_exp:
                    nl_exp = nl_exp["content"]
                answers['nl_exp'] = nl_exp
                conversation.append({"role": "assistant", "content": nl_exp})

            # saliency explanation
            for prompt_message in ellmer.utils.read_prompt('ellmer/prompts/er-saliency.txt'):
                conversation.append(
                    {"role": prompt_message[0],
                     "content": prompt_message[1].replace("feature", self.explanation_granularity)})
            saliency_exp = ellmer.utils.completion_with_backoff(
                deployment_id="gpt-35-turbo", model="gpt-3.5-turbo",
                messages=conversation, temperature=self.temperature
            )["choices"][0]["message"]
            if "content" in saliency_exp:
                saliency_answer = saliency_exp["content"]
                saliency_exp = dict()
                try:
                    saliency = saliency_answer.content.split('```')[1]
                    saliency_dict = json.loads(saliency)
                    saliency_exp = saliency_dict
                except:
                    pass
            answers['saliency'] = saliency_exp

            conversation.append({"role": "assistant", "content": str(saliency_exp)})

            # counterfactual explanation
            for prompt_message in ellmer.utils.read_prompt('ellmer/prompts/er-cf.txt'):
                conversation.append(
                    {"role": prompt_message[0],
                     "content": prompt_message[1].replace("feature", self.explanation_granularity)})
            cf_exp = ellmer.utils.completion_with_backoff(
                deployment_id="gpt-35-turbo", model="gpt-3.5-turbo",
                messages=conversation, temperature=self.temperature
            )["choices"][0]["message"]
            if "content" in cf_exp:
                cf_answer = cf_exp["content"]
                cf_exp = dict()
                try:
                    cf_answer_content = cf_answer.content
                    if '```' in cf_answer_content:
                        cf_answer_json = cf_answer_content.split('```')[1]
                    elif cf_answer_content.startswith("{"):
                        cf_answer_json = cf_answer_content
                    elif "{" in cf_answer_content and "}" in cf_answer_content:
                        cf_answer_json = cf_answer_content[
                                         cf_answer_content.index("{"):cf_answer_content.rfind("}") + 1]
                        # cf_answer_json = ''.join(cf_answer_content.split("{")[1].split("}")[0])
                    else:
                        cf_answer_json = cf_answer_content
                    cf_dict = ast.literal_eval(cf_answer_json)
                    keys = cf_dict.keys()
                    if "record_after" in keys:
                        cf_explanation = cf_dict["record_after"]

                        if list(cf_explanation.keys())[0].startswith('rtable_'):
                            cf_explanation = cf_explanation | ast.literal_eval(ltuple)
                        else:
                            cf_explanation = cf_explanation | ast.literal_eval(rtuple)
                    elif "counterfactual_record" in keys:
                        cf_explanation = cf_dict["counterfactual_record"]
                    elif "counterfactual" in keys:
                        cf_explanation = cf_dict['counterfactual']
                    elif "record1" in keys and "record2" in keys:
                        for k in cf_dict['record1'].keys():
                            if not k.startswith('ltable_'):
                                cf_dict['record1']['ltable_' + k] = cf_dict['record1'][k]
                                cf_dict['record1'].pop(k)

                        for k in cf_dict['record2'].keys():
                            if not k.startswith('rtable_'):
                                cf_dict['record2']['rtable_' + k] = cf_dict['record2'][k]
                                cf_dict['record2'].pop(k)
                        cf_exp = cf_dict['record1'] | cf_dict['record2']
                    else:
                        cf_exp = cf_dict
                except:
                    pass

            answers['cf'] = cf_exp
        return answers

    def predict(self, x, mojito=False):
        xcs = []
        for idx in range(len(x)):
            xc = x.iloc[[idx]].copy()
            ltuple, rtuple = ellmer.utils.get_tuples(xc)
            question = "record1:\n" + str(ltuple) + "\n record2:\n" + str(rtuple) + "\n"
            nomatch_score, match_score = ellmer.utils.text_to_match(self.__call__(question, er=True), self.__call__)
            xc['nomatch_score'] = nomatch_score
            xc['match_score'] = match_score
            if mojito:
                full_df = np.dstack((xc['nomatch_score'], xc['match_score'])).squeeze()
                xc = full_df
            xcs.append(xc)
        return pd.concat(xcs, axis=0)

    def evaluation(self, data_df):
        predictions = self.predict(data_df)
        predictions = predictions['match_score'].astype(int).values
        labels = data_df['label'].astype(int).values
        return f1_score(y_true=labels, y_pred=predictions)


# deprecated
class PredictAndSelfExplainER:

    def __init__(self, explanation_granularity: str = "attribute", temperature: float = 0):
        self.explanation_granularity = explanation_granularity
        self.temperature = temperature

    def er(self, ltuple: str, rtuple: str, temperature=0.99):
        question = "record1:\n" + ltuple + "\n record2:\n" + rtuple + "\n"
        return self.__call__(question, er=True, temperature=temperature)

    def predict_and_explain(self, ltuple, rtuple):
        question = "record1:\n" + str(ltuple) + "\n record2:\n" + str(rtuple) + "\n"
        prediction, saliency_explanation, cf_explanation = parse_pase_answer(self.__call__(question, er=True), self)
        if prediction is None:
            print(f'empty prediction!\nquestion:{question}\n')
        return {"prediction": prediction, "saliency": saliency_explanation, "cf": cf_explanation}

    def __call__(self, question, er: bool = False, *args, **kwargs):
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        conversation = []
        if er:
            for prompt_message in ellmer.utils.read_prompt('ellmer/prompts/constrained7.txt'):
                conversation.append(
                    {"role": prompt_message[0],
                     "content": prompt_message[1].replace("feature", self.explanation_granularity)})
        conversation.append({"role": "user", "content": question})
        response = ellmer.utils.completion_with_backoff(
            deployment_id="gpt-35-turbo", model="gpt-3.5-turbo",
            messages=conversation, temperature=self.temperature
        )
        try:
            answer = response["choices"][0]["message"]["content"]
        except:
            answer = response["choices"][0]["message"]
        return answer

    def predict(self, x, mojito=False):
        xcs = []
        for idx in range(len(x)):
            xc = x.iloc[[idx]].copy()
            ltuple, rtuple = ellmer.utils.get_tuples(xc)
            question = "record1:\n" + str(ltuple) + "\n record2:\n" + str(rtuple) + "\n"
            nomatch_score, match_score = ellmer.utils.text_to_match(self.__call__(question, er=True), self.__call__)
            xc['nomatch_score'] = nomatch_score
            xc['match_score'] = match_score
            if mojito:
                full_df = np.dstack((xc['nomatch_score'], xc['match_score'])).squeeze()
                xc = full_df
            xcs.append(xc)
        return pd.concat(xcs, axis=0)

    def evaluation(self, data_df):
        predictions = self.predict(data_df)
        predictions = predictions['match_score'].astype(int).values
        labels = data_df['label'].astype(int).values
        return f1_score(y_true=labels, y_pred=predictions)


# deprecated
class AzureOpenAIERModel(ERModel):

    def __init__(self, temperature: float = 0):
        self.temperature = temperature

    def __call__(self, question, er=False, *args, **kwargs):
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        conversation = []
        if er:
            for prompt_message in ellmer.utils.read_prompt('ellmer/prompts/er.txt'):
                conversation.append(
                    {"role": prompt_message[0],
                     "content": prompt_message[1]})
        conversation.append({"role": "user", "content": question})
        response = ellmer.utils.completion_with_backoff(
            deployment_id="gpt-35-turbo", model="gpt-3.5-turbo",
            messages=conversation, temperature=self.temperature
        )
        prediction = response["choices"][0]["message"]["content"]
        if er:
            return ellmer.utils.text_to_match(prediction, self.__call__)
        else:
            return prediction

    def predict(self, x, mojito=False):
        xcs = []
        for idx in range(len(x)):
            xc = x.iloc[[idx]].copy()
            ltuple, rtuple = ellmer.utils.get_tuples(xc)
            question = "record1:\n" + str(ltuple) + "\n record2:\n" + str(rtuple) + "\n"
            nomatch_score, match_score = self.__call__(question, er=True)
            xc['nomatch_score'] = nomatch_score
            xc['match_score'] = match_score
            if mojito:
                full_df = np.dstack((xc['nomatch_score'], xc['match_score'])).squeeze()
                xc = full_df
            xcs.append(xc)
        return pd.concat(xcs, axis=0)


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


class MinunExplainer(object):
    # The explainer class
    def __init__(self,
                 model,
                 ):
        # will be the model itself that can be directly used for prediction
        self.model = model

    def to_df(self, instance):
        df_d = dict()
        for i in range(len(instance[0])):
            df_d[instance[3][i]] = instance[0][i]
        for i in range(len(instance[1])):
            df_d[instance[3][i]] = instance[1][i]
        df_d['label'] = instance[2]
        return pd.DataFrame(pd.Series(df_d)).T

    def explain(self, instance, method="greedy", k=10):
        '''
            k: number of explaiantions to return, cannot exceed self.maximum_samples
        Returns:
            exp_result(DefaultDict): the explaination of given instance in the format of key-value pairs, where
        '''
        # 1. For each pais of attribute in the instance, identify all possible edit ops
        # 2. Repeatly search for N samples with _generate_samples algorithms
        # 3. for each sample, evaluate the black box model by calling _eval_model function
        # 4. End above loop if finding k samples that can flip the results or finish enumerating the search space
        # 5. refine the candidates and return explaination
        '''list(pd_instance.loc[:, pd_instance.columns.str.startswith('ltable_')].values[0])
        list(pd_instance.loc[:, pd_instance.columns.str.startswith('rtable_')].values[0])
        pd_instance['label']
        [c.replace('ltable_', '') for c in pd_instance.columns if c.startswith('ltable_')]'''

        attrs = self._entity2pair_attrs(instance)
        cands4attrs = [[]]
        for pair in attrs:
            if instance[2] == 0:
                res = self._generate_candidates_from_one_attribute_neg(pair[0], pair[1])
            else:
                res = self._generate_candidates_from_one_attribute_pos(pair[0], pair[1])
            cands4attrs.append(res)
        cands4attrs = cands4attrs[1:]
        assert len(cands4attrs) > 0
        if method == "greedy":
            candidates, eval_cnt = self._explain_permutation(instance, cands4attrs, k)
        elif method == "min":
            candidates, eval_cnt = self._explain_min_attr(instance, cands4attrs, k)
        elif method == "binary":
            candidates, eval_cnt = self._explain_binary(instance, cands4attrs, k)
        else:
            raise Exception('Not Support this kind of explaination method!')
            # corner case: if no flip happens:
        # original prediction is 0: just add all attributes of the right entity as explaination
        # original prediction is 1: return an empty dict denoting delete all attributes.
        exp_result = defaultdict()
        if len(candidates) == 0:
            print("cannot flip!")
            return (defaultdict(), pd.DataFrame()), eval_cnt
        else:
            top_explaination = self._find_explaination(candidates)
            for idx, onum in enumerate(top_explaination[-1]):
                if onum != 0:
                    exp_result[instance[-1][idx]] = cands4attrs[idx][onum]
            res = (exp_result, top_explaination[0])
            return res, eval_cnt

    def _explain_permutation(self, instance, cands4attrs, k):
        '''
        Args:
            cands4attrs(Array): list of candidates in each attributes: array length equals to
            the number of attributes, each item contain a list of possible transformation in that attribute
            k(int): number of explaiantions to return, cannot exceed self.maximum_samples
        Returns:
            candidates(Array): an array contains k explainations
        '''
        eval_cnt = 0
        candidates = []
        perms = []
        for cands in cands4attrs:
            perms.append(range(len(cands)))
        # print(perms)
        for indices in product(*perms):
            num = sum(indices)
            if num == 0:
                continue
            sample = self._format_instance(cands4attrs, indices, self.to_df(instance))
            predict, logit = self._eval_model(sample)
            eval_cnt = eval_cnt + 1
            if str(predict[0]) != str(instance[2]):  # result is flipped
                # candidates.append((sample,num,logit[0],indices))
                # print(str(predict[0])+"#"+str(instance[2])+"#")
                candidates.append((sample, num, max(logit[0][0], logit[0][1]), indices))
            if len(candidates) >= k:
                return candidates, eval_cnt
        return candidates, eval_cnt

    def _explain_min_attr(self, instance, cands4attrs, k):
        '''
        Args and Returns are the same above.
        Enumerate the combination w.r.t minimum number of attributes
        '''
        eval_cnt = 0
        candidates = []
        perms = []
        tmp_dict = defaultdict(list)
        num_dim = len(cands4attrs)
        for cands in cands4attrs:
            perms.append(range(len(cands)))
        for indices in product(*perms):
            attr_num = num_dim - list(indices).count(0)
            tmp_dict[attr_num].append(indices)
        dimnums = sorted(tmp_dict.keys())

        for num in dimnums:
            if num == 0:
                continue
            tmp_dict[num].sort(key=lambda d: sum(d))
            for item in tmp_dict[num]:
                sample = self._format_instance(cands4attrs, item, self.to_df(instance))
                predict, logit = self._eval_model(sample)
                eval_cnt += 1
                if str(predict[0]) != str(instance[2]):  # result is flipped
                    candidates.append((sample, num, max(logit[0][0], logit[0][1]), indices))
                if len(candidates) >= k:
                    return candidates, eval_cnt
        return candidates, eval_cnt

    def _explain_binary(self, instance, cands4attrs, k):
        '''
        Use binary search algorithm
        Args:
            N: maximum number of samples to be generated, if the total number of samples is smaller than N,
            then return all
        Returns:
            samples(Array): an array contains k explainations
        '''
        eval_cnt = 0
        candidates = []
        perms = []
        for cands in cands4attrs:
            perms.append(len(cands))
        num_dim = len(cands4attrs)
        flip_indices = []
        for attr_num in range(1, num_dim + 1):
            if attr_num == 1:
                for dim in range(num_dim):
                    cur_indice = [0] * num_dim
                    low = 0
                    high = perms[dim] - 1
                    while low < high:
                        mid = (high + low) / 2
                        cur_indice[dim] = int(mid)
                        sample = self._format_instance(cands4attrs, cur_indice, self.to_df(instance))
                        predict, logit = self._eval_model([sample])
                        eval_cnt += 1
                        if str(predict[0]) != str(instance[2]):  # result is flipped
                            candidates.append((sample, int(mid), max(logit[0][0], logit[0][1]), cur_indice))
                            if len(candidates) >= k:
                                return candidates, eval_cnt
                            high = mid - 1
                        else:
                            low = mid + 1
            else:
                for group in combinations(range(num_dim), attr_num):
                    lbs = []
                    # print(group)
                    for dim in group:
                        lbs.append(list((1, perms[dim])))
                    local_indice = [0] * attr_num
                    max_time = 1
                    for dim in group:
                        max_time *= perms[dim]
                    enum_time = 0
                    while enum_time < max_time:
                        tag = False
                        for attr_idx in range(attr_num):
                            enum_time += 1
                            if lbs[attr_idx][0] < lbs[attr_idx][1]:
                                tag = True
                                mid = (lbs[attr_idx][1] + lbs[attr_idx][0]) / 2
                                local_indice[attr_idx] = int(mid)
                                if enum_time >= attr_num:
                                    cur_indice = [0] * num_dim
                                    for idx, dim in enumerate(group):
                                        cur_indice[dim] = local_indice[idx]
                                    sample = self._format_ditto_instance(cands4attrs, cur_indice, instance)
                                    predict, logit = self._eval_model([sample])
                                    eval_cnt += 1
                                    if str(predict[0]) != str(instance[2]):  # result is flipped
                                        num = sum(cur_indice)
                                        candidates.append((sample, num, max(logit[0][0], logit[0][1]), cur_indice))
                                        if len(candidates) >= k:
                                            return candidates, eval_cnt
                                        lbs[attr_idx][1] = mid - 1
                                        # continue
                                    else:
                                        lbs[attr_idx][0] = mid + 1
                        if tag == False:
                            break
        return candidates, eval_cnt

    def _entity2pair_attrs(self, instance):
        attrs = []
        assert (len(instance[0]) == len(instance[1]))
        for idx in range(len(instance[0])):
            item = (instance[0][idx], instance[1][idx])
            attrs.append(item)
        return attrs

    def _min_cost_path(self, cost, operations):
        # operation at the last cell
        path = [operations[cost.shape[0] - 1][cost.shape[1] - 1]]

        # cost at the last cell
        min_cost = cost[cost.shape[0] - 1][cost.shape[1] - 1]

        row = cost.shape[0] - 1
        col = cost.shape[1] - 1

        while row > 0 and col > 0:
            if cost[row - 1][col - 1] <= cost[row - 1][col] and cost[row - 1][col - 1] <= cost[row][col - 1]:
                path.append(operations[row - 1][col - 1])
                row -= 1
                col -= 1
            elif cost[row - 1][col] <= cost[row - 1][col - 1] and cost[row - 1][col] <= cost[row][col - 1]:
                path.append(operations[row - 1][col])
                row -= 1
            else:
                path.append(operations[row][col - 1])
                col -= 1
        return "".join(path[::-1][1:])

    def _token_edit_distance(self, str1, str2):
        '''
        Args:
            str1/2: two entities for calculation
        Returns:
            dist(int): the token-level edit distance between two entities
            ops(nparray): the operations to formulate the path
        '''
        seq1 = str1.split(' ')
        seq2 = str2.split(' ')
        if len(str1) == 0 and len(str2) == 0:
            return 0, []
        if len(str1) == 0:
            ops = []
            for i in range(len(seq2)):
                ops.append('I')
            return len(seq2), ops
        if len(str2) == 0:
            ops = []
            for i in range(len(seq1)):
                ops.append('D')
            return len(seq1), ops
        matrix = np.zeros((len(seq1) + 1, len(seq2) + 1))
        matrix[0] = [i for i in range(len(seq2) + 1)]
        matrix[:, 0] = [i for i in range(len(seq1) + 1)]

        ops = np.asarray([['-' for j in range(len(seq2) + 1)] \
                          for i in range(len(seq1) + 1)])
        ops[0] = ['I' for i in range(len(seq2) + 1)]
        ops[:, 0] = ['D' for i in range(len(seq1) + 1)]
        ops[0, 0] = '-'

        for row in range(1, len(seq1) + 1):
            for col in range(1, len(seq2) + 1):
                if seq1[row - 1] == seq2[col - 1]:
                    matrix[row][col] = matrix[row - 1][col - 1]
                else:
                    insertion_cost = matrix[row][col - 1] + 1
                    deletion_cost = matrix[row - 1][col] + 1
                    substitution_cost = matrix[row - 1][col - 1] + 1
                    matrix[row][col] = min(insertion_cost, deletion_cost, substitution_cost)
                    if matrix[row][col] == substitution_cost:
                        ops[row][col] = 'S'
                    elif matrix[row][col] == insertion_cost:
                        ops[row][col] = 'I'
                    else:
                        ops[row][col] = 'D'
        dist = matrix[len(seq1), len(seq2)]
        operations = self._min_cost_path(matrix, ops)
        return dist, operations

    def _generate_candidates_from_one_attribute_pos(self, attr1, attr2):
        '''
        If the initial matching label is 1, then it requires to delete tokens from attr1 one by one until empty
        '''
        candidates = [""]
        seq = attr1.split(' ')
        tmp_str = ""
        for token in seq:
            tmp_str += str(token) + " "
            candidates.append(tmp_str.strip())
        return candidates[::-1]

    def _generate_candidates_from_one_attribute_neg(self, attr1, attr2):
        candidates = [attr1]
        dist, operations = self._token_edit_distance(attr1, attr2)
        tmp = attr1.split(' ')
        target = attr2.split(' ')
        if len(target) == 0:
            return candidates
        cur = 0  # for seq2
        dnum = 0
        for idx, op in enumerate(operations):
            if op == '-':
                cur += 1
                continue
            else:
                # errors of index out of range happens here when running on server
                # temp patch, will fix later
                if op == 'S':
                    pos1 = idx - dnum
                    pos2 = cur
                    if pos1 >= len(tmp):
                        pos1 = -1
                    if pos2 >= len(target):
                        pos2 = -1
                    tmp[pos1] = target[pos2]
                    cur += 1
                elif op == 'I':
                    pos3 = cur
                    if pos3 >= len(target):
                        pos3 = -1
                    tmp.insert(idx, target[pos3])
                    cur += 1
                else:
                    pos4 = idx - dnum
                    if pos4 >= len(tmp):
                        pos4 = -1
                    del tmp[pos4]
                    dnum += 1
                tmp_str = ""
                for token in tmp:
                    tmp_str += str(token) + " "
                candidates.append(tmp_str.strip())
        return candidates

    def _format_ditto_instance(self, attrs, indices, instance):
        '''
        Args:
            attrs (list): The candidates of left entity for each attribute
            indices (list): The list of indices for choosing candidate
            instance (tuple)
        Returns:
            ditto_item(str): The item that can be fed into the ditto model for testing
        '''
        assert len(attrs) == len(indices)
        left = ""
        right = ""
        for i in range(len(attrs)):
            left += " COL " + str(instance[3][i]) + " VAL " + str(attrs[i][indices[i]])
            right += " COL " + str(instance[3][i]) + " VAL " + str(instance[1][i])
        res = left.strip() + "\t" + right.strip() + "\t" + str(instance[2])
        return res

    def _format_instance(self, attrs, indices, pd_instance):
        '''
        Args:
            attrs (list): The candidates of left entity for each attribute
            indices (list): The list of indices for choosing candidate
            instance (tuple)
        Returns:
            ditto_item(str): The item that can be fed into the ditto model for testing
        '''
        perturbed_df = pd_instance.copy()
        for i in range(len(attrs)):
            perturbed_df[perturbed_df.columns[i]] = str(attrs[i][indices[i]])
        return perturbed_df

    def _eval_model_random(self, samples):
        '''
        Generate a random value instead of use ML model for prediction, just for testing
        '''
        results = []
        logits = []
        for sample in samples:
            val = random.random()
            if val > 0.5:
                results.append(1)
            else:
                results.append(0)
            logits.append(val)
        return results, logits

    def _find_explaination(self, candidates):
        '''
        Each candidate is a tuple with 3 attributes: the sentence, # ops, val of logit
        '''
        candidates.sort(key=lambda l: (l[1], -l[2]))
        return candidates[0]

    def _eval_model(self, samples, prob=True):
        '''
        Args:
            samples(Array): the set of candidate samples
            prob(Boolean): whether return both
        Returns:
            logits(Tensor): the tensor of logits for all samples, shape: N*2, where N is the number of samples
            predication(dataFrame): the predicted class label (0/1)
        '''

        predictions = self.model.predict(samples)
        results = np.argmax(predictions[['nomatch_score', 'match_score']])
        if prob == True:
            logits = [predictions['match_score'].values,
                      predictions['nomatch_score'].values]
            return [results], [logits]
        else:
            return [results]


def formulate_instance(tableA, tableB, inst):
    '''
    Args:
        tableA/tableB(dataFrame): the two tables
        inst(dataFrame): one test instance
    Returns:
        item: a triplet with two entities and label
    '''
    id1 = int(inst[0])
    id2 = int(inst[1])
    header = list(tableA)
    attr_num = len(header)
    left = []
    right = []
    for idx in range(attr_num):
        if pd.isnull(tableA.iloc[id1][idx]):
            left.append("")
        else:
            left.append(str(tableA.iloc[id1][idx]))
        if pd.isnull(tableB.iloc[id2][idx]):
            right.append("")
        else:
            right.append(str(tableB.iloc[id2][idx]))
    item = (left, right, inst[2], header)
    return item


class Landmark(object):

    def __init__(self, model, dataset, exclude_attrs=['id', 'label'], split_expression=' ',
                 lprefix='ltable_', rprefix='rtable_', exclude_tokens = None, **argv, ):
        """

        :param model: the model to be explained
        :param dataset: containing the elements that will be explained. Used to save the attribute structure.
        :param exclude_attrs: attributes to be excluded from the explanations
        :param split_expression: to divide tokens from string
        :param lprefix: left prefix
        :param rprefix: right prefix
        :param argv: other optional parameters that will be passed to LIME
        """
        self.splitter = re.compile(split_expression)
        self.split_expression = split_expression
        self.explainer = LimeTextExplainer(class_names=['NO match', 'MATCH'], split_expression=split_expression, **argv)
        self.model = model
        self.dataset = dataset
        self.lprefix = lprefix
        self.rprefix = rprefix
        self.exclude_attrs = exclude_attrs
        self.exclude_tokens = exclude_tokens

        self.cols = [x for x in dataset.columns if x not in exclude_attrs]
        self.left_cols = [x for x in self.cols if x.startswith(self.lprefix)]
        self.right_cols = [x for x in self.cols if x.startswith(self.rprefix)]
        self.cols = self.left_cols + self.right_cols
        self.explanations = {}

    def explain(self, elements, conf='auto', num_samples=500, **argv):
        """
        User interface to generate an explanations with the specified configurations for the elements passed in input.
        """
        assert type(elements) == pd.DataFrame, f'elements must be of type {pd.DataFrame}'
        allowed_conf = ['auto', 'single', 'double', 'LIME']
        assert conf in allowed_conf, 'conf must be in ' + repr(allowed_conf)
        if elements.shape[0] == 0:
            return None

        if 'auto' == conf:
            match_elements = elements[elements.label == 1]
            no_match_elements = elements[elements.label == 0]
            match_explanation = self.explain(match_elements, 'single', num_samples, **argv)
            no_match_explanation = self.explain(no_match_elements, 'double', num_samples, **argv)
            return pd.concat([match_explanation, no_match_explanation])

        impact_list = []
        if 'LIME' == conf:
            for idx in range(elements.shape[0]):
                impacts = self.explain_instance(elements.iloc[[idx]], variable_side='all', fixed_side=None,
                                                num_samples=num_samples, **argv)
                impacts['conf'] = 'LIME'
                impact_list.append(impacts)
            self.impacts = pd.concat(impact_list)
            return self.impacts

        landmark = 'right'
        variable = 'left'
        overlap = False
        if 'single' == conf:
            add_before = None
        elif 'double' == conf:
            add_before = landmark

        # right landmark
        for idx in range(elements.shape[0]):
            impacts = self.explain_instance(elements.iloc[[idx]], variable_side=variable, fixed_side=landmark,
                                            add_before_perturbation=add_before, num_samples=num_samples,
                                            overlap=overlap, **argv)
            impacts['conf'] = f'{landmark}_landmark' + ('_injection' if add_before is not None else '')
            impact_list.append(impacts)

        # switch sides
        landmark, variable = variable, landmark
        if add_before is not None:
            add_before = landmark

        # left landmark
        for idx in range(elements.shape[0]):
            impacts = self.explain_instance(elements.iloc[[idx]], variable_side=variable, fixed_side=landmark,
                                            add_before_perturbation=add_before, num_samples=num_samples,
                                            overlap=overlap, **argv)
            impacts['conf'] = f'{landmark}_landmark' + ('_injection' if add_before is not None else '')
            impact_list.append(impacts)

        self.impacts = pd.concat(impact_list)
        return self.impacts

    def explain_instance(self, el, variable_side='left', fixed_side='right', add_before_perturbation=None,
                         add_after_perturbation=None, overlap=True, num_samples=500, **argv):
        """
        Main method to wrap the explainer and generate an landmark. A sort of Facade for the explainer.

        :param el: DataFrame containing the element to be explained.
        :return: landmark DataFrame
        """
        variable_el = el.copy()
        for col in self.cols:
            variable_el[col] = ' '.join(re.split(r' +', str(variable_el[col].values[0]).strip()))

        variable_data = self.prepare_element(variable_el, variable_side, fixed_side, add_before_perturbation,
                                             add_after_perturbation, overlap)

        words = self.splitter.split(variable_data)
        explanation = self.explainer.explain_instance(variable_data, self.restucture_and_predict,
                                                      num_features=len(words), num_samples=num_samples,
                                                      **argv)
        self.variable_data = variable_data  # to test the addition before perturbation

        id = el.index.values[0]  # Use the index is the id column
        self.explanations[f'{self.fixed_side}{id}'] = explanation
        return self.explanation_to_df(explanation, words, self.mapper_variable.attr_map, id)

    def prepare_element(self, variable_el, variable_side, fixed_side, add_before_perturbation, add_after_perturbation,
                        overlap):
        """
        Compute the data and set parameters needed to perform the landmark.
            Set fixed_side, fixed_data, mapper_variable.
            Call compute_tokens if needed
        """

        self.add_after_perturbation = add_after_perturbation
        self.overlap = overlap
        self.fixed_side = fixed_side
        if variable_side in ['left', 'right']:
            variable_cols = self.left_cols if variable_side == 'left' else self.right_cols

            assert fixed_side in ['left', 'right']
            if fixed_side == 'left':
                fixed_cols, not_fixed_cols = self.left_cols, self.right_cols
            else:
                fixed_cols, not_fixed_cols = self.right_cols, self.left_cols
            mapper_fixed = Mapper(fixed_cols, self.split_expression)
            self.fixed_data = mapper_fixed.decode_words_to_attr(mapper_fixed.encode_attr(
                variable_el[fixed_cols]))  # encode and decode data of fixed source to ensure the same format
            self.mapper_variable = Mapper(not_fixed_cols, self.split_expression)

            if add_before_perturbation is not None or add_after_perturbation is not None:
                self.compute_tokens(variable_el)
                if add_before_perturbation is not None:
                    self.add_tokens(variable_el, variable_cols, add_before_perturbation, overlap)
            variable_data = Mapper(variable_cols, self.split_expression).encode_attr(variable_el)

        elif variable_side == 'all':
            variable_cols = self.left_cols + self.right_cols

            self.mapper_variable = Mapper(variable_cols, self.split_expression)
            self.fixed_data = None
            self.fixed_side = 'all'
            variable_data = self.mapper_variable.encode_attr(variable_el)
        else:
            assert False, f'Not a feasible configuration. variable_side: {variable_side} not allowed.'
        return variable_data

    def explanation_to_df(self, explanation, words, attribute_map, id):
        """
        Generate the DataFrame of the landmark from the LIME landmark.

        :param explanation: LIME landmark
        :param words: words of the element subject of the landmark
        :param attribute_map: attribute map to decode the attribute from a prefix
        :param id: id of the element under landmark
        :return: DataFrame containing the landmark
        """
        impacts_list = []
        dict_impact = {'id': id}
        for wordpos, impact in explanation.as_map()[1]:
            word = words[wordpos]
            dict_impact.update(column=attribute_map[word[0]], position=int(word[1:3]), word=word[4:], word_prefix=word,
                               impact=impact)
            impacts_list.append(dict_impact.copy())
        return pd.DataFrame(impacts_list).reset_index()

    def compute_tokens(self, el):
        """
        Divide tokens of the descriptions for each column pair in inclusive and exclusive sets.

        :param el: pd.DataFrame containing the 2 description to analyze
        """
        tokens = {col: np.array(self.splitter.split(str(el[col].values[0]))) for col in self.cols}
        tokens_intersection = {}
        tokens_not_overlapped = {}
        for col in [col.replace('left_', '') for col in self.left_cols]:
            lcol, rcol = self.lprefix + col, self.rprefix + col
            tokens_intersection[col] = np.intersect1d(tokens[lcol], tokens[rcol])
            tokens_not_overlapped[lcol] = tokens[lcol][~ np.in1d(tokens[lcol], tokens_intersection[col])]
            tokens_not_overlapped[rcol] = tokens[rcol][~ np.in1d(tokens[rcol], tokens_intersection[col])]
        self.tokens_not_overlapped = tokens_not_overlapped
        self.tokens_intersection = tokens_intersection
        self.tokens = tokens
        return dict(tokens=tokens, tokens_intersection=tokens_intersection, tokens_not_overlapped=tokens_not_overlapped)

    def add_tokens(self, el, dst_columns, src_side, overlap=True):
        """
        Takes tokens computed before from the src_sside with overlap or not
        and inject them into el in columns specified in dst_columns.

        """
        if not overlap:
            tokens_to_add = self.tokens_not_overlapped
        else:
            tokens_to_add = self.tokens

        if src_side == 'left':
            src_columns = self.left_cols
        elif src_side == 'right':
            src_columns = self.right_cols
        else:
            assert False, f'src_side must "left" or "right". Got {src_side}'

        for col_dst, col_src in zip(dst_columns, src_columns):
            if len(tokens_to_add[col_src]) == 0:
                continue
            el[col_dst] = el[col_dst].astype(str) + ' ' + ' '.join(tokens_to_add[col_src])

    def restucture_and_predict(self, perturbed_strings):
        """
            Restructure the perturbed strings from LIME and return the related predictions.
        """
        self.tmp_dataset = self.restructure_strings(perturbed_strings)
        self.tmp_dataset.reset_index(inplace=True, drop=True)
        predictions = self.model.predict(self.tmp_dataset)

        ret = np.ndarray(shape=(len(predictions), 2))
        ret[:, 1] = np.array(predictions['match_score'])
        ret[:, 0] = 1 - ret[:, 1]
        return ret

    def restructure_strings(self, perturbed_strings):
        """

        Decode :param perturbed_strings into DataFrame and
        :return reconstructed pairs appending the landmark entity.

        """
        df_list = []
        for single_row in perturbed_strings:
            df_list.append(self.mapper_variable.decode_words_to_attr_dict(single_row))
        variable_df = pd.DataFrame.from_dict(df_list)
        if self.add_after_perturbation is not None:
            self.add_tokens(variable_df, variable_df.columns, self.add_after_perturbation, overlap=self.overlap)
        if self.fixed_data is not None:
            fixed_df = pd.concat([self.fixed_data] * variable_df.shape[0])
            fixed_df.reset_index(inplace=True, drop=True)
        else:
            fixed_df = None
        return pd.concat([variable_df, fixed_df], axis=1)

    def double_explanation_conversion(self, explanation_df, item):
        """
        Compute and assign the original attribute of injected words.
        :return: explanation with original attribute for injected words.
        """
        view = explanation_df[['column', 'position', 'word', 'impact']].reset_index(drop=True)
        tokens_divided = self.compute_tokens(item)
        exchanged_idx = [False] * len(view)
        lengths = {col: len(words) for col, words in tokens_divided['tokens'].items()}
        for col, words in tokens_divided['tokens_not_overlapped'].items():  # words injected in the opposite side
            prefix, col_name = col.split('_')
            prefix = 'left_' if prefix == 'right' else 'right_'
            opposite_col = prefix + col_name
            exchanged_idx = exchanged_idx | ((view.position >= lengths[opposite_col]) & (view.column == opposite_col))
        exchanged = view[exchanged_idx]
        view = view[~exchanged_idx]
        # determine injected impacts
        exchanged['side'] = exchanged['column'].apply(lambda x: x.split('_')[0])
        col_names = exchanged['column'].apply(lambda x: x.split('_')[1])
        exchanged['column'] = np.where(exchanged['side'] == 'left', 'right_', 'left_') + col_names
        tmp = view.merge(exchanged, on=['word', 'column'], how='left', suffixes=('', '_injected'))
        tmp = tmp.drop_duplicates(['column', 'word', 'position'], keep='first')
        impacts_injected = tmp['impact_injected']
        impacts_injected = impacts_injected.fillna(0)

        view['score_right_landmark'] = np.where(view['column'].str.startswith('left'), view['impact'], impacts_injected)
        view['score_left_landmark'] = np.where(view['column'].str.startswith('right'), view['impact'], impacts_injected)
        view.drop('impact', 1, inplace=True)

        return view

    def plot(self, explanation, el, figsize=(16,6)):
        exp_double = self.double_explanation_conversion(explanation, el)
        PlotExplanation.plot(exp_double, figsize)


class Mapper(object):
    """
    This class is useful to encode a row of a dataframe in a string in which a prefix
    is added to each word to keep track of its attribute and its position.
    """

    def __init__(self, columns, split_expression):
        self.columns = columns
        self.attr_map = {chr(ord('A') + colidx): col for colidx, col in enumerate(self.columns)}
        self.arange = np.arange(100)
        self.split_expression = split_expression

    def decode_words_to_attr_dict(self, text_to_restructure):
        res = re.findall(r'(?P<attr>[A-Z]{1})(?P<pos>[0-9]{2})_(?P<word>[^' + self.split_expression + ']+)',
                         text_to_restructure)
        structured_row = {col: '' for col in self.columns}
        for col_code, pos, word in res:
            structured_row[self.attr_map[col_code]] += word + ' '
        for col in self.columns:  # Remove last space
            structured_row[col] = structured_row[col][:-1]
        return structured_row

    def decode_words_to_attr(self, text_to_restructure):
        return pd.DataFrame([self.decode_words_to_attr_dict(text_to_restructure)])

    def encode_attr(self, el):
        return ' '.join(
            [chr(ord('A') + colpos) + "{:02d}_".format(wordpos) + word for colpos, col in enumerate(self.columns) for
             wordpos, word in enumerate(re.split(self.split_expression, str(el[col].values[0])))])

    def encode_elements(self, elements):
        word_dict = {}
        res_list = []
        for i in np.arange(elements.shape[0]):
            el = elements.iloc[i]
            word_dict.update(id=el.id)
            for colpos, col in enumerate(self.columns):
                word_dict.update(column=col)
                for wordpos, word in enumerate(re.split(self.split_expression, str(el[col]))):
                    word_dict.update(word=word, position=wordpos,
                                     word_prefix=chr(ord('A') + colpos) + f"{wordpos:02d}_" + word)
                    res_list.append(word_dict.copy())
        return pd.DataFrame(res_list)



class PlotExplanation(object):
    @staticmethod
    def plot_impacts(data, target_col, ax, title):

        n = len(data)
        ax.set_xlim(-0.5, 0.5)  # set x axis limits
        ax.set_ylim(-1, n)  # set y axis limits
        ax.set_yticks(range(n))  # add 0-n ticks
        ax.set_yticklabels(data[['column', 'word']].apply(lambda x: ', '.join(x), 1))  # add y tick labels

        # define arrows
        arrow_starts = np.repeat(0, n)
        arrow_lengths = data[target_col].values
        # add arrows to plot
        for i, subject in enumerate(data['column']):

            if subject.startswith('l'):
                arrow_color = '#347768'
            elif subject.startswith('r'):
                arrow_color = '#6B273D'

            if arrow_lengths[i] != 0:
                ax.arrow(arrow_starts[i],  # x start point
                         i,  # y start point
                         arrow_lengths[i],  # change in x
                         0,  # change in y
                         head_width=0,  # arrow head width
                         head_length=0,  # arrow head length
                         width=0.4,  # arrow stem width
                         fc=arrow_color,  # arrow fill color
                         ec=arrow_color)  # arrow edge color

        # format plot
        ax.set_title(title)  # add title
        ax.axvline(x=0, color='0.9', ls='--', lw=2, zorder=0)  # add line at x=0
        ax.grid(axis='y', color='0.9')  # add a light grid
        ax.set_xlim(-0.5, 0.5)  # set x axis limits
        ax.set_xlabel('Token impact')  # label the x axis
        sns.despine(left=True, bottom=True, ax=ax)

    @staticmethod
    def plot_landmark(exp, landmark):

        if landmark == 'right':
            target_col = 'score_right_landmark'
        else:
            target_col = 'score_left_landmark'

        data = exp.copy()

        # sort individuals by amount of change, from largest to smallest
        data = data.sort_values(by=target_col, ascending=True) \
            .reset_index(drop=True)

        # initialize a plot
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))  # create figure

        if target_col == 'score_right_landmark':
            PlotExplanation.plot_impacts(data[data['column'].str.startswith('l')], target_col, axes[0],
                                         'Original Tokens')
            PlotExplanation.plot_impacts(data[data['column'].str.startswith('r')], target_col, axes[1],
                                         'Augmented Tokens')
        else:
            PlotExplanation.plot_impacts(data[data['column'].str.startswith('r')], target_col, axes[0],
                                         'Original Tokens')
            PlotExplanation.plot_impacts(data[data['column'].str.startswith('l')], target_col, axes[1],
                                         'Augmented Tokens')
            # fig.suptitle('Right Landmark Explanation')
        fig.tight_layout()

    @staticmethod
    def plot(exp, figsize=(16, 6)):
        data = exp.copy()
        # initialize a plot
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=figsize)  # create figure
        target_col = 'score_right_landmark'
        # sort individuals by amount of change, from largest to smallest
        data = data.sort_values(by=target_col, ascending=True).reset_index(drop=True)
        PlotExplanation.plot_impacts(data[data['column'].str.startswith('l')], target_col, axes[0], 'Original Tokens')
        PlotExplanation.plot_impacts(data[data['column'].str.startswith('r')], target_col, axes[1], 'Augmented Tokens')
        axes[0].set_ylabel('Right Landmark')
        axes[1].set_ylabel('Right Landmark')

        target_col = 'score_left_landmark'
        PlotExplanation.plot_impacts(data[data['column'].str.startswith('r')], target_col, axes[2], 'Original Tokens')
        PlotExplanation.plot_impacts(data[data['column'].str.startswith('l')], target_col, axes[3], 'Augmented Tokens')
        axes[2].set_ylabel('Left Landmark')
        axes[3].set_ylabel('Left Landmark')

        fig.tight_layout()


class HybridGeneric(BaseLLMExplainer):

    def __init__(self, explanation_granularity, pred_delegate, ellmers, lsource, rsource,
                 num_draws=1, num_triangles=10, combine : str = 'freq', top_k: int = 1):
        self.explanation_granularity = explanation_granularity
        self.minun = MinunExplainer(pred_delegate)
        self.num_triangles = num_triangles
        self.delegate = pred_delegate
        self.lsource = lsource
        self.rsource = rsource
        self.ellmers = ellmers
        self.predict_fn = lambda x: self.delegate.predict(x)
        self.num_draws = num_draws
        self.combine = combine
        self.top_k = top_k

    def predict_and_explain(self, ltuple, rtuple, max_predict: int = -1, verbose: bool = False):
        satisfied = False
        saliency_explanation = {}
        cf_explanation = {}
        filter_features = []

        if 'attribute' == self.explanation_granularity:
            no_features = len(ltuple) + len(rtuple)
        elif 'token' == self.explanation_granularity:
            no_features = len(str(ltuple).split(' ')) + len(str(rtuple).split(' '))
        else:
            raise ValueError('invalid explanation granularity')

        top_k = self.top_k
        its = 0
        pae_dicts = []
        for e in self.ellmers:
            for _ in range(self.num_draws):
                pae_dict = e.predict_and_explain(ltuple, rtuple)['saliency']
                pae_dicts.append(pae_dict)

        pae = self.delegate.predict_tuples(ltuple, rtuple)
        prediction = pae

        while not satisfied:
            if self.combine == 'freq':
                # get most frequent features from the self-explanations
                filter_features = []
                for se in pae_dicts:
                    fse = {k: v for k, v in se.items() if v > 0}
                    if len(fse) > 0:
                        sorted_attributes_dict = sorted(fse.items(), key=operator.itemgetter(1), reverse=True)[:top_k]
                        top_features = [f[0] for f in sorted_attributes_dict]
                        filter_features = filter_features + top_features
                fc = {}
                for f in filter_features:
                    if f in fc:
                        fc[f] = fc[f] + 1
                    else:
                        fc[f] = 1
                sorted_fc = sorted(fc.items(), key=operator.itemgetter(1), reverse=True)[:top_k]
                filter_features = [sfc[0] for sfc in sorted_fc]
            elif self.combine == 'union':
                # get all features from the self-explanations
                filter_features = []
                for se in pae_dicts:
                    sorted_attributes_dict = sorted(se.items(), key=operator.itemgetter(1), reverse=True)[:top_k]
                    top_features = [f[0] for f in sorted_attributes_dict]
                    for tf in top_features:
                        if tf not in filter_features:
                            filter_features.append(tf)
            elif self.combine == 'intersection':
                # get recurring features only from the self-explanations
                filter_features = set()
                for se in pae_dicts:
                    sorted_attributes_dict = sorted(se.items(), key=operator.itemgetter(1), reverse=True)[:top_k]
                    top_features = set([f[0] for f in sorted_attributes_dict])
                    if len(filter_features) == 0:
                        filter_features = top_features
                    filter_features = filter_features.intersection(top_features)
                filter_features = list(filter_features)
            else:
                raise ValueError("Unknown combination method")
            if len(filter_features) > 0:
                if 'token' == self.explanation_granularity:
                    token_attributes_filtered = []
                    for ff in filter_features:
                        if ff.startswith('ltable_'):
                            for token in ltuple[ff].split(' '):
                                token_attributes_filtered.append(ff+'__'+token)
                        if ff.startswith('rtable_'):
                            for token in rtuple[ff].split(' '):
                                token_attributes_filtered.append(ff+'__'+ token)
                        else:
                            for ic in ltuple.keys():
                                if ff in ltuple[ic].split(' '):
                                    token_attributes_filtered.append(ic+'__' + ff)
                            for ic in rtuple.keys():
                                if ff in rtuple[ic].split(' '):
                                    token_attributes_filtered.append(ic+'__' + ff)
                    filter_features = token_attributes_filtered

                pair_df = certa.utils.get_row(pd.Series(ltuple), pd.Series(rtuple), lprefix='', rprefix='')
                '''def predict_proba(lt, rt):
                    res = np.zeros(2)
                    res[self.delegate.predict_tuples(lt, rt)].add(1)
                    return res
'''
                exclude_attrs = []
                if 'attribute' == self.explanation_granularity:
                    exclude_attrs = filter_features
                exclude_tokens = []
                if 'token' == self.explanation_granularity:
                    exclude_tokens = [ta.split("__")[1] for ta in filter_features]


                try:
                    land_explanation = Landmark(self.delegate, pair_df, exclude_attrs=exclude_attrs, exclude_tokens=exclude_tokens).explain_instance(pair_df)
                    if 'token' == self.explanation_granularity:
                        ld = dict()
                        for i in range(len(land_explanation)):
                            row = land_explanation.iloc[i]
                            att = row['column']
                            if att not in ['id', 'rtable_id', 'ltable_id']:
                                ld[att + '__' + row['word']] = float(row['impact'])

                        saliency_df = ld
                    else:
                        saliency_df = land_explanation.groupby('column')['impact'].sum().to_dict()

                    #saliency_df = lemon.explain(self.lsource, self.rsource, pair_df, predict_proba)
                    cfs = self.minun.explain((list(ltuple.values()), list(rtuple.values()), prediction, list(ltuple.keys())+list(rtuple.keys())))

                    if len(saliency_df) > 0:
                        saliency_explanation = saliency_df
                        aggregated_pn = 0
                        for sev in saliency_explanation.values():
                            if type(sev) == list:
                                aggregated_pn += sev[0]
                            else:
                                aggregated_pn += sev
                        if aggregated_pn >= 0.1 and len(cfs) > 0 and max(cfs[0].to_dict().values()) > 0:
                            satisfied = True
                except:
                    pass
            top_k += 2
            its +=1
            if satisfied or top_k == no_features or its == 10:
                top_k -= 1
                break
        return {"prediction": prediction, "saliency": saliency_explanation, "cf": cf_explanation,
                "filter_features": filter_features, "self_explanations": pae_dicts, "top_k": top_k, "iterations": its}

    def count_predictions(self):
        return self.delegate.count_predictions()

    def count_tokens(self):
        return self.delegate.count_tokens()