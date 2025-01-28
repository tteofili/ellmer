import traceback
import operator
from collections import Counter
from langchain.prompts import ChatPromptTemplate, FewShotPromptTemplate
from langchain import PromptTemplate, HuggingFaceHub, OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_core.prompts import FewShotChatMessagePromptTemplate

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
        labels = int(data_df['label'].astype(float).values)
        return f1_score(y_true=labels, y_pred=predictions)

    def count_predictions(self):
        return self.pred_count

    def count_tokens(self):
        return self.tokens


class FullCerta(BaseLLMExplainer):

    def __init__(self, explanation_granularity, delegate, certa, num_triangles=10, max_predict=-1):
        self.explanation_granularity = explanation_granularity
        self.certa = certa
        self.num_triangles = num_triangles
        self.delegate = delegate
        self.predict_fn = lambda x: self.delegate.predict(x)
        self.max_predict = max_predict

    def predict_and_explain(self, ltuple, rtuple, verbose: bool = False):
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
                                                                       max_predict=self.max_predict, llm=self.delegate.llm)
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
                filtered_df = df.copy()
                for c in df.columns:
                    if c in filtered_df.columns and c in rc:
                        new_filtered_df = filtered_df.loc[filtered_df[c].isin(rc[c])]
                        if len(new_filtered_df) == 1:
                            return new_filtered_df.iloc[0]
                        elif len(new_filtered_df) > 0:
                            filtered_df = new_filtered_df
                if len(filtered_df) > 0:
                    return filtered_df.drop_duplicates().iloc[0]
        else:
            return result.iloc[0]

    def record_to_text(self, record, ignored_columns=['id', 'ltable_id', 'rtable_id', 'label']):
        return " ".join([str(val) for k, val in record.to_dict().items() if k not in [ignored_columns]])

    def cs(self, text1, text2):
        WORD = re.compile(r'\w+')
        vec1 = Counter(WORD.findall(text1))
        vec2 = Counter(WORD.findall(text2))
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
        sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    def count_predictions(self):
        return self.delegate.count_predictions()

    def count_tokens(self):
        return self.delegate.count_tokens()


class HybridCerta(FullCerta):
    def __init__(self, explanation_granularity, pred_delegate, certa, ellmers, num_draws=1, num_triangles=10,
                 combine: str = 'freq', top_k: int = -1):
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
            elif self.combine == 'random':
                filter_features = set()
                for se in pae_dicts:
                    for _ in range(top_k):
                        filter_features.add(random.choice(se)[0])
                filter_features = list(filter_features)
            else:
                raise ValueError("Unknown combination method")
            if len(filter_features) > 0:
                if 'token' == self.explanation_granularity:
                    token_attributes_filtered = []
                    for ff in filter_features:
                        if ff.startswith('ltable_'):
                            continue
                        if ff.startswith('rtable_'):
                            continue
                        else:
                            for ic in ltuple.keys():
                                if ff in ltuple[ic].split(' '):
                                    token_attributes_filtered.append(ic + '__' + ff)
                            for ic in rtuple.keys():
                                if ff in rtuple[ic].split(' '):
                                    token_attributes_filtered.append(ic + '__' + ff)
                    filter_features = token_attributes_filtered
                # regenerate support_samples, when empty
                if support_samples is not None and len(support_samples) == 0:
                    support_samples = None
                    num_triangles *= 2
                saliency_df, cf_summary, cfs, tri, _, support_samples = self.certa.explain(ltuple_series, rtuple_series,
                                                                                           self.predict_fn,
                                                                                           token="token" == self.explanation_granularity,
                                                                                           num_triangles=num_triangles,
                                                                                           max_predict=max_predict,
                                                                                           filter_features=filter_features,
                                                                                           support_samples=support_samples)
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
            its += 1
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
            #question = "record1:\n{ltuple}\n record2:\n{rtuple}\n"
            question = "record1: {ltuple}\n  record2: {rtuple}"
            conversation.append(("user", question))
            template = ChatPromptTemplate.from_messages(conversation)
            if self.model_type in ['falcon', 'llama2']:
                chain = LLMChain(llm=self.llm, prompt=template)
                er_answer = chain.predict(ltuple=ltuple, rtuple=rtuple)
            elif self.model_type in ['hf']:
                messages = template.format_messages(ltuple=ltuple, rtuple=rtuple, feature=self.explanation_granularity,)
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
            question = "record1: {ltuple} \nrecord2: {rtuple}"
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
                    saliency_explanation = dict(
                        [(x[0], x[1]['saliency_score']) for x in list(saliency_explanation.items())])
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
            question = "record1: {ltuple} \nrecord2: {rtuple}"
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
                                                        rtuple=rtuple, prediction=prediction)
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
                            saliency_content = saliency_answer[
                                               start_index + 3:saliency_answer.index('```', start_index + 3)]
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
                conversation.append(("assistant", json.dumps(saliency_explanation).replace('{', '').replace('}', '')))
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
                    saliency_explanation = dict(
                        [(x[0], x[1]['saliency_score']) for x in list(saliency_explanation.items())])
                except:
                    pass
            return {"prediction": prediction, "why": why, "saliency": saliency_explanation, "cf": cf_explanation,
                    "conversation": conversation}


def parse_pase_answer(answer, llm):
    matching = 0
    saliency = dict()
    cf = dict()

    original_answer = answer
    try:
        # find the json content
        split = answer.split('```')
        if len(split) == 2:
            answer = split[1]
            if answer.startswith('json'):
                answer = answer[4:]
        elif len(split) > 1:
            json_answer = None
            for a in split:
                try:
                    json_answer = json.loads(a)
                    return parse_pase_answer(json_answer, llm)
                except:
                    if '}}' in a:
                        return parse_pase_answer(a, llm)
            if json_answer is None:
                for a in split:
                    nm, ns, ncf = parse_pase_answer(a, llm)
                    try:
                        if nm is not None and ns is not None and cf is not None:
                            return nm, ns, ncf
                    except:
                        pass
            return "0", "{}", "{}"
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
            if prediction is not None:
                if prediction.lower() == 'yes' or prediction.lower() == 'true' or prediction.lower() == '1':
                    matching = 1
                elif not prediction:
                    matching = 0
            else:
                _, matching = ellmer.utils.text_to_match(original_answer, llm)
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


class ICLSelfExplainer(SelfExplainer):

    def __init__(self, examples, **kwargs):
        SelfExplainer.__init__(self, **kwargs)
        self.examples = examples

    def predict_and_explain(self, ltuple, rtuple):
        conversation = []

        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "prediction:{prediction}, reason:{why}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=self.examples,
        )
        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert assistant for Entity Resolution tasks."),
                few_shot_prompt,
                ("human", "{input}"),
            ]
        )
        conversation = final_prompt.messages
        chain = final_prompt | self.llm
        question = self.prompts['input']
        formatted_question = question.format(ltuple=ltuple, rtuple=rtuple)
        conversation.append(formatted_question)
        self.tokens += sum([len(str(m).split(' ')) for m in final_prompt.messages])  # input tokens
        answer = chain.invoke({"input": formatted_question})
        answer_content = answer.content
        conversation.append(answer_content)
        self.tokens += len(answer.content.split(' '))  # output tokens
        prediction = "0"
        saliency = {}
        cf = {}
        try:
            if "prediction:" in answer_content:
                p_start = answer_content.find("prediction:") + len("prediction:")
                p_end = answer_content.find(",", p_start)
                prediction = answer_content[p_start:p_end]
            else:
                _, prediction = ellmer.utils.text_to_match(answer_content, self.llm)
        except:
            pass
        if prediction not in ["0", "1", 0, 1]:
            _, prediction = ellmer.utils.text_to_match(answer_content, self.llm)
        try:
            s_start = answer_content.find("saliency:") + len("saliency:")
            s_end = answer_content.find("}", s_start) + 1
            saliency = answer_content[s_start:s_end].replace("'", "\"")
            saliency = json.loads(saliency)
            ns = dict()
            for k, v in saliency.items():
                if type(v) == list:
                    ns[k] = v[0]
                else:
                    ns[k] = v
            saliency = ns
        except:
            pass
        try:
            cf_start = answer_content.find("counterfactual:") + len("counterfactual:")
            cf_end = answer_content.find("}", cf_start) + 1
            cf = answer_content[cf_start:cf_end].replace("'", "\"")
            cf = json.loads(cf)
        except:
            pass
        self.pred_count += 1

        return {"prediction": prediction, "saliency": saliency, "cf": cf, "conversation": conversation}


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

    def __init__(self, model_type='azure_openai', temperature=0.01, max_length=512, fake=False,
                 hf_repo="tiiuae/falcon-7b",
                 verbose=False, delegate=None):
        template = "given the record:\n{ltuple}\n and the record:\n{rtuple}\n do they refer to the same entity in the real world?\nreply yes or no"
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


class HybridGeneric(BaseLLMExplainer):

    def __init__(self, saliency_fn, cf_fn, explanation_granularity, pred_delegate, ellmers, lsource, rsource,
                 num_draws=1, num_triangles=10, combine: str = 'freq', top_k: int = 1):
        self.explanation_granularity = explanation_granularity
        self.saliency_fn = saliency_fn
        self.cf_fn = cf_fn
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
                                token_attributes_filtered.append(ff + '__' + token)
                        if ff.startswith('rtable_'):
                            for token in rtuple[ff].split(' '):
                                token_attributes_filtered.append(ff + '__' + token)
                        else:
                            for ic in ltuple.keys():
                                if ff in ltuple[ic].split(' '):
                                    token_attributes_filtered.append(ic + '__' + ff)
                            for ic in rtuple.keys():
                                if ff in rtuple[ic].split(' '):
                                    token_attributes_filtered.append(ic + '__' + ff)
                    filter_features = token_attributes_filtered

                pair_df = certa.utils.get_row(pd.Series(ltuple), pd.Series(rtuple), lprefix='', rprefix='')

                try:
                    saliency_explanation = self.saliency_fn(self.delegate, pair_df, filter)
                    if 'token' == self.explanation_granularity:
                        ld = dict()
                        for i in range(len(saliency_explanation)):
                            row = saliency_explanation.iloc[i]
                            att = row['column']
                            if att not in ['id', 'rtable_id', 'ltable_id']:
                                ld[att + '__' + row['word']] = float(row['impact'])

                        saliency_df = ld
                    else:
                        saliency_df = saliency_explanation.groupby('column')['impact'].sum().to_dict()
                    cfs = self.cf_fn(pair_df, prediction, self.delegate)

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
            its += 1
            if satisfied or top_k == no_features or its == 10:
                top_k -= 1
                break
        return {"prediction": prediction, "saliency": saliency_explanation, "cf": cf_explanation,
                "filter_features": filter_features, "self_explanations": pae_dicts, "top_k": top_k, "iterations": its}

    def count_predictions(self):
        return self.delegate.count_predictions()

    def count_tokens(self):
        return self.delegate.count_tokens()
