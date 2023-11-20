import traceback

from langchain.prompts import ChatPromptTemplate
from langchain import PromptTemplate, HuggingFaceHub, OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import HuggingFacePipeline, LlamaCpp
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

hf_models = ['EleutherAI/gpt-neox-20b', 'tiiuae/falcon-7b-instruct', "Writer/camel-5b-hf", "databricks/dolly-v2-3b",
             "google/flan-t5-xxl", "tiiuae/falcon-40b", "tiiuae/falcon-7b", "internlm/internlm-chat-7b", "Qwen/Qwen-7B"]

openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")


class Ellmer:

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


class CertaEllmer(Ellmer):

    def __init__(self, explanation_granularity, delegate, certa, num_triangles=10):
        self.explanation_granularity = explanation_granularity
        self.certa = certa
        self.num_triangles = num_triangles
        self.delegate = delegate
        self.predict_fn = lambda x: self.delegate.predict(x)

    def predict_and_explain(self, ltuple, rtuple, max_predict: int = 40, verbose: bool = False):
        pae = self.delegate.predict_tuples(ltuple, rtuple)
        prediction = pae
        saliency_explanation = None
        cf_explanation = None
        if pae is not None:
            ltuple_series = self.get_row(ltuple, self.certa.lsource, prefix="ltable_")
            rtuple_series = self.get_row(rtuple, self.certa.rsource, prefix="rtable_")
            saliency_df, cf_summary, cfs, tri, _ = self.certa.explain(ltuple_series, rtuple_series, self.predict_fn,
                                                                      token="token" == self.explanation_granularity,
                                                                      num_triangles=self.num_triangles, verbose=verbose,
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


class GenericEllmer(Ellmer):

    def __init__(self, model_type='azure_openai', temperature=0.01, max_length=512, fake=False, model_name="",
                 verbose=False, delegate=None, explanation_granularity="attribute", explainer_fn="self", prompts=None,
                 deployment_name="", model_version="2023-05-15"):
        self.fake = fake
        self.model_type = model_type
        if model_type == 'hf':
            self.llm = HuggingFaceHub(repo_id=model_name,
                                      model_kwargs={'temperature': temperature, 'max_length': max_length})
        elif model_type == 'openai':
            self.llm = OpenAI(temperature=temperature, model_name=model_name)
        elif model_type == 'azure_openai':
            self.llm = AzureChatOpenAI(deployment_name=deployment_name, model_name=model_name,
                                       openai_api_version=model_version, temperature=temperature)
        elif model_type == 'delegate':
            self.llm = delegate
        elif model_type == 'falcon':
            self.llm = falcon_pipeline(model_id=model_name)
        elif model_type == 'llama2':
            self.llm = llama2_llm(verbose=verbose)
        self.verbose = verbose
        self.explanation_granularity = explanation_granularity
        if "self" == explainer_fn:
            self.explainer_fn = "self"
        else:
            self.explainer_fn = explainer_fn
        self.prompts = prompts

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
            if "hf" == self.model_type:
                chain = LLMChain(llm=self.llm, prompt=template)
                er_answer = chain.predict(ltuple=ltuple, rtuple=rtuple)
            else:
                messages = template.format_messages(feature=self.explanation_granularity, ltuple=ltuple, rtuple=rtuple)
                answer = self.llm(messages)
                er_answer = answer.content

            # parse answer into prediction
            _, prediction = ellmer.utils.text_to_match(er_answer, self.llm)
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
            prompt = self.prompts['pase']
            for prompt_message in ellmer.utils.read_prompt(prompt):
                conversation.append((prompt_message[0], prompt_message[1]))
            question = "record1:\n{ltuple}\n record2:\n{rtuple}\n"
            conversation.append(("user", question))
            template = ChatPromptTemplate.from_messages(conversation)
            if self.model_type in ['hf', 'falcon', 'llama2']:
                chain = LLMChain(llm=self.llm, prompt=template)
                content = chain.predict(ltuple=ltuple, rtuple=rtuple, feature=self.explanation_granularity)
            else:
                messages = template.format_messages(feature=self.explanation_granularity, ltuple=ltuple, rtuple=rtuple)
                answer = self.llm(messages)
                content = answer.content
            if self.verbose:
                print(content)
            prediction, saliency_explanation, cf_explanation = parse_pase_answer(content, self.llm)
            if prediction is None:
                print(f'empty prediction!\nquestion{question}\nconversation{conversation}')
            return {"prediction": prediction, "saliency": saliency_explanation, "cf": cf_explanation}
        elif "ptse" in self.prompts:
            ptse_prompts = self.prompts["ptse"]
            er_prompt = ptse_prompts['er']
            for prompt_message in ellmer.utils.read_prompt(er_prompt):
                conversation.append((prompt_message[0], prompt_message[1]))
            question = "record1:\n{ltuple}\n record2:\n{rtuple}\n"
            conversation.append(("user", question))
            template = ChatPromptTemplate.from_messages(conversation)
            if self.model_type in ['hf', 'falcon', 'llama2']:
                chain = LLMChain(llm=self.llm, prompt=template)
                er_answer = chain.predict(ltuple=ltuple, rtuple=rtuple)
            else:
                messages = template.format_messages(feature=self.explanation_granularity, ltuple=ltuple, rtuple=rtuple)
                answer = self.llm(messages)
                er_answer = answer.content
            if self.verbose:
                print(er_answer)
            # parse answer into prediction
            _, prediction = ellmer.utils.text_to_match(er_answer, self.llm)
            conversation.append(("assistant", er_answer))

            why = None
            saliency_explanation = None
            cf_explanation = None

            # get explanations
            if "why" in ptse_prompts:
                for prompt_message in ellmer.utils.read_prompt(ptse_prompts["why"]):
                    conversation.append((prompt_message[0], prompt_message[1]))
                template = ChatPromptTemplate.from_messages(conversation)
                if self.model_type in ['hf', 'falcon', 'llama2']:
                    chain = LLMChain(llm=self.llm, prompt=template)
                    why_answer = chain.predict(ltuple=ltuple, rtuple=rtuple, prediction=prediction)
                else:
                    messages = template.format_messages(feature=self.explanation_granularity, ltuple=ltuple,
                                                        rtuple=rtuple,
                                                        prediction=prediction)

                    answer = self.llm(messages)
                    why_answer = answer.content
                if self.verbose:
                    print(why_answer)
                why = why_answer
                conversation.append(("assistant", why_answer))

            # saliency explanation
            if "saliency" in ptse_prompts:
                for prompt_message in ellmer.utils.read_prompt(ptse_prompts["saliency"]):
                    conversation.append((prompt_message[0], prompt_message[1]))
                template = ChatPromptTemplate.from_messages(conversation)
                if self.model_type in ['hf', 'falcon', 'llama2']:
                    chain = LLMChain(llm=self.llm, prompt=template)
                    saliency_answer = chain.predict(ltuple=ltuple, rtuple=rtuple, prediction=prediction,
                                                    feature=self.explanation_granularity)
                else:
                    messages = template.format_messages(feature=self.explanation_granularity, ltuple=ltuple,
                                                        rtuple=rtuple,
                                                        prediction=prediction)
                    answer = self.llm(messages)
                    saliency_answer = answer.content
                if self.verbose:
                    print(saliency_answer)
                saliency_explanation = dict()
                try:
                    saliency = saliency_answer.replace('`', '').split('```')[1]
                    saliency_dict = json.loads(saliency)
                    saliency_explanation = saliency_dict
                except:
                    try:
                        saliency = saliency_answer[saliency_answer.index("{"):saliency_answer.rfind("}") + 1]
                        saliency_dict = json.loads(saliency)
                        saliency_explanation = saliency_dict
                    except:
                        pass

                conversation.append(("assistant", "{" + str(saliency_explanation) + "}"))

            # counterfactual explanation
            if "cf" in ptse_prompts:
                for prompt_message in ellmer.utils.read_prompt(ptse_prompts["cf"]):
                    conversation.append((prompt_message[0], prompt_message[1]))
                template = ChatPromptTemplate.from_messages(conversation)

                if self.model_type in ['hf', 'falcon', 'llama2']:
                    chain = LLMChain(llm=self.llm, prompt=template)
                    cf_answer = chain.predict(ltuple=ltuple, rtuple=rtuple, prediction=prediction,
                                              feature=self.explanation_granularity)
                    print(cf_answer)
                else:
                    messages = template.format_messages(feature=self.explanation_granularity, ltuple=ltuple,
                                                        rtuple=rtuple,
                                                        prediction=prediction)

                    answer = self.llm(messages)
                    cf_answer = answer.content
                if self.verbose:
                    print(cf_answer)
                cf_explanation = dict()
                try:
                    cf_answer_content = cf_answer.replace('`', '')
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
                conversation.append(("assistant", str(cf_explanation)))
            return {"prediction": prediction, "why": why, "saliency": saliency_explanation, "cf": cf_explanation}


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
            answer = json.loads(answer)
            if "matching" in answer.keys():
                prediction = answer['matching']
            elif "matching_prediction" in answer.keys():
                prediction = answer['matching_prediction']
            elif "match" in answer.keys():
                prediction = answer['match']
            elif "prediction" in answer.keys():
                prediction = answer['prediction']
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
            except:
                pass
            try:
                if "counterfactual_explanation" in answer.keys():
                    cf = answer['counterfactual_explanation']
                elif "counterfactual_explanation_table" in answer.keys():
                    cf = answer['counterfactual_explanation_table']
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


class Certa(Ellmer):

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

        saliency_df, cf_summary, cfs, tri, _ = self.certa.explain(ltuple_series, rtuple_series, self.predict_fn,
                                                                  token="token" == self.explanation_granularity,
                                                                  num_triangles=self.num_triangles, max_predict=100)
        saliency = saliency_df.to_dict('list')
        if len(cfs) > 0:
            counterfactuals = [cfs.drop(
                ['alteredAttributes', 'droppedValues', 'copiedValues', 'triangle', 'attr_count'], axis=1).T.to_dict()]
        else:
            counterfactuals = [{}]
        return matching, saliency, counterfactuals


class PASE(Ellmer):

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


class PTSE(Ellmer):

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
                json_answer = json.loads(answer[1])
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


def falcon_pipeline(model_id="vilsonrodrigues/falcon-7b-instruct-sharded"):
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
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    fpip = pipeline(
        "text-generation",
        model=model_4bit,
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
