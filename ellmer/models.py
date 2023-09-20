from langchain import PromptTemplate, HuggingFaceHub, OpenAI
import random
from certa.models.ermodel import ERModel
import numpy as np
import pandas as pd
import ellmer.utils
import openai
import os
import json

hf_models = ['EleutherAI/gpt-neox-20b', 'tiiuae/falcon-7b-instruct']

openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")


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
                answer = self.llm(question)
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
    def __init__(self, explanation_granularity: str = "attribute", why: bool = False):
        self.explanation_granularity = explanation_granularity
        self.why = why

    def er(self, ltuple: str, rtuple: str, temperature=0.99):
        question = "record1:\n" + ltuple + "\n record2:\n" + rtuple + "\n"
        return self.__call__(question, er=True, explanation=False, temperature=temperature)

    def explain(self, ltuple: str, rtuple: str, prediction: str, temperature=0.99, *args, **kwargs):
        question = "record1:\n" + ltuple + "\n record2:\n" + rtuple + "\n"
        return self.__call__(question, er=True, explanation=True, prediction=prediction, temperature=temperature,
                             *args, **kwargs)

    def __call__(self, question, er: bool = False, saliency: bool = False, cf: bool = False,
                 explanation=False, prediction=None, temperature=0.99, *args, **kwargs):
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
                messages=conversation, temperature=temperature
            )
            prediction = response["choices"][0]["message"]["content"]
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
                    messages=conversation, temperature=temperature
                )["choices"][0]["message"]["content"]
                answers['nl_exp'] = nl_exp
                conversation.append({"role": "assistant", "content": nl_exp})

            # saliency explanation
            for prompt_message in ellmer.utils.read_prompt('ellmer/prompts/er-saliency.txt'):
                conversation.append(
                    {"role": prompt_message[0],
                     "content": prompt_message[1].replace("feature", self.explanation_granularity)})
            saliency_exp = ellmer.utils.completion_with_backoff(
                deployment_id="gpt-35-turbo", model="gpt-3.5-turbo",
                messages=conversation, temperature=temperature
            )["choices"][0]["message"]["content"]
            answers['saliency_exp'] = saliency_exp
            conversation.append({"role": "assistant", "content": saliency_exp})

            # counterfactual explanation
            for prompt_message in ellmer.utils.read_prompt('ellmer/prompts/er-cf.txt'):
                conversation.append(
                    {"role": prompt_message[0],
                     "content": prompt_message[1].replace("feature", self.explanation_granularity)})
            cf_exp = ellmer.utils.completion_with_backoff(
                deployment_id="gpt-35-turbo", model="gpt-3.5-turbo",
                messages=conversation, temperature=temperature
            )["choices"][0]["message"]["content"]
            answers['cf_exp'] = cf_exp
        return answers


class PredictAndSelfExplainER:

    def __init__(self, explanation_granularity: str = "attribute"):
        self.explanation_granularity = explanation_granularity

    def er(self, ltuple: str, rtuple: str, temperature=0.99):
        question = "record1:\n" + ltuple + "\n record2:\n" + rtuple + "\n"
        return self.__call__(question, er=True, temperature=temperature)

    def __call__(self, question, er: bool = False, temperature=0.99, *args, **kwargs):
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
            messages=conversation, temperature=temperature
        )
        try:
            answer = response["choices"][0]["message"]["content"]
        except:
            answer = response["choices"][0]["message"]
        return answer


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
