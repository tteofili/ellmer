import ast
import json
import openai
import os
from langchain import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from time import time

import ellmer.utils
from ellmer.explainer import BaseLLMExplainer, falcon_pipeline, llama2_llm

openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")


class SelfExplainer(BaseLLMExplainer):

    def __init__(self, model_type='azure_openai', temperature=0.01, max_length=512, fake=False, model_name="",
                 verbose=False, delegate=None, explanation_granularity="attribute", explainer_fn="self", prompts=None,
                 deployment_name="", model_version="2023-05-15"):
        self.fake = fake
        self.model_type = model_type
        if model_type == 'hf':
            llm = HuggingFaceEndpoint(repo_id=model_name, task="text-generation",
                                 temperature= temperature, max_new_tokens= 1024)
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

    def predict_tuples(self, ltuple, rtuple, append_conversation=None):
        conversation = []
        if "ptse" in self.prompts:
            ptse_prompts = self.prompts["ptse"]
            er_prompt = ptse_prompts['er']
            for prompt_message in ellmer.utils.read_prompt(er_prompt):
                conversation.append((prompt_message[0], prompt_message[1]))
            if append_conversation is not None:
                conversation.extend(append_conversation)
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
                if '[/INST]' in er_answer:
                    try:
                        er_answer = er_answer.split('[/INST]')[-1]
                    except:
                        pass
                elif "<|end_header_id|>" in er_answer:
                    try:
                        er_answer = er_answer.split("<|end_header_id|>")[-1]
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

            if len(er_answer) > 5:
                conversation.append(("assistant", prediction))
            else:
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
                    if '[/INST]' in why_answer:
                        try:
                            why_answer = why_answer.split('[/INST]')[-1]
                        except:
                            pass
                    elif "<|end_header_id|>" in why_answer:
                        try:
                            why_answer = why_answer.split("<|end_header_id|>")[-1]
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
                    if '[/INST]' in saliency_answer:
                        try:
                            saliency_answer = saliency_answer.split('[/INST]')[-1]
                        except:
                            pass
                    elif "<|end_header_id|>" in saliency_answer:
                        try:
                            saliency_answer = saliency_answer.split("<|end_header_id|>")[-1]
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
                    if '[/INST]' in cf_answer:
                        try:
                            cf_answer = cf_answer.split('[/INST]')[-1]
                        except:
                            pass
                    elif "<|end_header_id|>" in cf_answer:
                        try:
                            cf_answer = cf_answer.split("<|end_header_id|>")[-1]
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
    if type(answer) != str:
        answer = str(answer)

    matching = 0
    saliency = dict()
    cf = dict()

    try:
        prediction, saliency, cf = ellmer.utils.text_to_data(answer, llm)
        if prediction is not None and saliency is not None and cf is not None:
            return prediction, saliency, cf
    except:
        pass

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
                    json_answer = json.loads(a.replace('json', ''))
                    return parse_pase_answer(json_answer, llm)
                except:
                    if '}}' in a:
                        return parse_pase_answer(a, llm)
            if json_answer is None:
                for a in split:
                    nm, ns, ncf = parse_pase_answer(a, llm)
                    try:
                        if nm is not None and len(ns) is not 0 and len(cf) is not 0:
                            return nm, ns, ncf
                    except:
                        pass
            return "0", "{}", "{}"
        elif "\n\n{" in answer and "}\n\n" in answer:
            answer = '{' + ''.join(answer.split("\n\n{")[1].split("}\n\n")[0]) + '}'

        # decode the json content
        try:
            answer = answer.replace('´', '').replace('`', '')
            try:
                answer = json.loads(answer)
            except:
                answer = json.loads(answer[:len(answer) - 1])
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
            elif 'is_match' in answer.keys():
                prediction = answer['is_match']
            else:
                print(f"cannot find 'matching' key in {answer}")
                prediction = None
            if prediction is not None:
                prediction = str(prediction).strip()
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
                elif "counterfactual" in answer.keys():
                    cf = answer['counterfactual']
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


class ICLSelfExplainer(SelfExplainer):

    def __init__(self, examples, **kwargs):
        SelfExplainer.__init__(self, **kwargs)
        self.examples = examples

    def predict_and_explain(self, ltuple, rtuple):
        fs_prompts = self.prompts["fs"]
        fs_conversation = []
        for prompt_message in ellmer.utils.read_prompt(fs_prompts):
            fs_conversation.append((prompt_message[0], prompt_message[1]))
        example_prompt = ChatPromptTemplate.from_messages(fs_conversation)
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
        chain = final_prompt | self.llm
        question = self.prompts['input']
        formatted_question = question.format(ltuple=ltuple, rtuple=rtuple)
        answer = chain.invoke({"input": formatted_question.replace('"', '').replace("'",'')})

        conversation = [str(m) for m in final_prompt.messages]
        conversation.append(formatted_question)
        if self.verbose:
            self.tokens += sum([len(str(m).split(' ')) for m in final_prompt.messages])  # input tokens
        answer_content = answer.content
        conversation.append(answer_content)
        if self.verbose:
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
