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

    def __init__(self, model_type='azure_openai', temperature=0.01, fake=False, model_name="",
                 verbose=False, delegate=None, explanation_granularity="attribute", explainer_fn="self", prompts=None,
                 deployment_name="", model_version="2023-05-15"):
        self.fake = fake
        self.model_type = model_type
        if model_type == 'hf':
            if model_name.startswith('https://'):
                llm = HuggingFaceEndpoint(endpoint_url=model_name, task="text-generation",
                                          temperature=temperature, max_new_tokens=1024)
            else:
                llm = HuggingFaceEndpoint(repo_id=model_name, task="text-generation",
                                 temperature= temperature, max_new_tokens= 1024, provider="auto")
            self.llm = ChatHuggingFace(llm=llm, token=True)
        elif model_type == 'openai':
            self.llm = OpenAI(temperature=temperature, model_name=model_name)
        elif model_type == 'azure_openai':
            self.llm = AzureChatOpenAI(model_name=model_name, request_timeout=120,
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

    @staticmethod
    def _strip_response(content: str) -> str:
        if not content:
            return content
        if '[/INST]' in content:
            content = content.split('[/INST]')[-1]
        elif '<|end_header_id|>' in content:
            content = content.split('<|end_header_id|>')[-1]
        return content.strip() if isinstance(content, str) else content

    def _invoke(self, template, _remote_timings=None, **kwargs) -> str:
        """Call LLM (chain.predict or llm.invoke). If _remote_timings is a list, append elapsed seconds for remote-only time."""
        t0 = time()
        if self.model_type in ['falcon', 'llama2']:
            chain = LLMChain(llm=self.llm, prompt=template)
            out = chain.predict(**kwargs)
        else:
            messages = template.format_messages(**kwargs)
            raw = self.llm.invoke(messages)
            out = getattr(raw, 'content', raw) if raw else ''
        if _remote_timings is not None:
            _remote_timings.append(time() - t0)
        if self.model_type == 'hf':
            return self._strip_response(out)
        return self._strip_response(out) if out else out

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
            er_kwargs = dict(ltuple=ltuple, rtuple=rtuple)
            if self.model_type != 'falcon' and self.model_type != 'llama2':
                er_kwargs['feature'] = self.explanation_granularity
            er_answer = self._invoke(template, **er_kwargs)

            # parse answer into prediction
            _, prediction = ellmer.utils.text_to_match(er_answer, self.llm)
            self.pred_count += 1
            self.tokens += sum([len(m[1].split(' ')) for m in conversation])  # input tokens
            self.tokens += len(str(ltuple).split(' ')) + len(str(rtuple).split(' '))
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
        remote_timings = []
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
            content = self._invoke(template, _remote_timings=remote_timings, ltuple=ltuple, rtuple=rtuple, feature=self.explanation_granularity)
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
            self.tokens += len(str(ltuple).split(' ')) + len(str(rtuple).split(' '))
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
                    "conversation": conversation, "llm_time": sum(remote_timings)}
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
            er_kwargs = dict(ltuple=ltuple, rtuple=rtuple)
            if self.model_type not in ['falcon', 'llama2']:
                er_kwargs['feature'] = self.explanation_granularity
            er_answer = self._invoke(template, _remote_timings=remote_timings, **er_kwargs)
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
                conversation.append(("assistant", str(prediction)))
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
                why_kwargs = dict(ltuple=ltuple, rtuple=rtuple, prediction=prediction)
                if self.model_type not in ['falcon', 'llama2']:
                    why_kwargs['feature'] = self.explanation_granularity
                why_answer = self._invoke(template, _remote_timings=remote_timings, **why_kwargs)
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
                saliency_answer = self._invoke(
                    template, _remote_timings=remote_timings,
                    ltuple=ltuple, rtuple=rtuple, prediction=prediction,
                    feature=self.explanation_granularity,
                )
                if self.verbose:
                    print(saliency_answer)
                    parse_t = time()
                saliency_explanation = dict()
                try:
                    saliency_content = saliency_answer
                    try:
                        saliency_content = saliency_answer.split('```')[1].replace('`', '').replace('´', '').strip()
                    except Exception:
                        if '```' in saliency_answer:
                            start_index = saliency_answer.index('```')
                            saliency_content = saliency_answer[
                                               start_index + 3:saliency_answer.index('```', start_index + 3)]
                    if not saliency_content and "{" in (saliency_answer or ""):
                        saliency_content = saliency_answer[saliency_answer.index("{"):saliency_answer.rfind("}") + 1]
                    saliency_dict = json.loads(saliency_content) if saliency_content else {}
                    if 'saliency_explanation' in saliency_dict:
                        saliency_explanation = saliency_dict['saliency_explanation']
                    else:
                        saliency_explanation = saliency_dict
                except Exception:
                    try:
                        saliency = saliency_answer[saliency_answer.index("{"):saliency_answer.rfind("}") + 1]
                        saliency_dict = json.loads(saliency)
                        if 'saliency_explanation' in saliency_dict:
                            saliency_explanation = saliency_dict['saliency_explanation']
                        else:
                            saliency_explanation = saliency_dict
                    except Exception:
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
                cf_answer = self._invoke(
                    template, _remote_timings=remote_timings,
                    ltuple=ltuple, rtuple=rtuple, prediction=prediction,
                    feature=self.explanation_granularity,
                )
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
                    try:
                        cf_dict = json.loads(cf_answer_json)
                    except Exception:
                        cf_dict = ast.literal_eval(cf_answer_json)
                    keys = list(cf_dict.keys()) if hasattr(cf_dict, 'keys') else []
                    if "record_after" in keys:
                        cf_explanation = cf_dict["record_after"]
                        if list(cf_explanation.keys())[0].startswith('rtable_'):
                            cf_explanation = cf_explanation | ast.literal_eval(ltuple)
                        else:
                            cf_explanation = cf_explanation | ast.literal_eval(rtuple)
                    elif "counterfactual_record" in keys:
                        cf_explanation = cf_dict["counterfactual_record"]
                    elif "counterfactual_explanation" in keys:
                        cf_explanation = cf_dict["counterfactual_explanation"]
                    elif "counterfactual" in keys:
                        cf_explanation = cf_dict['counterfactual']
                    elif "record1" in keys and "record2" in keys:
                        for k in list(cf_dict['record1'].keys()):
                            if not k.startswith('ltable_'):
                                cf_dict['record1']['ltable_' + k] = cf_dict['record1'][k]
                                cf_dict['record1'].pop(k)

                        for k in list(cf_dict['record2'].keys()):
                            if not k.startswith('rtable_'):
                                cf_dict['record2']['rtable_' + k] = cf_dict['record2'][k]
                                cf_dict['record2'].pop(k)
                        cf_explanation = cf_dict['record1'] | cf_dict['record2']
                    else:
                        cf_explanation = cf_dict
                except Exception:
                    pass
                if self.verbose:
                    parse_t = time() - parse_t
                    print(f'cf_parse_time:{parse_t}')
                conversation.append(("assistant", str(cf_explanation)))
                self.pred_count += 1

            self.tokens += sum([len(m[1].split(' ')) for m in conversation])
            self.tokens += len(str(ltuple).split(' ')) + len(str(rtuple).split(' '))
            try:
                saliency_explanation = dict([(x[0], x[1]['saliency']) for x in list(saliency_explanation.items())])
            except:
                try:
                    saliency_explanation = dict(
                        [(x[0], x[1]['saliency_score']) for x in list(saliency_explanation.items())])
                except:
                    pass
            return {"prediction": prediction, "why": why, "saliency": saliency_explanation, "cf": cf_explanation,
                    "conversation": conversation, "llm_time": sum(remote_timings)}


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
                    if 'prediction' in json_answer and 'saliency_explanation' in json_answer and 'counterfactual_explanation' in json_answer:
                        return json_answer['prediction'], json_answer['saliency_explanation'], json_answer['counterfactual_explanation']
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
        t0 = time()
        answer = chain.invoke({"input": formatted_question.replace('"', '').replace("'",'')})
        llm_time = time() - t0

        conversation = [str(m) for m in final_prompt.messages]
        conversation.append(formatted_question)
        self.tokens += sum([len(str(m).split(' ')) for m in final_prompt.messages])  # input tokens
        self.tokens += len(str(ltuple).split(' ')) + len(str(rtuple).split(' '))
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

        return {"prediction": prediction, "saliency": saliency, "cf": cf, "conversation": conversation, "llm_time": llm_time}
