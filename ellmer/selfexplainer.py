import ast
import json
import openai
import os
try:
    from langchain.chains import LLMChain
except ImportError:
    from langchain_classic.chains import LLMChain

try:
    from langchain import OpenAI
except ImportError:
    from langchain_community.llms import OpenAI

try:
    from langchain.chat_models import AzureChatOpenAI
except ImportError:
    from langchain_openai import AzureChatOpenAI

try:
    from langchain.prompts import ChatPromptTemplate
except ImportError:
    from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from time import time

import ellmer.utils
from ellmer.base_explainer import BaseLLMExplainer
from ellmer.llm_output_parse import (
    collapse_saliency_token_keys_to_attributes,
    normalize_saliency_dict,
    parse_cf_tsv_or_json,
    parse_prediction_line,
    parse_saliency_response,
    to_numeric_saliency_map,
)

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
            from ellmer.explainer import falcon_pipeline

            self.llm = falcon_pipeline(model_id=model_name)
        elif model_type == 'llama2':
            from ellmer.explainer import llama2_llm

            self.llm = llama2_llm(verbose=verbose, temperature=temperature, quantized_model_path=model_name)
        elif model_type == 'bedrock':
            from ellmer.bedrock_llm import build_chat_bedrock

            # Scripts often pass Azure default deployment_name; only treat as Bedrock profile if set to a non-Azure value.
            dep = (deployment_name or "").strip()
            if dep in {"gpt-35-turbo", "gpt-3.5-turbo", "gpt-4", "gpt-4-32k"}:
                dep = ""
            self.llm = build_chat_bedrock(
                model_name=model_name,
                temperature=temperature,
                inference_profile_id=dep or None,
            )
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
        if self.prompts and "ptse_staged" in self.prompts:
            ps = self.prompts["ptse_staged"]
            er_prompt = ps["er"]
            for prompt_message in ellmer.utils.read_prompt(er_prompt):
                conversation.append((prompt_message[0], prompt_message[1]))
            if append_conversation is not None:
                conversation.extend(append_conversation)
            template = ChatPromptTemplate.from_messages(conversation)
            er_kwargs = dict(ltuple=ltuple, rtuple=rtuple, feature=self.explanation_granularity)
            er_answer = self._invoke(template, **er_kwargs)
            prediction = parse_prediction_line(er_answer, self.llm)
            self.pred_count += 1
            self.tokens += sum([len(m[1].split(" ")) for m in conversation])
            self.tokens += len(str(ltuple).split(" ")) + len(str(rtuple).split(" "))
            self.tokens += len(er_answer.split(" "))
        elif "ptse" in self.prompts:
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
            if self.model_type not in ('falcon', 'llama2'):
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
        if self.prompts and "ptse_staged" in self.prompts:
            out = self._predict_and_explain_ptse_staged(ltuple, rtuple)
            return out
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
            saliency_explanation = normalize_saliency_dict(saliency_explanation)
            if self.explanation_granularity == "attribute":
                saliency_explanation = collapse_saliency_token_keys_to_attributes(saliency_explanation)
            saliency_explanation = to_numeric_saliency_map(saliency_explanation)
            if not isinstance(cf_explanation, dict):
                cf_explanation = parse_cf_tsv_or_json(content, ltuple, rtuple)
            if not isinstance(cf_explanation, dict):
                cf_explanation = {}
            return {"prediction": prediction, "saliency": saliency_explanation, "cf": cf_explanation,
                    "saliency_present": bool(saliency_explanation), "cf_present": bool(cf_explanation),
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
            saliency_explanation = {}
            cf_explanation = {}

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
                saliency_explanation = parse_saliency_response(saliency_answer)
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
                cf_explanation = parse_cf_tsv_or_json(cf_answer, ltuple, rtuple)
                if self.verbose:
                    parse_t = time() - parse_t
                    print(f'cf_parse_time:{parse_t}')
                conversation.append(("assistant", str(cf_explanation)))
                self.pred_count += 1

            self.tokens += sum([len(m[1].split(' ')) for m in conversation])
            self.tokens += len(str(ltuple).split(' ')) + len(str(rtuple).split(' '))
            saliency_explanation = to_numeric_saliency_map(normalize_saliency_dict(saliency_explanation))
            if not isinstance(cf_explanation, dict):
                cf_explanation = {}
            return {"prediction": prediction, "why": why, "saliency": saliency_explanation, "cf": cf_explanation,
                    "saliency_present": bool(saliency_explanation), "cf_present": bool(cf_explanation),
                    "conversation": conversation, "llm_time": sum(remote_timings)}
        return {"prediction": 0, "why": None, "saliency": {}, "cf": {}, "saliency_present": False, "cf_present": False,
                "conversation": conversation, "llm_time": sum(remote_timings)}

    def _predict_and_explain_ptse_staged(self, ltuple, rtuple):
        """Three-call CoT pipeline with TSV-tagged saliency and counterfactual blocks."""
        conversation = []
        remote_timings = []
        ps = self.prompts["ptse_staged"]
        why = None
        saliency_explanation = {}
        cf_explanation = {}

        def _append_prompt_messages(path_key):
            for role, body in ellmer.utils.read_prompt(ps[path_key]):
                conversation.append((role, body))

        # --- Call 1: match ---
        _append_prompt_messages("er")
        template = ChatPromptTemplate.from_messages(conversation)
        er_kwargs = dict(ltuple=ltuple, rtuple=rtuple, feature=self.explanation_granularity)
        er_answer = self._invoke(template, _remote_timings=remote_timings, **er_kwargs)
        conversation.append(("assistant", er_answer))
        prediction = parse_prediction_line(er_answer, self.llm)
        self.pred_count += 1

        pred_int = int(prediction) if prediction in (0, 1) else int(bool(prediction))
        pred_label = "MATCH" if pred_int == 1 else "NON-MATCH"
        staged_kw = dict(
            ltuple=ltuple,
            rtuple=rtuple,
            feature=self.explanation_granularity,
            prediction_int=pred_int,
            prediction_label=pred_label,
        )

        if "why" in ps:
            _append_prompt_messages("why")
            template = ChatPromptTemplate.from_messages(conversation)
            why_answer = self._invoke(template, _remote_timings=remote_timings, **staged_kw)
            why = why_answer
            conversation.append(("assistant", why_answer))
            self.pred_count += 1

        # --- Call 2: saliency ---
        _append_prompt_messages("saliency")
        template = ChatPromptTemplate.from_messages(conversation)
        saliency_answer = self._invoke(template, _remote_timings=remote_timings, **staged_kw)
        conversation.append(("assistant", saliency_answer))
        saliency_explanation = to_numeric_saliency_map(parse_saliency_response(saliency_answer))
        self.pred_count += 1

        # --- Call 3: counterfactual ---
        _append_prompt_messages("cf")
        template = ChatPromptTemplate.from_messages(conversation)
        cf_answer = self._invoke(template, _remote_timings=remote_timings, **staged_kw)
        conversation.append(("assistant", cf_answer))
        cf_explanation = parse_cf_tsv_or_json(cf_answer, ltuple, rtuple)
        self.pred_count += 1

        self.tokens += sum([len(m[1].split(" ")) for m in conversation])
        self.tokens += len(str(ltuple).split(" ")) + len(str(rtuple).split(" "))

        return {
            "prediction": pred_int,
            "why": why,
            "saliency": saliency_explanation,
            "cf": cf_explanation,
            "saliency_present": bool(saliency_explanation),
            "cf_present": bool(cf_explanation),
            "conversation": conversation,
            "llm_time": sum(remote_timings),
        }


def parse_pase_answer(answer, llm):
    """Parse single-shot PASE JSON. Optional repair: set env ``ELLMER_PASE_USE_TEXT_TO_DATA=1`` to run a second LLM call via ``text_to_data`` (legacy, off by default)."""
    if type(answer) != str:
        answer = str(answer)

    matching = 0
    saliency = dict()
    cf = dict()

    _pase_llm_json_repair = os.environ.get("ELLMER_PASE_USE_TEXT_TO_DATA", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if _pase_llm_json_repair:
        try:
            prediction, saliency, cf = ellmer.utils.text_to_data(answer, llm)
            if prediction is not None and saliency is not None and cf is not None:
                return prediction, saliency, cf
        except Exception:
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
                        if nm is not None and len(ns) != 0 and len(cf) != 0:
                            return nm, ns, ncf
                    except:
                        pass
            return 0, {}, {}
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
    if not isinstance(saliency, dict):
        saliency = {}
    if not isinstance(cf, dict):
        cf = {}
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
        saliency = to_numeric_saliency_map(parse_saliency_response(answer_content))
        if not saliency and "saliency:" in answer_content.lower():
            saliency = to_numeric_saliency_map(
                parse_saliency_response(answer_content.replace("saliency:", "SALIENCY_JSON:", 1))
            )
        cf = parse_cf_tsv_or_json(answer_content, ltuple, rtuple)
        if not cf and "counterfactual:" in answer_content.lower():
            cf = parse_cf_tsv_or_json(answer_content.replace("counterfactual:", "CF_JSON:", 1), ltuple, rtuple)
        self.pred_count += 1

        return {"prediction": prediction, "saliency": saliency, "cf": cf, "saliency_present": bool(saliency),
                "cf_present": bool(cf), "conversation": conversation, "llm_time": llm_time}
