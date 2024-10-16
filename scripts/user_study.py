import itertools
import ellmer.models
import ellmer.metrics
from langchain.prompts import ChatPromptTemplate
from langchain import HuggingFaceHub
from langchain.chat_models import AzureChatOpenAI
from langchain_community.chat_models.huggingface import ChatHuggingFace
import ellmer.utils
import openai
import os
import json

openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_judgment(ltuple, rtuple, prediction, explanation1, explanation2, llm):
    conversation = []
    for prompt_message in ellmer.utils.read_prompt(judge_prompt):
        conversation.append((prompt_message[0], prompt_message[1]))
    template = ChatPromptTemplate.from_messages(conversation)
    if model_type in ['hf']:
        messages = template.format_messages(ltuple=ltuple, rtuple=rtuple, prediction=prediction,
                                            explanation1=explanation1, explanation2=explanation2)
        raw_content = llm.invoke(messages)
        try:
            judge_answer = raw_content.content.split('|>')[-1]
        except:
            judge_answer = raw_content.content
        conversation.append(('assistant', judge_answer))
        conversation.append(('human', 'summarize your judgement as either "system1" if you prefer the first explanation, "system2" if you prefer the second explanation or "none" if they are equally good or bad'))
        summarize_template = ChatPromptTemplate.from_messages(conversation)
        summarize_messages = summarize_template.format_messages(ltuple=ltuple, rtuple=rtuple, prediction=prediction,
                                            explanation1=explanation1, explanation2=explanation2, judgement=judge_answer)
        raw_summary = llm.invoke(summarize_messages)
        try:
            summary = raw_summary.content.split('|>')[-1]
        except:
            summary = raw_summary.content
    else:
        messages = template.format_messages(ltuple=ltuple, rtuple=rtuple, prediction=prediction,
                                            explanation1=explanation1, explanation2=explanation2)
        answer = llm(messages)
        judge_answer = answer.content
        conversation.append(('assistant', judge_answer))
        conversation.append(('human',
                             'summarize your judgement as either "system1" if you prefer the first explanation, "system2" if you prefer the second explanation or "none" if they are equally good or bad'))
        summarize_template = ChatPromptTemplate.from_messages(conversation)
        summarize_messages = summarize_template.format_messages(ltuple=ltuple, rtuple=rtuple, prediction=prediction,
                                                                explanation1=explanation1, explanation2=explanation2,
                                                                judgement=judge_answer)
        summary = llm.invoke(summarize_messages).content

    return judge_answer, summary


def get_prediction(pred):
    try:
        if "prediction" in pred:
            mnm = pred['prediction']
        elif "answer" in pred:
            mnm = pred['answer']['matching_prediction']
        else:
            mnm = pred['matching_prediction']
    except:
        mnm = None
    return mnm


def get_saliency(pred):
    try:
        if "saliency" in pred:
            sal = pred['saliency']
        elif "answer" in pred:
            sal = pred['answer']['saliency_explanation']
        else:
            sal = pred['saliency_exp']
    except:
        sal = None
    return sal


def get_cf(pred):
    try:
        if "cfs" in pred:
            cf = pred['cfs'][0]
        elif "answer" in pred:
            cf = pred['answer']['counterfactual_explanation']
        else:
            if "counterfactual_record" in pred:
                cf = pred['cf_exp']['counterfactual_record']
            else:
                try:
                    cf = pred['cf_exp']['record1'] | pred['cf_exp']['record2']
                except:
                    cf = list(pred['cf_exp'].values())[0]
                    cf.pop("nomatch_score")
                    cf.pop("match_score")
    except:
        cf = None
    return cf


if __name__ == "__main__":

    model_type = 'azure_openai'
    if model_type in ['hf']:
        model_type = 'hf'
        model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    else:
        model_type = 'azure_openai'
        model_name = 'gpt-4-32k'

    deployment_name = 'gpt-4-32k'
    temperature = 0.01

    if model_type == 'hf':
        max_length = 512
        delegate_llm = HuggingFaceHub(repo_id=model_name, task="text-generation",
                             model_kwargs={'temperature': temperature, 'max_length': max_length,
                                           'max_new_tokens': 1024})
        llm = ChatHuggingFace(llm=delegate_llm, token=True)
    elif model_type == 'azure_openai':
        model_version = "2023-05-15"
        llm = AzureChatOpenAI(deployment_name=deployment_name, model_name=model_name, request_timeout=30,
                                   openai_api_version=model_version, temperature=temperature)
    else:
        raise Exception("no llm")

    judge_prompt = "ellmer/prompts/user1.txt"
    base_directory = '/Users/tteofili/Desktop/explanations/'
    target = "carparts-token"
    directory = base_directory + target
    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    all_files = [f for f in all_files if not '.DS_Store' in f]
    picks = []
    for pair in itertools.combinations(all_files, 2):
        sys1 = pair[0]
        file1 = os.path.join(directory, sys1)
        sys2 = pair[1]
        file2 = os.path.join(directory, sys2)
        print(f'opening {file1}')
        with open(file1) as json1_file:
            pred1_json = json.load(json1_file)
        print(f'opening {file2}')
        with open(file2) as json2_file:
            pred2_json = json.load(json2_file)

        if 'data' in pred1_json:
            pred1_json = pred1_json['data']

        if 'data' in pred2_json:
            pred2_json = pred2_json['data']

        p1_ids = [p['id'] for p in pred1_json]
        p2_ids = [p['id'] for p in pred2_json]

        pids = set(p1_ids).intersection(set(p2_ids))

        transcript = []

        # for each prediction, identify:
        lidx = 0
        ridx = 0

        for pid in pids:
            try:
                pred1 = None
                for cpr1 in pred1_json[lidx:]:
                    if cpr1['id'] == pid:
                        pred1 = cpr1
                        break
                    lidx += 1
                pred2 = None
                for cpr2 in pred2_json[ridx:]:
                    if cpr2['id'] == pid:
                        pred2 = cpr2
                        break
                    ridx += 1
                if pred1 is None or pred2 is None:
                    continue
                observation = dict()
                observation['id'] = pid

                sal1 = get_saliency(pred1)
                sal2 = get_saliency(pred2)

                prediction = get_prediction(pred1)
                ltuple = pred1['ltuple']
                rtuple = pred1['rtuple']
                observation['ltuple'] = ltuple
                observation['rtuple'] = rtuple

                if sal1 is not None and sal2 is not None:
                    try:
                        observation['saliency_1'] = sal1
                        observation['saliency_2'] = sal2
                        judgment, pick = get_judgment(ltuple, rtuple, prediction, sal1, sal2, llm)
                        if 'system1' in pick.lower():
                            pick = sys1
                        elif 'system2' in pick.lower():
                            pick = sys1
                        picks.append(pick)
                        observation['saliency_judgment'] = judgment
                        observation['saliency_pick'] = pick
                    except Exception as e:
                        print(e)
                        judgment = ''
                        pass

                # for counterfactual explanations, identify
                cf1 = get_cf(pred1)
                cf2 = get_cf(pred2)
                observation['cf_1'] = cf1
                observation['cf_2'] = cf2

                # the similarity between the counterfactuals using different similarity metrics
                if cf1 is not None and cf2 is not None:
                    try:
                        judgment, pick = get_judgment(ltuple, rtuple, prediction, cf1, cf2, llm)
                        if 'system1' in pick.lower():
                            pick = sys1
                        elif 'system2' in pick.lower():
                            pick = sys1
                        picks.append(pick)
                        observation['cf_judgment'] = judgment
                        observation['cf_pick'] = pick
                    except Exception as e:
                        judgment = ''
                        print(e)
                        pass
                transcript.append(observation)
            except:
                pass
        os.makedirs(f'./experiments/user_study/{model_type}/{target}', exist_ok=True)
        sys1 = sys1.replace('.json','')
        sys2 = sys2.replace('.json','')
        output_file_path = f'./experiments/user_study/{model_type}/{target}/user_study_{sys1}-{sys2}.json'
        with open(output_file_path, 'w') as fout:
            json.dump(transcript, fout)
        print(picks)

    picks_counts = dict((x, picks.count(x)) for x in picks)
    print(picks_counts)
    output_file_path = f'./experiments/user_study/{model_type}/user_study_{target}_{model_type}_picks.json'
    with open(output_file_path, 'w') as fout:
        json.dump(picks_counts, fout)
