import traceback

import pandas as pd
import numpy as np
import openai
import random
import time
import math
import re
import os
import operator
from sklearn.metrics import auc
from collections import Counter
import json

WORD = re.compile(r'\w+')


def predict(x: pd.DataFrame, llm_fn, verbose: bool = True, mojito: bool = False):
    count = 0
    xcs = []
    for idx in range(len(x)):
        xc = x.iloc[[idx]].copy()
        ltuple, rtuple = get_tuples(xc)
        answer = llm_fn.er(ltuple, rtuple)
        if verbose:
            print(f'{ltuple}\n{rtuple}')
            print(answer)
        nomatch_score, match_score = text_to_match(answer, llm_fn)
        xc['nomatch_score'] = nomatch_score
        xc['match_score'] = match_score
        count += 1
        if mojito:
            full_df = np.dstack((xc['nomatch_score'], xc['match_score'])).squeeze()
            xc = full_df
        xcs.append(xc)
        # print(f'{count},{self.summarized},{self.idks}')
    return pd.concat(xcs, axis=0)


def get_tuples(xc):
    elt = dict()
    ert = dict()
    for c in xc.columns:
        if c in ['ltable_id', 'rtable_id']:
            continue
        if c.startswith('ltable_'):
            elt[str(c)] = xc[c].astype(str).values[0]
        if c.startswith('rtable_'):
            ert[str(c)] = xc[c].astype(str).values[0]
    return elt, ert


def text_to_match(answer, llm_fn, n=0):
    summarized = 0
    idks = 0
    no_match_score = 0
    match_score = 0
    if answer.lower().startswith("yes"):
        match_score = 1
    elif answer.lower().startswith("no"):
        no_match_score = 1
    else:
        if "yes".casefold() in answer.casefold():
            match_score = 1
        elif "no".casefold() in answer.casefold():
            no_match_score = 1
        elif n == 0:
            template = "summarize \"response\" as yes or no"
            try:
                summarized_answer = llm_fn(template.replace("response", answer))
            except:
                summarized_answer = "false"
            summarized += 1
            snms, sms = text_to_match(summarized_answer, llm_fn, n=1)
            if snms == 0 and sms == 0:
                idks += 1
                no_match_score = 1
    return no_match_score, match_score


def read_prompt(file_path: str):
    with open(file_path) as file:
        lines = [(line.rstrip().split(':')) for line in file]
    return lines


def concordance_correlation(y_pred, y_true):
    # Raw data
    dct = {
        'y_true': y_true,
        'y_pred': y_pred
    }
    df = pd.DataFrame(dct)
    # Remove NaNs
    df = df.dropna()
    # Pearson product-moment correlation coefficients
    y_true = df['y_true']
    y_pred = df['y_pred']
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Population variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Population standard deviations
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / denominator


def completion_with_backoff(deployment_id="gpt-35-turbo", model="gpt-3.5-turbo", messages=None, temperature=0,
                            initial_delay=1, max_retries=10, exponential_base: float = 2, jitter: bool = True,
                            errors: tuple = (openai.RateLimitError, openai.Timeout), ):
    num_retries = 0
    delay = initial_delay

    while True:
        try:
            openai.api_type = "azure"
            openai.api_version = "2023-05-15"
            return openai.ChatCompletion.create(deployment_id=deployment_id, model=model, messages=messages,
                                                temperature=temperature)

        # Retry on specified errors
        except errors as e:
            print(e)
            # Increment retries
            num_retries += 1

            # Check if max retries has been reached
            if num_retries > max_retries:
                raise Exception(
                    f"Maximum number of retries ({max_retries}) exceeded."
                )

            # Increment the delay
            delay *= exponential_base * (1 + jitter * random.random())

            # Sleep for the delay
            time.sleep(delay)

        # Raise exceptions for any errors not specified
        except Exception as e:
            raise e


def rbo(list1, list2, p=0.9):
    # tail recursive helper function
    def helper(ret, i, d):
        l1 = set(list1[:i]) if i < len(list1) else set(list1)
        l2 = set(list2[:i]) if i < len(list2) else set(list2)
        a_d = len(l1.intersection(l2)) / i
        term = math.pow(p, i) * a_d
        if d == i:
            return ret + term
        return helper(ret + term, i + 1, d)

    k = max(len(list1), len(list2))
    x_k = len(set(list1).intersection(set(list2)))
    summation = helper(0, 1, k)
    return ((float(x_k) / k) * math.pow(p, k)) + ((1 - p) / p * summation)


def cosine_similarity(tuple1, tuple2):
    string1 = concatenate_list(tuple1)
    string2 = concatenate_list(tuple2)
    cos_sim = get_cosine(text_to_vector(string1), text_to_vector(string2))
    vector = [cos_sim]
    return vector


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


def concatenate_list(text_features):
    return ' '.join(text_features)


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def get_faithfulness(saliency_names: list, eval_fn, base_dir: str, test_set_df: pd.DataFrame):
    np.random.seed(0)

    thresholds = [0.1, 0.33, 0.5, 0.7, 0.9]

    attr_len = len(test_set_df.columns) - 2
    aucs = dict()
    for saliency in saliency_names:
        model_scores = []
        reverse = True
        json_path = os.path.join(base_dir, saliency + '_results.json')
        with open(json_path) as fd:
            results_json = json.load(fd)
        saliencies = []
        predictions = []
        for v in results_json['data']:
            saliencies.append(v['saliency'])
            predictions.append(v['prediction'])
        for threshold in thresholds:
            top_k = int(threshold * attr_len)
            test_set_df_c = test_set_df.copy().astype(str)
            for i in range(len(predictions)):
                if int(predictions[i]) == 0:
                    reverse = False
                attributes_dict = saliencies[i]
                if saliency.startswith('certa'):
                    sorted_attributes_dict = sorted(attributes_dict.items(), key=operator.itemgetter(1),
                                                    reverse=True)
                else:
                    sorted_attributes_dict = sorted(attributes_dict.items(), key=operator.itemgetter(1),
                                                    reverse=reverse)
                top_k_attributes = sorted_attributes_dict[:top_k]
                for t in top_k_attributes:
                    split = t[0].split('__')
                    if len(split) == 2:
                        test_set_df_c.at[i, split[0]] = test_set_df_c.iloc[i][split[0]].replace(split[1], '')
                    else:
                        test_set_df_c.at[i, t[0]] = ''
            try:
                evaluation = eval_fn(test_set_df_c)
                model_scores.append(evaluation)
            except Exception as e:
                print(f'skipped faithfulness for {saliency}: {e}')
                traceback.print_exc()
                model_scores.append(evaluation)
        if len(thresholds) == len(model_scores):
            auc_sal = auc(thresholds, model_scores)
            aucs[saliency] = auc_sal
    return aucs


def get_cf_metrics(explainer_names: list, predict_fn, base_dir, test_set_df: pd.DataFrame):
    to_drop = ['ltable_id', 'rtable_id', 'match', 'label']
    rows = dict()
    for explainer_name in explainer_names:
        json_path = os.path.join(base_dir, explainer_name + '_results.json')
        with open(json_path) as fd:
            results_json = json.load(fd)
        cfs = []
        predictions = []
        indexes = []
        for v in results_json['data']:
            cfs.append(v['cfs'])
            predictions.append(v['prediction'])
            indexes.append(v['id'])
        validity = 0
        proximity = 0
        sparsity = 0
        diversity = 0
        count = 1e-10
        for i in range(len(test_set_df)):
            try:
                if len(cfs[i]) == 0 or len(cfs[i][0].keys()) == 0 or indexes[i] != i:
                    continue

                instance = test_set_df.iloc[i].copy()
                for c in to_drop:
                    if c in test_set_df.columns:
                        instance = instance.drop(c)
                matching = int(predictions[i])

                # validity
                validity += get_validity(predict_fn, cfs[i], matching)

                # proximity
                proximity += get_proximity(cfs[i], instance.to_dict())

                # sparsity
                sparsity += get_sparsity(cfs[i], instance.to_dict())

                # diversity
                diversity += get_diversity(pd.DataFrame(cfs[i]))

                count += 1
            except:
                traceback.print_exc()
                pass

        mean_validity = validity / count
        mean_proximity = proximity / count
        mean_sparsity = sparsity / count
        mean_diversity = diversity / count
        row = {'validity': mean_validity, 'proximity': mean_proximity,
               'sparsity': mean_sparsity, 'diversity': mean_diversity}
        rows[explainer_name] = row
    return rows


def get_validity(predict_fn, counterfactuals, original):
    rowsc_df = pd.DataFrame(counterfactuals.copy())
    predicted = predict_fn(rowsc_df)['match_score'].values[0]
    return 1 - abs(predicted - original)


def get_proximity(counterfactuals, original_row):
    proximity_all = 0
    for i in range(len(counterfactuals)):
        curr_row = counterfactuals[i]
        sum_cat_dist = 0
        for c, v in curr_row.items():
            if c in original_row and v == original_row[c]:
                sum_cat_dist += 1

        proximity = 1 - (1 / len(original_row)) * sum_cat_dist
        proximity_all += proximity
    return proximity_all / len(counterfactuals)


def get_diversity(expl_df):
    diversity = 0
    for i in range(len(expl_df)):
        for j in range(len(expl_df)):
            if i == j:
                continue
            curr_row1 = expl_df.iloc[i]
            curr_row2 = expl_df.iloc[j]
            sum_cat_dist = 0

            for c, v in curr_row1.items():
                if v != curr_row2[c]:
                    sum_cat_dist += 1

            dist = sum_cat_dist / len(curr_row1)
            diversity += dist
    return diversity / math.pow(len(expl_df), 2)


def get_sparsity(expl_df, instance):
    return 1 - get_proximity(expl_df, instance) / (len(expl_df[0].keys()) / 2)
