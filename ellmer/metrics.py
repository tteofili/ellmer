import math
import traceback
import numpy as np
from scipy.stats import kendalltau
import pandas as pd
import re
import os
import operator
from sklearn.metrics import auc
from collections import Counter
import json

WORD = re.compile(r'\w+')


def make_predictions_boolean(preds):
    preds_copy = []
    for pred in preds:
        if "prediction" in pred:
            str_pred = str(pred['prediction'])
            if str_pred.lower().startswith("yes") or str_pred.lower().startswith("true"):
                bp = True
            else:
                bp = False
            pred['matching_prediction'] = bp
        preds_copy.append(pred)


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


def eq_ratio(l1, l2):
    same = 0
    for idx in range(len(l1)):
        if l1[idx] == l2[idx]:
            same += 1
    return same / len(l1)


def get_concordance(pred1_file, pred2_file):
    # read json files into dictionaries
    with open(pred1_file) as json1_file:
        pred1_json = json.load(json1_file)
    with open(pred2_file) as json2_file:
        pred2_json = json.load(json2_file)

    if 'data' in pred1_json:
        pred1_json = pred1_json['data']

    if 'data' in pred2_json:
        pred2_json = pred2_json['data']

    # transform predictions from text to boolean, where necessary
    make_predictions_boolean(pred1_json)
    make_predictions_boolean(pred2_json)

    p1_ids = [p['id'] for p in pred1_json]
    p2_ids = [p['id'] for p in pred2_json]

    pids = set(p1_ids).intersection(set(p2_ids))

    observations = []
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
                lidx+=1
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

            # the rate of agreement between predictions
            mnm1 = get_prediction(pred1)
            observation['pred1'] = mnm1
            mnm2 = get_prediction(pred2)
            observation['pred2'] = mnm2

            if mnm1 is None or mnm2 is None:
                continue

            # the rate of agreement between each set of predictions and the ground truth
            observation['label'] = bool(pred1['label'])

            # how many match / non-match predictions respectively, for each list (pred1, pred2, labels)
            if mnm2 == mnm1:
                agree = True
            else:
                agree = False
            observation['agree'] = agree

            # for saliency explanations, identify
            # the avg kendall-tau between saliency explanations
            sal1 = get_saliency(pred1)
            sal2 = get_saliency(pred2)

            if sal1 is not None and sal2 is not None:
                sal1_ranked_keys = list(
                    {k: v for k, v in sorted(sal1.items(), key=lambda item: item[1], reverse=True)}.keys())
                sal2_ranked_keys = list(
                    {k: v for k, v in sorted(sal2.items(), key=lambda item: item[1], reverse=True)}.keys())
                msz = min(len(sal1_ranked_keys), len(sal2_ranked_keys))
                try:
                    kt = kendalltau(sal1_ranked_keys[:msz], sal2_ranked_keys[:msz]).statistic
                except:
                    kt = 0
                observation['kt'] = kt

            # for counterfactual explanations, identify
            cf1 = get_cf(pred1)
            cf2 = get_cf(pred2)

            # the similarity between the counterfactuals using different similarity metrics
            if cf1 is not None and cf2 is not None:
                observation['cos_sim'] = cosine_similarity(list(cf1.values()), list(cf2.values()))[0]

            # examples that are particularly dissimilar (probably as the least similar 5% perc)
            # the similarity between counterfactuals in different groups (match, non-matching, disagree with label, etc.)
            observations.append(pd.Series(observation))
        except:
            traceback.print_exc()
            pass

    if len(observations) > 0:
        obs_df = pd.concat(observations, axis=1).T
        print(f"pred1-pred2 eq rate: {eq_ratio(obs_df['pred1'].values, obs_df['pred2'].values)}")
        print(f"pred1-gt eq rate: {eq_ratio(obs_df['label'].values, obs_df['pred2'].values)}")
        print(f"pred2-gt eq rate: {eq_ratio(obs_df['pred1'].values, obs_df['label'].values)}")
        print(f"avg kt: {obs_df['kt'].mean()}")
        print(f"avg cf cos_sim: {obs_df['cos_sim'].mean()}")
        return obs_df
    else:
        return pd.DataFrame()


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
    print(test_set_df.shape)
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

        if 'data' in results_json:
            results_json = results_json['data']

        saliencies = []
        predictions = []
        for v in results_json:
            saliencies.append(v['saliency'])
            predictions.append(v['prediction'])
        for threshold in thresholds:
            top_k = int(threshold * attr_len)
            test_set_df_c = test_set_df.copy().astype(str)
            for i in range(len(predictions)):
                if int(predictions[i]) == 0:
                    reverse = False
                attributes_dict = dict()
                sal_dict = saliencies[i]
                if type(sal_dict) == str:
                    continue
                for k,v in sal_dict.items():
                    try:
                        attributes_dict[k] = float(v)
                    except:
                        try:
                            attributes_dict[k] = float(v[0])
                        except:
                            attributes_dict[k] = 0
                            print(f'{v} is not a float in {sal_dict}')
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
        if 'data' in results_json:
            results_json = results_json['data']
        cfs = []
        predictions = []
        indexes = []
        for v in results_json:
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
                if i >= len(cfs):
                    break

                if type(cfs[i]) == str:
                    continue

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
