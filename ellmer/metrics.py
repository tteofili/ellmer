import argparse
import json
from scipy.stats import kendalltau
from ellmer.utils import cosine_similarity
import pandas as pd


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

    # transform predictions from text to boolean, where necessary
    make_predictions_boolean(pred1_json)
    make_predictions_boolean(pred2_json)

    observations = []
    # for each prediction, identify:
    for idx in range(len(pred1_json)):
        pred1 = pred1_json[idx]
        pred2 = pred2_json[idx]
        if 'id' not in pred1:
            continue
        if pred1['id'] != pred2['id']:
            idx = max(int(pred1['id']), int(pred2['id']))
            pred1 = pred1_json[idx]
            pred2 = pred2_json[idx]
        observation = dict()
        observation['id'] = idx

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
            try:
                kt = kendalltau(sal1_ranked_keys, sal2_ranked_keys).statistic
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

    obs_df = pd.concat(observations, axis=1).T
    print(f"pred1-pred2 eq rate: {eq_ratio(obs_df['pred1'].values, obs_df['pred2'].values)}")
    print(f"pred1-gt eq rate: {eq_ratio(obs_df['label'].values, obs_df['pred2'].values)}")
    print(f"pred2-gt eq rate: {eq_ratio(obs_df['pred1'].values, obs_df['label'].values)}")
    print(f"avg kt: {obs_df['kt'].mean()}")
    print(f"avg cf cos_sim: {obs_df['cos_sim'].mean()}")

    return obs_df
