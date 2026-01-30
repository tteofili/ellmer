import json
import math
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def flatten(param):
    if type(param) == list:
        return param[0]
    else:
        return param


def get_attribute_from_token(token_feat):
    return token_feat.split("__")[0]


def topk_attr_token_overlap(token_sal, attr_sal, k_tokens=30, k_attrs=3):
    token_sal = {k: flatten(v) for k, v in token_sal.items()}
    attr_sal = {k: flatten(v) for k, v in attr_sal.items()}

    try:
        top_tokens = sorted(token_sal.items(), key=lambda x: x[1], reverse=True)[:k_tokens]
        token_attrs = {get_attribute_from_token(t) for t, _ in top_tokens}
        top_attrs = {
            a for a, _ in sorted(attr_sal.items(), key=lambda x: x[1], reverse=True)[:k_attrs]
        }
    except:
        return 0.0

    if not top_attrs:
        return 0.0

    return len(token_attrs & top_attrs) / len(top_attrs)


def saliency_consistency(token_sal, attr_sal):

    score = 0.0
    kt = 0.0
    pr_val = 0.0
    rho = 0.0
    avg_overlap = 0.0
    cos_sim = 0.0

    token_sal = {k: flatten(v) for k, v in token_sal.items()}
    attr_sal = {k: flatten(v) for k, v in attr_sal.items()}

    token_mass = defaultdict(float)
    try:
        for tok, val in token_sal.items():
            token_mass[get_attribute_from_token(tok)] += val

        total_tok = sum(token_mass.values())
        total_attr = sum(attr_sal.values())
    except:
        return score, kt, pr_val, rho, avg_overlap, cos_sim

    if total_tok == 0 or total_attr == 0:
        return score, kt, pr_val, rho, avg_overlap, cos_sim

    att_vec = []
    tok_vec = []
    for tok, val in attr_sal.items():
        tok_vec.append(val / total_tok)
        att_vec.append(val)

    tok_ranked_keys = list(
        {k: v for k, v in sorted(token_mass.items(), key=lambda item: item[1], reverse=True)}.keys())
    att_ranked_keys = list(
        {k: v for k, v in sorted(attr_sal.items(), key=lambda item: item[1], reverse=True)}.keys())

    avg_overlap = 0.0
    idx = 0
    if len(att_vec) == len(tok_vec) and len(att_vec) > 0:
        pr_val = pearsonr(att_vec, tok_vec)[0]
    else:
        pr_val = 0.0

    rho = spearmanr(att_vec, tok_vec)[0]
    min_len = min(len(att_ranked_keys), len(tok_ranked_keys))
    kt = kendalltau(att_ranked_keys[:min_len], tok_ranked_keys[:min_len]).statistic
    for top_k_kt in range(2, len(tok_ranked_keys), 2):
        top_a = set(np.argsort(-np.abs(att_vec))[:top_k_kt])
        top_b = set(np.argsort(-np.abs(tok_vec))[:top_k_kt])
        avg_overlap += len(top_a & top_b) / top_k_kt

        idx += 1
    avg_overlap /= idx

    for a in set(token_mass) | set(attr_sal):
        p_tok = token_mass.get(a, 0.0) / total_tok
        p_attr = attr_sal.get(a, 0.0) / total_attr
        score += min(p_tok, p_attr)

    cos_sim = local_cosine_similarity(att_vec, tok_vec)

    return score, kt, pr_val, rho, avg_overlap, cos_sim


def extract_cf_tokens(original, cf, side):
    changed = set()

    for k, v in original.items():
        if not k.startswith(side):
            continue

        orig = set(v.lower().split()) if isinstance(v, str) else set()
        new = set(cf.get(k, "").lower().split())

        for t in orig.symmetric_difference(new):
            changed.add(f"{k}__{t}")

    return changed


def local_cosine_similarity(a, b, eps=1e-12):
    num = np.dot(a, b)
    den = np.linalg.norm(a) * np.linalg.norm(b) + eps
    return num / den


def extract_cf_attributes(original, cf):
    changed = set()
    for k in original:
        if original.get(k) != cf.get(k):
            changed.add(k)
    return changed


def dcg(relevances: List[int]) -> float:
    return sum(
        rel / math.log2(idx + 2)
        for idx, rel in enumerate(relevances)
    )


def ndcg_cf(
        attr_cf: List[str],
        token_cf: List[str],
        k: int = None
) -> float:

    if not attr_cf or not token_cf:
        return 0.0

    projected_attrs = list(token_cf)

    if k is not None:
        projected_attrs = projected_attrs[:k]

    relevances = [
        1 if attr in attr_cf else 0
        for attr in projected_attrs
    ]

    # Compute DCG
    dcg_val = dcg(relevances)

    # Compute ideal DCG (all relevant attributes ranked first)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg_val = dcg(ideal_relevances)

    if idcg_val == 0:
        return 0.0

    return dcg_val / idcg_val


def cf_attr_token_alignment(token_attrs, attr_cf_attrs):
    if not attr_cf_attrs:
        return 0.0, 0.0, 0.0, 0.0

    intersection = token_attrs & attr_cf_attrs
    union = token_attrs | attr_cf_attrs

    coverage = len(intersection) / len(attr_cf_attrs)
    precision = len(intersection) / len(token_attrs) if token_attrs else 0.0
    jacc = len(intersection) / len(union) if union else 0.0

    if coverage + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * coverage * precision / (coverage + precision)

    return coverage, precision, f1, jacc


def compare(pair, model, verbose: bool = False,
            explainers=['zs_sample', 'cot_sample', 'fs_sample', 'certa_sample', 'hybrid_sample', ], k_tokens=30,
            k_attrs=3):
    agg_results = []
    for explainer in explainers:
        try:
            with open(pair[0] + explainer + "_results.json") as f:
                token_data = {d["id"]: d for d in json.load(f)["data"]}
        except:
            token_data = {}

        try:
            with open(pair[1] + explainer + "_results.json") as f:
                attr_data = {d["id"]: d for d in json.load(f)["data"]}
        except:
            attr_data = {}

        results = []

        for id_ in token_data:
            try:
                tok = token_data[id_]["saliency"]
                attr = attr_data[id_]["saliency"]
            except:
                continue

            if not tok or not attr or type(tok) == str or type(attr) == str:
                continue

            new_attr = attr.copy()
            for k, v in attr.items():
                if '__' in k:
                    target_att = get_attribute_from_token(k)
                    if target_att in new_attr:
                        new_attr[target_att] += v
                    else:
                        new_attr[target_att] = v
                    new_attr.pop(k)
            attr = new_attr

            try:
                avg_amc, kt, pc, rho, avg_overlap, cos_sim = saliency_consistency(tok, attr)
                results.append({
                    "id": id_,
                    "topk_attr_token_overlap": topk_attr_token_overlap(tok, attr, k_tokens=k_tokens, k_attrs=k_attrs),
                    "attr_token_mass_consistency": avg_amc,
                    "kt": kt,
                    "pc": pc,
                    "spearman": rho,
                    "avg_overlap": avg_overlap,
                })
            except:
                continue
        if len(results) == 0:
            continue
        avg_topk = sum(r["topk_attr_token_overlap"] for r in results) / len(results)
        avg_mass = sum(r["attr_token_mass_consistency"] for r in results) / len(results)
        avg_kt = sum(r["kt"] for r in results) / len(results)
        avg_pc = sum(r["pc"] for r in results) / len(results)
        avg_spearman = sum(r["spearman"] for r in results) / len(results)
        avg_overlap = sum(r["avg_overlap"] for r in results) / len(results)

        if verbose:
            print("Average Top-k Attribute–Token Overlap:", avg_topk)
            print("Average Attribute–Token Mass Consistency:", avg_mass)

        results = []

        for id_ in token_data:
            try:
                tok_item = token_data[id_]
                attr_item = attr_data[id_]
            except:
                continue

            if tok_item['ltuple'] != attr_item['ltuple'] or tok_item['rtuple'] != attr_item['rtuple']:
                continue

            original = {}
            original.update(dict(map(lambda item: ("ltable_" + item[0], item[1]), tok_item['ltuple'].items())))
            original.update(dict(map(lambda item: ("rtable_" + item[0], item[1]), tok_item['rtuple'].items())))

            tok_cf = tok_item["cfs"][0]
            attr_cf = attr_item["cfs"][0]

            if not tok_cf or not attr_cf:
                continue

            token_cf_tokens = set()
            try:
                token_cf_tokens |= extract_cf_tokens(original, tok_cf, "ltable")
                token_cf_tokens |= extract_cf_tokens(original, tok_cf, "rtable")

                attr_cf_attrs = extract_cf_attributes(original, attr_cf)
            except:
                continue

            token_attrs = {get_attribute_from_token(t) for t in token_cf_tokens}

            attr_cf_text = ' '.join([str(t) for k, t in attr_item['cfs'][0].items() if 'match_score' not in k])
            tok_cf_text = ' '.join([str(t) for k, t in tok_item['cfs'][0].items() if 'match_score' not in k])
            original_text = ' '.join([str(t) for t in original.values()])
            try:
                cf_sim = tfidf_counterfactual_similarity(original_text, [attr_cf_text, tok_cf_text])[0, 1]
            except:
                cf_sim = 0.5

            cov, prec, f1, jacc = cf_attr_token_alignment(token_attrs, attr_cf_attrs)

            score = ndcg_cf(attr_cf_attrs, token_attrs, k=k_attrs)

            attr_sparsity = len(attr_cf_attrs) / len(attr_cf)
            tok_sparsity = len(token_cf_tokens) / len(' '.join([v for v in original.values()]).split(' '))
            ger = tok_sparsity / (1e-10 + attr_sparsity)

            results.append({
                "id": id_,
                "cf_attr_coverage": cov,
                "cf_attr_precision": prec,
                "cf_attr_f1": f1,
                "cf_jacc": jacc,
                "cf_ndcg": score,
                "attr_sparsity": attr_sparsity,
                "tok_sparsity": tok_sparsity,
                "ger": ger,
                "cf_similarity": cf_sim,
            })
        if len(results) == 0:
            continue
        avg_coverage = sum(r["cf_attr_coverage"] for r in results) / len(results)
        avg_prec = sum(r["cf_attr_precision"] for r in results) / len(results)
        avg_f1 = sum(r["cf_attr_f1"] for r in results) / len(results)
        avg_jacc = sum(r["cf_jacc"] for r in results) / len(results)
        avg_ndcg = sum(r["cf_ndcg"] for r in results) / len(results)
        avg_att_sparsity = sum(r["attr_sparsity"] for r in results) / len(results)
        avg_tok_sparsity = sum(r["tok_sparsity"] for r in results) / len(results)
        avg_ger = sum(r["ger"] for r in results) / len(results)
        avg_cf_sim = sum(r["cf_similarity"] for r in results) / len(results)

        if verbose:
            print("Avg Coverage:", avg_coverage)
            print("Avg Precision:", avg_prec)
            print("Avg F1:", avg_f1)
        row = {"model": model, "explainer": explainer, "coverage": avg_coverage, "prec": avg_prec,
               "f1": avg_f1, "topk_overlap": avg_topk, "attr_token_mass": avg_mass, "jacc": avg_jacc,
               "ndcg": avg_ndcg, "avg_att_sparsity": avg_att_sparsity,
               "avg_tok_sparsity": avg_tok_sparsity,
               "avg_ger": avg_ger, "avg_kt": avg_kt, "avg_pc": avg_pc, "spearman": avg_spearman,
               "avg_overlap": avg_overlap, "avg_cf_sim": avg_cf_sim}
        agg_results.append(row)
        dname = pair[0].split('/')[5]
        pd.DataFrame.from_records(agg_results)[['explainer', 'avg_kt', 'avg_cf_sim']].to_csv(
            f'../experiments/{model}_{dname}.csv', index=False)
    return pd.DataFrame(agg_results)


def extract_changed_tokens(original_tokens, cf_tokens):
    orig = set(original_tokens)
    cf = set(cf_tokens)

    removed = orig - cf
    added = cf - orig

    return list(removed | added)


def counterfactual_change_text(original_tokens, cf_tokens):
    changed = extract_changed_tokens(original_tokens, cf_tokens)
    return " ".join(changed)


def tfidf_counterfactual_similarity(
        original_tokens,
        cf_tokens_list
):
    texts = [
        counterfactual_change_text(original_tokens, cf)
        for cf in cf_tokens_list
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b"
    )

    tfidf = vectorizer.fit_transform(texts)
    return cosine_similarity(tfidf)


precomputed_data = {
    "chatgpt4": [
        [
            "../experiments/azure_openai/gpt-4-32k/token/abt_buy/20260125/23_49/",
            "../experiments/azure_openai/gpt-4-32k/attribute/abt_buy/20260129/11_10/",
        ],
        [
            "../experiments/azure_openai/gpt-4-32k/token/beers/20260126/01_49/",
            "../experiments/azure_openai/gpt-4-32k/attribute/beers/20260126/11_51/",
        ],
        [
            "../experiments/azure_openai/gpt-4-32k/token/books/20260125/20_01/",
            "../experiments/azure_openai/gpt-4-32k/attribute/books/20260126/18_08/",
        ],
        [
            "../experiments/azure_openai/gpt-4-32k/token/carparts/20260125/21_58/",
            "../experiments/azure_openai/gpt-4-32k/attribute/carparts/20260126/19_00/",
        ],
        [
            "../experiments/azure_openai/gpt-4-32k/token/cameras_large/20260125/08_56/",
            "../experiments/azure_openai/gpt-4-32k/attribute/cameras_large/20260123/18_35/",
        ],
        [
            "../experiments/azure_openai/gpt-4-32k/token/fodo_zaga/20260126/03_49/",
            "../experiments/azure_openai/gpt-4-32k/attribute/fodo_zaga/20260126/12_25/",
        ],
        [
            "../experiments/azure_openai/gpt-4-32k/token/walmart_amazon/20260126/07_09/",
            "../experiments/azure_openai/gpt-4-32k/attribute/walmart_amazon/20260126/13_36/",
        ],
        [
            "../experiments/azure_openai/gpt-4-32k/token/watches_large/20260125/13_37/",
            "../experiments/azure_openai/gpt-4-32k/attribute/watches_large/20260124/11_15/",
        ],
        [
            "../experiments/azure_openai/gpt-4-32k/token/watches_large/20260126/05_32/",
            "../experiments/azure_openai/gpt-4-32k/attribute/amazon_google/20260126/13_12/",
        ],
    ],
    "chatgpt5": [
        [
            "../experiments/azure_openai/gpt-5-nano/token/abt_buy/20251225/06_03/",
            "../experiments/azure_openai/gpt-5-nano/attribute/abt_buy/20251226/01_45/",
        ],
        [
            "../experiments/azure_openai/gpt-5-nano/token/beers/20251225/11_58/",
            "../experiments/azure_openai/gpt-5-nano/attribute/beers/20251224/17_34/",
        ],
        [
            "../experiments/azure_openai/gpt-5-nano/token/books/20250917/10_53/",
            "../experiments/azure_openai/gpt-5-nano/attribute/books/20251110/15_04/",
        ],
        [
            "../experiments/azure_openai/gpt-5-nano/token/cameras_small/20251224/13_15/",
            "../experiments/azure_openai/gpt-5-nano/attribute/cameras_small/20251223/22_36/",
        ],
        [
            "../experiments/azure_openai/gpt-5-nano/token/amazon_google/20251224/21_01/",
            "../experiments/azure_openai/gpt-5-nano/attribute/amazon_google/20251225/15_03/",
        ],
        [
            "../experiments/azure_openai/gpt-5-nano/token/fodo_zaga/20251225/02_38/",
            "../experiments/azure_openai/gpt-5-nano/attribute/fodo_zaga/20251225/21_11/",
        ],
        [
            "../experiments/azure_openai/gpt-5-nano/token/walmart_amazon/20251224/23_33/",
            "../experiments/azure_openai/gpt-5-nano/attribute/walmart_amazon/20251225/17_35/",
        ],
        [
            "../experiments/azure_openai/gpt-5-nano/token/watches_small/20251224/08_56/",
            "../experiments/azure_openai/gpt-5-nano/attribute/watches_small/20251223/17_40/",
        ],
    ],
    "llama-3.1": [
        [
            "../experiments/hf/meta-llama/Llama-3.1-8B-Instruct/token/walmart_amazon/20260103/00_53/",
            "../experiments/hf/meta-llama/Llama-3.1-8B-Instruct/attribute/walmart_amazon/20260102/14_25/",
        ],
        [
            "../experiments/hf/meta-llama/Llama-3.1-8B-Instruct/token/fodo_zaga/20260102/15_27/",
            "../experiments/hf/meta-llama/Llama-3.1-8B-Instruct/attribute/fodo_zaga/20260103/01_50/",
        ],
        [
            "../experiments/hf/meta-llama/Llama-3.1-8B-Instruct/token/abt_buy/20260103/03_05/",
            "../experiments/hf/meta-llama/Llama-3.1-8B-Instruct/attribute/abt_buy/20260102/16_56/",
        ],
    ]
}

if __name__ == '__main__':
    for k, v in precomputed_data.items():
        attribute_token_metrics = []
        for pair in v:
            for kt, ka in [
                [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [9, 1], [10, 1],
                [4, 2], [5, 2], [6, 2], [7, 2], [8, 2], [9, 2], [10, 2],
                [10, 3], [20, 3], [15, 3], [9, 3], [8, 3], [7, 3],
                [20, 4], [15, 4], [25, 4], [10, 4], [9, 4], [8, 4],
                [20, 5], [10, 5], [15, 5], [25, 5], [30, 5], [15, 5],
                [20, 6], [30, 6], [25, 6], [30, 6], [35, 6], [40, 6],
            ]:
                results = compare(pair, k, k_tokens=kt, k_attrs=ka)
                attribute_token_metrics.append(results)
        agreement_stats = pd.concat(attribute_token_metrics).groupby(["model", "explainer"]).mean()
        agreement_stats.to_csv(f"../experiments/attribute_token_agreement_{k}.csv")
        toprint = agreement_stats
        print(f'{k}:\n{toprint}')

'''

                
                '''
