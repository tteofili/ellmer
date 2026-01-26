import json
from collections import defaultdict

import pandas as pd


def flatten(param):
    if type(param) == list:
        return param[0]
    else:
        return param


def get_attribute_from_token(token_feat):
    # ltable_name__sony -> ltable_name
    return token_feat.split("__")[0]


def topk_attr_token_overlap(token_sal, attr_sal, k_tokens=20, k_attrs=3):
    # flatten token saliency
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


def attribution_mass_consistency(token_sal, attr_sal):
    """Computes consistency score between token and attribute attributions"""
    token_sal = {k: flatten(v) for k, v in token_sal.items()}
    attr_sal = {k: flatten(v) for k, v in attr_sal.items()}

    token_mass = defaultdict(float)
    try:
        for tok, val in token_sal.items():
            token_mass[get_attribute_from_token(tok)] += val

        total_tok = sum(token_mass.values())
        total_attr = sum(attr_sal.values())
    except:
        return 0.0

    if total_tok == 0 or total_attr == 0:
        return 0.0

    score = 0.0
    # Accumulates minimum cooccurrence probability for each attribute
    for a in set(token_mass) | set(attr_sal):
        p_tok = token_mass.get(a, 0.0) / total_tok
        p_attr = attr_sal.get(a, 0.0) / total_attr
        score += min(p_tok, p_attr)

    return score


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


def extract_cf_attributes(original, cf):
    changed = set()
    for k in original:
        if original.get(k) != cf.get(k):
            changed.add(k)
    return changed


def cf_attr_token_alignment(token_cf_tokens, attr_cf_attrs):
    token_attrs = {get_attribute_from_token(t) for t in token_cf_tokens}

    if not attr_cf_attrs:
        return 0.0, 0.0, 0.0

    intersection = token_attrs & attr_cf_attrs

    coverage = len(intersection) / len(attr_cf_attrs)
    precision = len(intersection) / len(token_attrs) if token_attrs else 0.0

    if coverage + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * coverage * precision / (coverage + precision)

    return coverage, precision, f1


def compare(pair, model, verbose: bool = False,
            explainers=['zs_sample', 'hybrid_sample', 'fs_sample', 'cot_sample', 'certa_sample'], ):
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

            if not tok or not attr:
                continue
            try:
                results.append({
                    "id": id_,
                    "topk_attr_token_overlap": topk_attr_token_overlap(tok, attr),
                    "attr_token_mass_consistency": attribution_mass_consistency(tok, attr)
                })
            except:
                continue
        if len(results) == 0:
            continue
        avg_topk = sum(r["topk_attr_token_overlap"] for r in results) / len(results)
        avg_mass = sum(r["attr_token_mass_consistency"] for r in results) / len(results)

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
                attr_cf_attrs = set()

            cov, prec, f1 = cf_attr_token_alignment(token_cf_tokens, attr_cf_attrs)

            results.append({
                "id": id_,
                "cf_attr_coverage": cov,
                "cf_attr_precision": prec,
                "cf_attr_f1": f1
            })
        if len(results) == 0:
            continue
        avg_coverage = sum(r["cf_attr_coverage"] for r in results) / len(results)
        avg_prec = sum(r["cf_attr_precision"] for r in results) / len(results)
        avg_f1 = sum(r["cf_attr_f1"] for r in results) / len(results)
        if verbose:
            print("Avg Coverage:", avg_coverage)
            print("Avg Precision:", avg_prec)
            print("Avg F1:", avg_f1)
        agg_results.append({"model": model, "explainer": explainer, "coverage": avg_coverage, "prec": avg_prec,
                            "f1": avg_f1, "topk_overlap": avg_topk, "attr_token_mass": avg_mass})
    return pd.DataFrame(agg_results)


precomputed_data = {
    "chatgpt4": [
        [
            "../experiments/azure_openai/gpt-4-32k/token/abt_buy/20260125/23_49/",
            "../experiments/azure_openai/gpt-4-32k/attribute/abt_buy/20241014/22_34/",
        ],
        [
            "../experiments/azure_openai/gpt-4-32k/token/beers/20260126/01_49/",
            "../experiments/azure_openai/gpt-4-32k/attribute/beers/20241014/23_41/",
        ],
        [
            "../experiments/azure_openai/gpt-4-32k/token/books/20260125/20_01/",
            "../experiments/azure_openai/gpt-4-32k/attribute/books/20241011/11_12/",
        ],
        [
            "../experiments/azure_openai/gpt-4-32k/token/carparts/20260125/21_58/",
            "../experiments/azure_openai/gpt-4-32k/attribute/carparts/20241011/09_19/",
        ],
        [
            "../experiments/azure_openai/gpt-4-32k/token/cameras_large/20260125/08_56/",
            "../experiments/azure_openai/gpt-4-32k/attribute/cameras_large/20260123/18_35/",
        ],
        [
            "../experiments/azure_openai/gpt-4-32k/token/fodo_zaga/20241014/08_25/",
            "../experiments/azure_openai/gpt-4-32k/attribute/fodo_zaga/20260126/03_49/",
        ],
        [
            "../experiments/azure_openai/gpt-4-32k/token/walmart_amazon/20241016/12_10/",
            "../experiments/azure_openai/gpt-4-32k/attribute/walmart_amazon/20260126/07_09/",
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
}

if __name__ == '__main__':
    for k, v in precomputed_data.items():
        attribute_token_metrics = []
        for pair in v:
            results = compare(pair, k)
            attribute_token_metrics.append(results)
        agreement_stats = pd.concat(attribute_token_metrics).groupby(["model", "explainer"]).mean()
        agreement_stats.to_csv(f"../experiments/attribute_token_agreement_{k}.csv")
        print(f'{k}:\n{agreement_stats}')
