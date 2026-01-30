import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd


def tokenize(text):
    if text is None:
        return set()
    return set(re.findall(r"\w+", text.lower()))


def extract_tokens_from_record(record, side):
    tokens = {}
    if type(record) == str:
        try:
            record = json.loads(record)
        except json.decoder.JSONDecodeError:
            return dict()
    for k, v in record.items():
        if k.startswith(f"{side}_") and isinstance(v, str):
            tokens[k] = tokenize(v)
    return tokens


def get_counterfactual_changed_tokens(original, cf, side):
    changed = set()

    orig_tokens = extract_tokens_from_record(original, side)
    cf_tokens = extract_tokens_from_record(cf, side)

    for attr in orig_tokens:
        orig = orig_tokens.get(attr, set())
        new = cf_tokens.get(attr, set())

        diff = orig.symmetric_difference(new)
        for t in diff:
            changed.add(f"{attr}__{t}")

    return changed


def top_k_overlap(saliency, changed_tokens, k):
    if not changed_tokens:
        return 0.0

    try:
        ranked = sorted(saliency.items(), key=lambda x: x[1], reverse=True)
        top_k = {feat for feat, _ in ranked[:k]}

        return len(top_k & changed_tokens) / len(changed_tokens)
    except:
        return 0.0


def attribution_mass_on_cf(saliency, changed_tokens):
    try:
        total_mass = sum(saliency.values())
        if total_mass == 0:
            return 0.0

        cf_mass = sum(v for f, v in saliency.items() if f in changed_tokens)
        return cf_mass / total_mass
    except:
        return 0.0


def flatten(param):
    if type(param) == list:
        return param[0]
    else:
        return param


def compute_metrics(json_path, ks=[1, 2, 3, 4, 5], granularity='token'):
    with open(json_path) as f:
        data = json.load(f)["data"]

    results = []

    for item in data:
        if type(item['saliency']) == str:
            continue

        # --- saliency (flatten lists) ---
        saliency = {k: flatten(v) for k, v in item["saliency"].items()}

        # --- original records ---
        original = {}
        if list(item['ltuple'].keys())[0].startswith('ltable_'):
            original.update({k: v for k, v in item['ltuple'].items()})
        else:
            original.update(dict(map(lambda item: ("ltable_" + item[0], item[1]), item['ltuple'].items())))
        if list(item['rtuple'].keys())[0].startswith('rtable_'):
            original.update({k: v for k, v in item['rtuple'].items()})
        else:
            original.update(dict(map(lambda item: ("rtable_" + item[0], item[1]), item['rtuple'].items())))

        # --- use 1 counterfactual per instance ---
        cf = item["cfs"][0]

        if len(cf) == 0:
            continue

        changed_tokens = set()
        changed_tokens |= get_counterfactual_changed_tokens(original, cf, "ltable")
        changed_tokens |= get_counterfactual_changed_tokens(original, cf, "rtable")

        if granularity == 'attribute':
            changed_tokens = set([t.split('__')[0] for t in changed_tokens])
            new_saliency = {}
            for tk, v in saliency.items():
                attrib = tk.split('__')[0]
                if attrib not in new_saliency:
                    new_saliency[attrib] = v
                else:
                    new_saliency[attrib] = new_saliency[attrib] + v
            saliency = new_saliency
        tks = []
        for k in ks:
            tks.append(top_k_overlap(saliency, changed_tokens, k))

        mass = attribution_mass_on_cf(saliency, changed_tokens)

        row = {
            "id": item["id"],
            "attribution_mass_cf": mass,
            "num_cf_tokens": len(changed_tokens)
        }
        idx_k = 1
        for tk in tks:
            row[f'top_k_overlap@{idx_k}'] = tk
            idx_k += 1
        row['avg_top_k_overlap'] = np.mean(tks)
        results.append(row)
    model_name = json_path.split('/')[3]
    dataset = json_path.split('/')[5]
    explainer = json_path.split('/')[-1].split('.')[0].replace('_sample_results','')
    pd.DataFrame.from_dict(results).to_csv(f'sal_cf_{model_name}_{dataset}_{explainer}', index=False)

    if len(results) > 0:
        # Aggregate
        avg_topk = sum(r["avg_top_k_overlap"] for r in results) / len(results)
        avg_mass = sum(r["attribution_mass_cf"] for r in results) / len(results)
    else:
        avg_topk = 0.0
        avg_mass = 0.0

    return results, {
        "xd_avg_top_k_overlap": avg_topk,
        "avg_attribution_mass_cf": avg_mass
    }


def compute_topk_curve(data, ks):
    curve = defaultdict(list)

    for item in data:
        saliency = {k: flatten(v) for k, v in item["saliency"].items()}

        original = {}
        original.update(dict(map(lambda item: ("ltable_" + item[0], item[1]), item['ltuple'].items())))
        original.update(dict(map(lambda item: ("rtable_" + item[0], item[1]), item['rtuple'].items())))

        cf = item["cfs"][0]

        if len(cf) == 0:
            continue

        changed_tokens = set()
        changed_tokens |= get_counterfactual_changed_tokens(original, cf, "ltable")
        changed_tokens |= get_counterfactual_changed_tokens(original, cf, "rtable")

        for k in ks:
            curve[k].append(top_k_overlap(saliency, changed_tokens, k))

    return {k: sum(v) / len(v) for k, v in curve.items()}


import matplotlib.pyplot as plt


def plot_topk_curves(topk_curves):
    plt.figure()
    for k, v in topk_curves.items():
        ks = sorted(v.keys())
        values = [topk_curve[k] for k in ks]
        plt.plot(ks, values, label=k)

    plt.xlabel("k (Top-k salient tokens)")
    plt.ylabel("Top-k Overlap")
    plt.title("Saliency–Counterfactual Alignment (Top-k Overlap)")
    plt.grid(True)
    plt.show()


def plot_topk_curve(topk_curve, label):
    ks = sorted(topk_curve.keys())
    values = [topk_curve[k] for k in ks]

    plt.figure()
    plt.plot(ks, values, marker="o")
    plt.xlabel("k (Top-k salient tokens)")
    plt.ylabel("Top-k Overlap")
    plt.title(f"{label}: Saliency–Counterfactual Alignment (Top-k Overlap)")
    plt.grid(True)
    plt.show()


def plot_attribution_mass_distribution(results):
    values = [r["attribution_mass_cf"] for r in results]

    plt.figure()
    plt.hist(values, bins=20)
    plt.xlabel("Attribution Mass on Counterfactual Tokens")
    plt.ylabel("Frequency")
    plt.title("Distribution of Attribution Mass on CF Tokens")
    plt.grid(True)
    plt.show()


def plot_cf_size_vs_mass(results):
    x = [r["num_cf_tokens"] for r in results]
    y = [r["attribution_mass_cf"] for r in results]

    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("Number of Counterfactual Tokens")
    plt.ylabel("Attribution Mass on CF Tokens")
    plt.title("Counterfactual Size vs Saliency Faithfulness")
    plt.grid(True)
    plt.show()


def compare(json_path, explainers=['zs_sample', 'hybrid_sample', 'fs_sample', 'cot_sample', 'certa_sample'],
            ks=[1, 3, 5, 10, 20, 50], plot: bool = False, verbose: bool = False, granularity: str = 'token'):
    topk_curves = dict()
    eval_results = []
    for explainer in explainers:
        json_path_cur = json_path + explainer + "_results.json"
        try:
            with open(json_path_cur) as f:
                data = json.load(f)["data"]
        except:
            continue

        results, summary = compute_metrics(json_path_cur, granularity=granularity)
        if plot:
            topk_curve = compute_topk_curve(data, ks)
            topk_curves[explainer] = topk_curve
        summary['explainer'] = explainer.replace("_sample", "")
        eval_results.append(summary)
        if verbose:
            print("Summary:")
            for k, v in summary.items():
                print(f"{k}: {v:.4f}")

        if plot:
            plot_topk_curve(topk_curve, explainer)
            plot_attribution_mass_distribution(results)
            plot_cf_size_vs_mass(results)
    if plot:
        plot_topk_curves(topk_curves)
    return pd.DataFrame(eval_results)


precomputed_data = {
    "chatgpt4-attribute": [
        "../experiments/azure_openai/gpt-4-32k/attribute/abt_buy/20260126/11_32/",
        "../experiments/azure_openai/gpt-4-32k/attribute/beers/20260126/11_51/",
        "../experiments/azure_openai/gpt-4-32k/attribute/fodo_zaga/20260126/12_25/",
        "../experiments/azure_openai/gpt-4-32k/attribute/walmart_amazon/20260126/13_36/",
        "../experiments/azure_openai/gpt-4-32k/attribute/carparts/20241014/09_19/",
        "../experiments/azure_openai/gpt-4-32k/attribute/books/20241011/09_31/",

    ],
    "chatgpt4-token": [
        "../experiments/azure_openai/gpt-4-32k/token/abt_buy/20260125/23_49/",
        "../experiments/azure_openai/gpt-4-32k/token/beers/20260125/01_49/",
        "../experiments/azure_openai/gpt-4-32k/token/amazon_google/20260126/05_32/",
        "../experiments/azure_openai/gpt-4-32k/token/walmart_amazon/20260126/07_09/",
        "../experiments/azure_openai/gpt-4-32k/token/watches_large/20260125/13_37/",
        "../experiments/azure_openai/gpt-4-32k/token/cameras_large/20260125/08_56/",
        "../experiments/azure_openai/gpt-4-32k/token/books/20260125/20_10/",
        "../experiments/azure_openai/gpt-4-32k/token/carparts/20260125/20_10/",
    ],
    "chatgpt5-attribute": [
        "../experiments/azure_openai/gpt-5-nano/attribute/abt_buy/20251226/01_45/",
        "../experiments/azure_openai/gpt-5-nano/attribute/amazon_google/20251225/15_03/",
        "../experiments/azure_openai/gpt-5-nano/attribute/beers/20251225/11_58/",
        "../experiments/azure_openai/gpt-5-nano/attribute/fodo_zaga/20251225/21_11/",
        "../experiments/azure_openai/gpt-5-nano/attribute/walmart_amazon/20251225/17_35/",
        "../experiments/azure_openai/gpt-5-nano/attribute/watches_small/20251223/17_40/",
        "../experiments/azure_openai/gpt-5-nano/attribute/cameras_small/20251223/22_36/",
        "../experiments/azure_openai/gpt-5-nano/attribute/books/20251110/15_04/",

    ],
    "chatgpt5-token": [
        "../experiments/azure_openai/gpt-5-nano/token/abt_buy/20251225/06_03/",
        "../experiments/azure_openai/gpt-5-nano/token/amazon_google/20251224/21_01/",
        "../experiments/azure_openai/gpt-5-nano/token/beers/20251224/17_34/",
        "../experiments/azure_openai/gpt-5-nano/token/fodo_zaga/20251225/02_38/",
        "../experiments/azure_openai/gpt-5-nano/token/walmart_amazon/20251224/23_33/",
        "../experiments/azure_openai/gpt-5-nano/token/watches_small/20251224/08_56/",
        "../experiments/azure_openai/gpt-5-nano/token/watches_small/20251217/10_12/",
        "../experiments/azure_openai/gpt-5-nano/token/cameras_small/20251224/13_15/",
        "../experiments/azure_openai/gpt-5-nano/token/books/20250917/10_53/",
    ],
    "llama-3.1-attribute": [
        "../experiments/hf/meta-llama/Llama-3.1-8B-Instruct/attribute/carparts/20260108/00_44/",
        "../experiments/hf/meta-llama/Llama-3.1-8B-Instruct/attribute/books/20260107/23_42/",
        "../experiments/hf/meta-llama/Llama-3.1-8B-Instruct/attribute/cameras_small/20260125/17_07/",
        "../experiments/hf/meta-llama/Llama-3.1-8B-Instruct/attribute/walmart_amazon/20260102/14_25/",
        "../experiments/hf/meta-llama/Llama-3.1-8B-Instruct/attribute/fodo_zaga/20260103/01_50/",
        "../experiments/hf/meta-llama/Llama-3.1-8B-Instruct/attribute/abt_buy/20260102/16_56/",
        "../experiments/hf/meta-llama/Llama-3.1-8B-Instruct/attribute/beers/20260102/12_51/",
        "../experiments/hf/meta-llama/Llama-3.1-8B-Instruct/attribute/amazon_google/20260102/13_43/",
    ],
    "llama-3.1-token": [
        "../experiments/hf/meta-llama/Llama-3.1-8B-Instruct/token/walmart_amazon/20260103/00_53/",
        "../experiments/hf/meta-llama/Llama-3.1-8B-Instruct/token/fodo_zaga/20260103/01_50/",
        "../experiments/hf/meta-llama/Llama-3.1-8B-Instruct/token/abt_buy/20260103/03_05/",
        "../experiments/hf/meta-llama/Llama-3.1-8B-Instruct/token/amazon_google/20260102/23_58/",
        "../experiments/hf/meta-llama/Llama-3.1-8B-Instruct/token/beers/20260102/23_04/",
        "../experiments/hf/meta-llama/Llama-3.1-8B-Instruct/token/books/20250412/23_04/",

    ]

}

if __name__ == '__main__':
    for k, v in precomputed_data.items():
        saliency_token_metrics = []
        for path in v:
            saliency_token_metrics.append(compare(path, granularity=k.split('-')[1]))
        saliency_cf_agreement = pd.concat(saliency_token_metrics).groupby("explainer").mean()
        saliency_cf_agreement.to_csv(f"../experiments/saliency_cf_agreement_{k}.csv")
        print(f'{k}:\n{saliency_cf_agreement}')
