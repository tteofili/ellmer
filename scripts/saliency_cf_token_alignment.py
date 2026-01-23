import json
import re
from collections import defaultdict

# -----------------------------
# Utility functions
# -----------------------------

def tokenize(text):
    """Simple whitespace + punctuation tokenizer."""
    if text is None:
        return set()
    return set(re.findall(r"\w+", text.lower()))

def extract_tokens_from_record(record, side):
    """
    Extract token sets per attribute from a record.
    Returns: { f"{side}_{attr}": set(tokens) }
    """
    tokens = {}
    for k, v in record.items():
        if k.startswith(f"{side}_") and isinstance(v, str):
            tokens[k] = tokenize(v)
    return tokens

def get_counterfactual_changed_tokens(original, cf, side):
    """
    Compute tokens changed by the counterfactual (symmetric difference).
    Returns a set of feature-level token keys:
    e.g. ltable_description__pink
    """
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

# -----------------------------
# Metrics
# -----------------------------

def top_k_overlap(saliency, changed_tokens, k):
    """
    Recall-style Top-k Overlap:
    |Top-k saliency ∩ CF tokens| / |CF tokens|
    """
    if not changed_tokens:
        return 0.0

    ranked = sorted(saliency.items(), key=lambda x: x[1], reverse=True)
    top_k = {feat for feat, _ in ranked[:k]}

    return len(top_k & changed_tokens) / len(changed_tokens)

def attribution_mass_on_cf(saliency, changed_tokens):
    """
    Attribution mass on counterfactual tokens:
    sum saliency on CF tokens / total saliency mass
    """
    total_mass = sum(saliency.values())
    if total_mass == 0:
        return 0.0

    cf_mass = sum(v for f, v in saliency.items() if f in changed_tokens)
    return cf_mass / total_mass

# -----------------------------
# Main computation
# -----------------------------

def flatten(param):
    if type(param) == list:
        return param[0]
    else:
        return param


def compute_metrics(json_path, k=10):
    with open(json_path) as f:
        data = json.load(f)["data"]

    results = []

    for item in data:
        # --- saliency (flatten lists) ---
        saliency = {k: flatten(v) for k, v in item["saliency"].items()}

        # --- original records ---
        original = {}
        original.update(dict(map(lambda item: ("ltable_" + item[0], item[1]), item['ltuple'].items())))
        original.update(dict(map(lambda item: ("rtable_" + item[0], item[1]), item['rtuple'].items())))

        # --- assume 1 counterfactual per instance ---
        cf = item["cfs"][0]

        if len(cf) == 0:
            continue

        changed_tokens = set()
        changed_tokens |= get_counterfactual_changed_tokens(original, cf, "ltable")
        changed_tokens |= get_counterfactual_changed_tokens(original, cf, "rtable")

        tk = top_k_overlap(saliency, changed_tokens, k)
        mass = attribution_mass_on_cf(saliency, changed_tokens)

        results.append({
            "id": item["id"],
            "top_k_overlap": tk,
            "attribution_mass_cf": mass,
            "num_cf_tokens": len(changed_tokens)
        })

    # Aggregate
    avg_topk = sum(r["top_k_overlap"] for r in results) / len(results)
    avg_mass = sum(r["attribution_mass_cf"] for r in results) / len(results)

    return results, {
        "avg_top_k_overlap": avg_topk,
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

def plot_topk_curve(topk_curve):
    ks = sorted(topk_curve.keys())
    values = [topk_curve[k] for k in ks]

    plt.figure()
    plt.plot(ks, values, marker="o")
    plt.xlabel("k (Top-k salient tokens)")
    plt.ylabel("Top-k Overlap")
    plt.title("Saliency–Counterfactual Alignment (Top-k Overlap)")
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




# -----------------------------
# Run
# -----------------------------

if __name__ == "__main__":
    json_path = "../experiments/azure_openai/gpt-5-nano/token/abt_buy/20251225/06_03/"
    ks = [1, 3, 5, 10, 20, 50]
    topk_curves = dict()
    for explainer in ['zs_sample', 'hybrid_sample', 'fs_sample', 'cot_sample', 'certa_sample']:
        print(f'{explainer}:')
        json_path_cur = json_path + explainer + "_results.json"
        with open(json_path_cur) as f:
            data = json.load(f)["data"]

        results, summary = compute_metrics(json_path_cur, k=10)
        topk_curve = compute_topk_curve(data, ks)
        topk_curves[explainer] = topk_curve

        print("Summary:")
        for k, v in summary.items():
            print(f"{k}: {v:.4f}")

        plot_topk_curve(topk_curve)
        plot_attribution_mass_distribution(results)
        plot_cf_size_vs_mass(results)

    plot_topk_curves(topk_curves)


