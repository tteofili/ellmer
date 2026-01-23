import json
from collections import defaultdict

with open("../experiments/azure_openai/gpt-5-nano/token/abt_buy/20251225/06_03/zs_sample_results.json") as f:
    token_data = {d["id"]: d for d in json.load(f)["data"]}

with open("../experiments/azure_openai/gpt-5-nano/attribute/abt_buy/20251226/01_45/zs_sample_results.json") as f:
    attr_data = {d["id"]: d for d in json.load(f)["data"]}


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

    top_tokens = sorted(token_sal.items(), key=lambda x: x[1], reverse=True)[:k_tokens]
    token_attrs = {get_attribute_from_token(t) for t, _ in top_tokens}

    top_attrs = {
        a for a, _ in sorted(attr_sal.items(), key=lambda x: x[1], reverse=True)[:k_attrs]
    }

    if not top_attrs:
        return 0.0

    return len(token_attrs & top_attrs) / len(top_attrs)


def attribution_mass_consistency(token_sal, attr_sal):
    """Computes consistency score between token and attribute attributions"""
    token_sal = {k: flatten(v) for k, v in token_sal.items()}
    attr_sal = {k: flatten(v) for k, v in attr_sal.items()}

    token_mass = defaultdict(float)
    for tok, val in token_sal.items():
        token_mass[get_attribute_from_token(tok)] += val

    total_tok = sum(token_mass.values())
    total_attr = sum(attr_sal.values())

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


results = []

for id_ in token_data:
    tok = token_data[id_]["saliency"]
    attr = attr_data[id_]["saliency"]

    if not tok or not attr:
        continue

    results.append({
        "id": id_,
        "topk_attr_token_overlap": topk_attr_token_overlap(tok, attr),
        "attr_token_mass_consistency": attribution_mass_consistency(tok, attr)
    })

avg_topk = sum(r["topk_attr_token_overlap"] for r in results) / len(results)
avg_mass = sum(r["attr_token_mass_consistency"] for r in results) / len(results)

print("Average Top-k Attribute–Token Overlap:", avg_topk)
print("Average Attribute–Token Mass Consistency:", avg_mass)


results = []

for id_ in token_data:
    tok_item = token_data[id_]
    attr_item = attr_data[id_]

    original = {}
    original.update(dict(map(lambda item: ("ltable_" + item[0], item[1]), tok_item['ltuple'].items())))
    original.update(dict(map(lambda item: ("rtable_" + item[0], item[1]), tok_item['rtuple'].items())))

    tok_cf = tok_item["cfs"][0]
    attr_cf = attr_item["cfs"][0]

    if not tok_cf or not attr_cf:
        continue

    token_cf_tokens = set()
    token_cf_tokens |= extract_cf_tokens(original, tok_cf, "ltable")
    token_cf_tokens |= extract_cf_tokens(original, tok_cf, "rtable")

    attr_cf_attrs = extract_cf_attributes(original, attr_cf)

    cov, prec, f1 = cf_attr_token_alignment(token_cf_tokens, attr_cf_attrs)

    results.append({
        "id": id_,
        "cf_attr_coverage": cov,
        "cf_attr_precision": prec,
        "cf_attr_f1": f1
    })

print("Avg Coverage:", sum(r["cf_attr_coverage"] for r in results) / len(results))
print("Avg Precision:", sum(r["cf_attr_precision"] for r in results) / len(results))
print("Avg F1:", sum(r["cf_attr_f1"] for r in results) / len(results))

