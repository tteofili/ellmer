# BSD-3-Clause
# Counterfactual search adapted from Minun (DittoExplainer:
# https://github.com/megagonlabs/minun/blob/main/ditto_explainer.py).
# Copyright (c) Megagon Labs; used here under BSD-3-Clause with minimal changes
# for ellmer DataFrame predict_fn and attribute masks.

from __future__ import annotations

from itertools import product
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ellmer.post_hoc.explanation_mask import ExplanationMask


def _min_cost_path(cost, operations):
    path = [operations[cost.shape[0] - 1][cost.shape[1] - 1]]
    row = cost.shape[0] - 1
    col = cost.shape[1] - 1
    while row > 0 and col > 0:
        if cost[row - 1][col - 1] <= cost[row - 1][col] and cost[row - 1][col - 1] <= cost[row][col - 1]:
            path.append(operations[row - 1][col - 1])
            row -= 1
            col -= 1
        elif cost[row - 1][col] <= cost[row - 1][col - 1] and cost[row - 1][col] <= cost[row][col - 1]:
            path.append(operations[row - 1][col])
            row -= 1
        else:
            path.append(operations[row][col - 1])
            col -= 1
    return "".join(path[::-1][1:])


def token_edit_distance(str1: str, str2: str):
    seq1 = str1.split(" ")
    seq2 = str2.split(" ")
    if len(str1) == 0 and len(str2) == 0:
        return 0, []
    if len(str1) == 0:
        return len(seq2), ["I"] * len(seq2)
    if len(str2) == 0:
        return len(seq1), ["D"] * len(seq1)
    matrix = np.zeros((len(seq1) + 1, len(seq2) + 1))
    matrix[0] = [i for i in range(len(seq2) + 1)]
    matrix[:, 0] = [i for i in range(len(seq1) + 1)]
    ops = np.asarray([["-" for _ in range(len(seq2) + 1)] for _ in range(len(seq1) + 1)])
    ops[0] = ["I" for _ in range(len(seq2) + 1)]
    ops[:, 0] = ["D" for _ in range(len(seq1) + 1)]
    ops[0, 0] = "-"
    for row in range(1, len(seq1) + 1):
        for col in range(1, len(seq2) + 1):
            if seq1[row - 1] == seq2[col - 1]:
                matrix[row][col] = matrix[row - 1][col - 1]
            else:
                insertion_cost = matrix[row][col - 1] + 1
                deletion_cost = matrix[row - 1][col] + 1
                substitution_cost = matrix[row - 1][col - 1] + 1
                matrix[row][col] = min(insertion_cost, deletion_cost, substitution_cost)
                if matrix[row][col] == substitution_cost:
                    ops[row][col] = "S"
                elif matrix[row][col] == insertion_cost:
                    ops[row][col] = "I"
                else:
                    ops[row][col] = "D"
    dist = int(matrix[len(seq1), len(seq2)])
    operations = _min_cost_path(matrix, ops)
    return dist, operations


def generate_candidates_from_one_attribute_pos(attr1: str, attr2: str) -> List[str]:
    candidates = [""]
    seq = attr1.split(" ")
    tmp_str = ""
    for token in seq:
        tmp_str += str(token) + " "
        candidates.append(tmp_str.strip())
    return candidates[::-1]


def generate_candidates_from_one_attribute_neg(attr1: str, attr2: str) -> List[str]:
    candidates = [attr1]
    _dist, operations = token_edit_distance(attr1, attr2)
    tmp = attr1.split(" ")
    target = attr2.split(" ")
    if len(target) == 0:
        return candidates
    cur = 0
    dnum = 0
    for idx, op in enumerate(operations):
        if op == "-":
            cur += 1
            continue
        if op == "S":
            pos1 = idx - dnum
            pos2 = cur
            if pos1 >= len(tmp):
                pos1 = -1
            if pos2 >= len(target):
                pos2 = -1
            tmp[pos1] = target[pos2]
            cur += 1
        elif op == "I":
            pos3 = cur
            if pos3 >= len(target):
                pos3 = -1
            tmp.insert(idx, target[pos3])
            cur += 1
        else:
            pos4 = idx - dnum
            if pos4 >= len(tmp):
                pos4 = -1
            del tmp[pos4]
            dnum += 1
        tmp_str = ""
        for token in tmp:
            tmp_str += str(token) + " "
        candidates.append(tmp_str.strip())
    return candidates


def align_attr_names(ltuple: dict, rtuple: dict) -> List[str]:
    return sorted((set(ltuple) | set(rtuple)) - {"id"})


def minun_attr_is_masked(attr_name: str, mask: ExplanationMask) -> bool:
    if mask.granularity == "attribute":
        return f"ltable_{attr_name}" in mask.allowed
    return any(x.startswith(f"ltable_{attr_name}__") for x in mask.allowed)


def build_candidate_lists(
    left: Sequence[str],
    right: Sequence[str],
    attr_names: Sequence[str],
    mask: ExplanationMask,
    positive_class: int,
) -> List[List[str]]:
    cands = []
    for i, name in enumerate(attr_names):
        if not minun_attr_is_masked(name, mask):
            cands.append([left[i]])
            continue
        if positive_class == 1:
            cands.append(generate_candidates_from_one_attribute_pos(left[i], right[i]))
        else:
            cands.append(generate_candidates_from_one_attribute_neg(left[i], right[i]))
    return cands


def wide_from_entities(ltuple: dict, rtuple: dict) -> pd.DataFrame:
    row = {}
    for k, v in ltuple.items():
        row["ltable_" + str(k)] = v
    for k, v in rtuple.items():
        row["rtable_" + str(k)] = v
    return pd.DataFrame([row])


def _match_prob(predict_fn: Callable[[pd.DataFrame], pd.DataFrame], row: pd.DataFrame) -> float:
    out = predict_fn(row)
    return float(out["match_score"].values[0])


def prediction_flipped(match_prob: float, positive_class: int) -> bool:
    if positive_class == 1:
        return match_prob < 0.5
    return match_prob >= 0.5


def cf_dict_from_edit(ltuple: dict, rtuple: dict, attr_names: Sequence[str], new_left: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i, name in enumerate(attr_names):
        old = str(ltuple.get(name, ""))
        new = str(new_left[i])
        if new != old:
            out["ltable_" + name] = new
    return out


def minun_counterfactual_greedy(
    ltuple: dict,
    rtuple: dict,
    predict_fn: Callable[[pd.DataFrame], pd.DataFrame],
    mask: ExplanationMask,
    positive_class: int,
    *,
    k: int = 10,
    max_evals: int = 800,
) -> Tuple[Dict[str, str], int]:
    attr_names = align_attr_names(ltuple, rtuple)
    left = [str(ltuple.get(a, "")) for a in attr_names]
    right = [str(rtuple.get(a, "")) for a in attr_names]
    cands4attrs = build_candidate_lists(left, right, attr_names, mask, positive_class)

    eval_cnt = 0
    candidates_found = []
    perms = [range(len(c)) for c in cands4attrs]
    for indices in product(*perms):
        if sum(indices) == 0:
            continue
        if eval_cnt >= max_evals:
            break
        new_left = [cands4attrs[i][indices[i]] for i in range(len(attr_names))]
        lt = dict(ltuple)
        for i, name in enumerate(attr_names):
            lt[name] = new_left[i]
        row = wide_from_entities(lt, rtuple)
        mp = _match_prob(predict_fn, row)
        eval_cnt += 1
        if prediction_flipped(mp, positive_class):
            num = sum(1 for i in range(len(indices)) if indices[i] > 0)
            candidates_found.append((num, mp, indices, new_left))
            if len(candidates_found) >= k:
                break

    if not candidates_found:
        return {}, eval_cnt

    candidates_found.sort(key=lambda t: (t[0], -t[1]))
    _n, _mp, _idx, best_left = candidates_found[0]
    return cf_dict_from_edit(ltuple, rtuple, attr_names, best_left), eval_cnt


def minun_counterfactual_binary(
    ltuple: dict,
    rtuple: dict,
    predict_fn: Callable[[pd.DataFrame], pd.DataFrame],
    mask: ExplanationMask,
    positive_class: int,
    *,
    k: int = 10,
    max_evals: int = 800,
) -> Tuple[Dict[str, str], int]:
    """
    Single-attribute binary search over candidate indices per dimension (Minun ``attr_num == 1`` case).
    Falls back to greedy when no flip is found.
    """
    attr_names = align_attr_names(ltuple, rtuple)
    left = [str(ltuple.get(a, "")) for a in attr_names]
    right = [str(rtuple.get(a, "")) for a in attr_names]
    cands4attrs = build_candidate_lists(left, right, attr_names, mask, positive_class)
    perms = [len(c) for c in cands4attrs]
    num_dim = len(cands4attrs)
    eval_cnt = 0
    candidates = []

    for dim in range(num_dim):
        if perms[dim] <= 1 or eval_cnt >= max_evals:
            continue
        cur_indice = [0] * num_dim
        low = 0
        high = perms[dim] - 1
        while low < high and eval_cnt < max_evals:
            mid = int((high + low) / 2)
            cur_indice[dim] = mid
            new_left = [cands4attrs[i][cur_indice[i]] for i in range(num_dim)]
            lt = dict(ltuple)
            for i, name in enumerate(attr_names):
                lt[name] = new_left[i]
            row = wide_from_entities(lt, rtuple)
            mp = _match_prob(predict_fn, row)
            eval_cnt += 1
            if prediction_flipped(mp, positive_class):
                num = int(sum(cur_indice))
                candidates.append((num, mp, list(cur_indice), list(new_left)))
                if len(candidates) >= k:
                    break
                high = mid - 1
            else:
                low = mid + 1
        cur_indice[dim] = 0

        if len(candidates) >= k:
            break

    if candidates:
        candidates.sort(key=lambda t: (t[0], -t[1]))
        _n, _mp, _ci, best_left = candidates[0]
        return cf_dict_from_edit(ltuple, rtuple, attr_names, best_left), eval_cnt

    return minun_counterfactual_greedy(
        ltuple, rtuple, predict_fn, mask, positive_class, k=k, max_evals=max(0, max_evals - eval_cnt)
    )


def minun_counterfactual(
    ltuple: dict,
    rtuple: dict,
    predict_fn: Callable[[pd.DataFrame], pd.DataFrame],
    mask: ExplanationMask,
    positive_class: int,
    *,
    method: str = "greedy",
    k: int = 10,
    max_evals: int = 800,
) -> Tuple[Dict[str, str], int]:
    if method == "greedy":
        return minun_counterfactual_greedy(
            ltuple, rtuple, predict_fn, mask, positive_class, k=k, max_evals=max_evals
        )
    if method == "binary":
        return minun_counterfactual_binary(
            ltuple, rtuple, predict_fn, mask, positive_class, k=k, max_evals=max_evals
        )
    raise ValueError(f"Unknown Minun method {method!r}; use 'greedy' or 'binary'")
