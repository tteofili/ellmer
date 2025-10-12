from collections import defaultdict
from functools import partialmethod
from itertools import combinations, chain

import pandas as pd
import logging
import functools
from joblib import Parallel, delayed

import random
from copy import deepcopy
import nltk
import numpy as np
from tqdm import tqdm
from nltk.util import ngrams

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

SML = 10


def _renameColumnsWithPrefix(prefix, df):
    newcol = []
    for col in list(df):
        newcol.append(prefix + col)
    df.columns = newcol


def _powerset(xs, minlen, maxlen):
    return [subset for i in range(minlen, maxlen + 1)
            for subset in combinations(xs, i)]


def get_triangles(dataset, sources):
    dataset_c = dataset.copy()
    dataset_c['ltable_id'] = list(map(lambda lrid: str(lrid).split("#")[0], dataset_c.id.values))
    dataset_c['rtable_id'] = list(map(lambda lrid: str(lrid).split("#")[1], dataset_c.id.values))

    sourcesmap = {}
    for i in range(len(sources)):
        sourcesmap[i] = sources[i]
    sample_ids_df = dataset_c[['ltable_id', 'rtable_id', 'label']].copy()
    original_prediction = sample_ids_df.iloc[0]
    triangles = []
    for idx in range(1, len(sample_ids_df)):
        support_prediction = sample_ids_df.iloc[idx]
        if original_prediction.rtable_id == support_prediction.rtable_id:  # left open triangle
            triangle = (
                original_prediction['ltable_id'], original_prediction['rtable_id'], support_prediction['ltable_id'])
        else:
            triangle = (
                original_prediction['rtable_id'], original_prediction['ltable_id'], support_prediction['rtable_id'])
        triangles.append(triangle)
    return triangles, sourcesmap


def getMixedTriangles(dataset,
                      sources):  # returns a list of triangles as tuples (free, pivot, support) and a dictionary of left and right sources
    # a triangle is a triple <u, v, w> where <u, v> is a match and <v, w> is a non-match (<u,w> should be a non-match)
    triangles = []
    # to not alter original dataset
    dataset_c = dataset.copy()
    sourcesmap = {}
    # the id is so composed: lsourcenumber@id#rsourcenumber@id
    for i in range(len(sources)):
        sourcesmap[i] = sources[i]
    dataset_c['ltable_id'] = list(map(lambda lrid: str(lrid).split("#")[0], dataset_c.id.values))
    dataset_c['rtable_id'] = list(map(lambda lrid: str(lrid).split("#")[1], dataset_c.id.values))
    positives = dataset_c[dataset_c.label == 1].astype('str')  # match classified samples
    negatives = dataset_c[dataset_c.label == 0].astype('str')  # no-match classified samples
    l_pos_ids = positives.ltable_id.astype('str').values  # left ids of positive samples
    r_pos_ids = positives.rtable_id.astype('str').values  # right ids of positive samples
    for lid, rid in zip(l_pos_ids, r_pos_ids):  # iterate through positive l_id, r_id pairs
        if np.count_nonzero(
                negatives.rtable_id.values == rid) >= 1:  # if r_id takes part also in a negative predictions
            relatedTuples = negatives[
                negatives.rtable_id == rid]  # find all tuples where r_id participates in a negative prediction
            for curr_lid in relatedTuples.ltable_id.values:  # collect all other l_ids that also are part of the negative prediction
                # add a new triangle with l_id1, a r_id1 participating in a positive prediction (with l_id1), and another l_id2 that participates in a negative prediction with r_id1
                triangles.append((lid, rid, curr_lid))
        if np.count_nonzero(
                negatives.ltable_id.values == lid) >= 1:  # dual but starting from l_id1 in positive prediction with r_id1, looking for r_id2s where l_id participates in a negative prediction
            relatedTuples = negatives[negatives.ltable_id == lid]
            for curr_rid in relatedTuples.rtable_id.values:
                triangles.append((rid, lid, curr_rid))
    return triangles, sourcesmap


def __get_records(sourcesMap, triangleIds, lprefix, rprefix):
    triangle = []
    for sourceid_recordid in triangleIds:
        split = str(sourceid_recordid).split("@")
        source_index = int(split[0])
        if source_index == 0:
            prefix = lprefix
        else:
            prefix = rprefix
        currentSource = sourcesMap[source_index]
        currentRecordId = int(split[1])
        currentRecord = currentSource[currentSource[prefix + 'id'] == currentRecordId].iloc[0]
        triangle.append(currentRecord)
    return triangle


def create_perturbations_from_triangle(triangleIds, sourcesMap, attributes, max_len_attribute_sets, lprefix, rprefix):
    # generate power set of attributes
    all_attributes_subsets = list(_powerset(attributes, max_len_attribute_sets, max_len_attribute_sets))
    triangle = __get_records(sourcesMap, triangleIds, lprefix, rprefix)  # get triangle values
    perturbations = []
    perturbed_attributes = []
    dropped_values = []
    copied_values = []

    for subset in all_attributes_subsets:  # iterate over the attribute power set
        dv = []
        cv = []
        new_record = triangle[0].copy()
        if not all(elem in new_record.index.to_list() for elem in subset):
            continue
        perturbed_attributes.append(subset)
        for att in subset:
            dv.append(new_record[att])
            cv.append(triangle[2][att])
            new_record[att] = triangle[2][att]
        perturbations.append(new_record)
        dropped_values.append(dv)
        copied_values.append(cv)
    perturbations_df = pd.DataFrame(perturbations, index=np.arange(len(perturbations)))
    r2 = triangle[1].copy()
    r2_copy = [r2] * len(perturbations_df)
    r2_df = pd.DataFrame(r2_copy, index=np.arange(len(perturbations)))
    if perturbations_df.columns[0].startswith(lprefix):
        all_perturbations = pd.concat([perturbations_df, r2_df], axis=1)
    else:
        all_perturbations = pd.concat([r2_df, perturbations_df], axis=1)
    all_perturbations = all_perturbations.drop([lprefix + 'id', rprefix + 'id'], axis=1)
    all_perturbations['altered_attributes'] = perturbed_attributes
    all_perturbations['droppedValues'] = dropped_values
    all_perturbations['copiedValues'] = copied_values

    return all_perturbations


def token_perturbations_from_triangle(triangle_ids, sources_map, attributes, max_len_attribute_set, class_to_explain,
                                      lprefix, rprefix, idx=None, check=False, predict_fn=None, min_t=0.45, max_t=0.55):
    all_good = False
    triangle = __get_records(sources_map, triangle_ids, lprefix, rprefix)  # get triangle values
    if class_to_explain == 1:
        support = triangle[2].copy()
        free = triangle[0].copy()
    else:
        support = triangle[0].copy()
        free = triangle[2].copy()

    # generate power set of token-attributes
    all_attributes_subsets = list(_powerset(attributes, max_len_attribute_set, max_len_attribute_set))
    filtered_attribute_sets = []
    for att_set in all_attributes_subsets:
        good = True
        las = 0
        lp = -1
        while good and las < len(att_set):
            atc = att_set[las]
            clp = attributes.index(atc)
            good = clp > lp
            las += 1
        if good:
            filtered_attribute_sets.append(att_set)
    # filtered_attribute_sets = random.sample(filtered_attribute_sets, 10)
    perturbations = []
    perturbed_attributes = []
    droppedValues = []
    copiedValues = []

    for subset in filtered_attribute_sets:  # iterate over the attribute/token power set
        if not all(elem.split('__')[0] in free.index.to_list() for elem in subset):
            continue

        repls = []  # list of replacement attribute_token items
        aa = []  # list of affected attributes
        replacements = dict()
        for tbc in subset:  # iterate over the attribute:token items
            affected_attribute = tbc.split('__')[0]  # attribute to be affected
            aa.append(affected_attribute)
            if affected_attribute in support.index:  # collect all possible tokens in the affected attribute to be used as replacements from the support record
                replacement_value = support[affected_attribute]
                replacement_tokens = list(set(str(replacement_value).split(' ')) - set(tbc.split('__')[1].split(' ')))
                replacements[affected_attribute] = replacement_tokens
                for rt in replacement_tokens:  # create attribute_token items for each replacement token
                    new_repl = '__'.join([affected_attribute, rt])
                    if not new_repl in repls:
                        repls.append(new_repl)

        all_rt_combs = list(_powerset(repls, max_len_attribute_set, max_len_attribute_set))
        filtered_combs = []
        for comb in all_rt_combs:
            # aff_att = None
            naas = []
            for rt in comb:
                aspl = rt.split('__')[0]
                # if aff_att is not None and aff_att != aspl:
                #    continue
                if aspl not in support.index:
                    continue
                naas.append(aspl)
                aff_att = aspl
            if aa == naas:
                filtered_combs.append(comb)

        # filtered_combs = random.sample(filtered_combs, min(max_combs, len(filtered_combs)))
        for comb in filtered_combs:
            newRecord = free.copy()
            dv = []
            cv = []
            affected_attributes = []
            ic = 0
            for tbc in subset:  # iterate over the attribute_token items
                affected_attribute = tbc.split('__')[0]  # attribute to be affected
                affected_token = tbc.split('__')[1]  # token to be replaced
                if affected_attribute in support.index and affected_attribute in replacements \
                        and len(replacements[affected_attribute]) > 0:
                    replacement_token = comb[ic].split('__')[1]
                    newRecord[affected_attribute] = str(newRecord[affected_attribute]).replace(affected_token,
                                                                                               replacement_token)
                    dv.append(affected_token)
                    cv.append(replacement_token)
                    affected_attributes.append(tbc)
                    ic += 1
            if not all(newRecord == free) and len(dv) == max_len_attribute_set:
                good = True
                if check:
                    if predict_fn is not None:
                        conf = predict_fn(pd.DataFrame(newRecord).T)['match_score'].values[0]
                        if conf > min_t and conf < max_t:
                            good = False
                    else:
                        for c in newRecord.columns:
                            good = good and all(t in idx for t in nltk.bigrams(newRecord[c].astype(str).split(' ')))
                            if not good:
                                break
                if good:
                    droppedValues.append(dv)
                    copiedValues.append(cv)
                    perturbations.append(newRecord)
                    perturbed_attributes.append(subset)

    perturbations_df = pd.DataFrame(perturbations, index=np.arange(len(perturbations)))
    r2 = triangle[1].copy()
    r2_copy = [r2] * len(perturbations_df)
    r2_df = pd.DataFrame(r2_copy, index=np.arange(len(perturbations)))
    all_perturbations = pd.DataFrame()
    if len(perturbations_df) > 0:
        if perturbations_df.columns[0].startswith(lprefix):
            all_perturbations = pd.concat([perturbations_df, r2_df], axis=1)
        else:
            all_perturbations = pd.concat([r2_df, perturbations_df], axis=1)
        all_perturbations = all_perturbations.drop([lprefix + 'id', rprefix + 'id'], axis=1)
    all_perturbations['altered_attributes'] = perturbed_attributes
    all_perturbations['droppedValues'] = droppedValues
    all_perturbations['copiedValues'] = copiedValues
    all_perturbations['triangle'] = ' '.join(triangle_ids)

    currPerturbedAttr = all_perturbations.altered_attributes.values
    try:
        predictions = predict_fn(
            all_perturbations.drop(['altered_attributes', 'droppedValues', 'copiedValues', 'triangle'], axis=1))
        predictions = pd.concat(
            [predictions, all_perturbations[['altered_attributes', 'droppedValues', 'copiedValues', 'triangle']]],
            axis=1)

        proba = predictions[['nomatch_score', 'match_score']].values

        curr_flippedPredictions = predictions[proba[:, class_to_explain] < 0.5]

        ranking = get_attribute_ranking(proba, currPerturbedAttr, class_to_explain)

        if len(curr_flippedPredictions) == len(perturbations_df):
            logging.info(f'skipped predictions at depth >= {max_len_attribute_set}')
            all_good = True
        else:
            logging.debug(f'predicted depth {max_len_attribute_set}')

        return all_perturbations, predictions, curr_flippedPredictions, all_good, ranking
    except:
        return all_perturbations, pd.DataFrame(), pd.DataFrame(), all_good, dict()


def tokenize_record(record):
    return {k: str(v).split() for k, v in record.items()}

def detokenize_record(tokenized):
    return {k: " ".join(v) for k, v in tokenized.items()}

def find_diff_spans(free_toks, support_toks, allowed_attributes=None, max_n=3):
    """
    Identify token spans (attr, start, n) where free differs from support,
    only for allowed attributes.
    """
    allowed_attributes = allowed_attributes or list(free_toks.keys())
    spans = []
    for attr in allowed_attributes:
        if attr not in free_toks or attr not in support_toks:
            continue
        f = free_toks[attr]
        s = support_toks[attr]
        max_len = max(len(f), len(s))
        for i in range(max_len):
            for n in range(1, max_n + 1):
                if i + n > max_len:
                    continue
                f_ngram = f[i:i+n]
                s_ngram = s[i:i+n]
                if f_ngram != s_ngram:
                    spans.append((attr, i, n))
    return spans

def apply_ngram_change(state_toks, support_toks, attr, start, n):
    """
    Replace tokens [start:start+n] in one attribute with the support’s n-gram.
    Also return dropped and copied tokens for explainability.
    """
    new_state = deepcopy(state_toks)
    f = new_state[attr]
    s = support_toks[attr]

    dropped = f[start:start+n] if start < len(f) else []
    copied = s[start:start+n]

    # replace or extend
    if start < len(f):
        new_state[attr] = f[:start] + copied + f[start+n:]
    else:
        new_state[attr].extend(copied)

    return new_state, dropped, copied

def generate_one_ngram_path(free, support, max_n=3, randomize=False, allowed_attributes=None):
    """Generate one transformation path with explainable token-level perturbations."""
    free_toks = tokenize_record(free)
    support_toks = tokenize_record(support)
    spans = find_diff_spans(free_toks, support_toks, allowed_attributes, max_n=max_n)

    if randomize:
        random.shuffle(spans)
    else:
        spans.sort()

    path = []
    current = deepcopy(free_toks)

    alt_attributes = []
    for step, (attr, start, n) in enumerate(spans, 1):
        if attr not in allowed_attributes or current[attr] == support_toks[attr]:
            continue

        new_state, dropped, copied = apply_ngram_change(current, support_toks, attr, start, n)
        if new_state == current:
            continue
        for d in dropped:
            alt_attributes.append(attr+'__'+d)
        current = new_state
        rec = detokenize_record(current)
        rec.update({
            "step": step,
            "altered_attributes": tuple(alt_attributes),
            "changed_span": (start, n),
            "dropped_values": dropped,
            "copied_values": copied
        })
        path.append(rec)

        # stop only if all allowed attributes match
        if all(current[a] == support_toks[a] for a in allowed_attributes if a in current):
            break

    return path

def generate_multiple_ngram_paths(free, support, n_paths=10, max_n=3, allowed_attributes=None):
    """Generate up to n_paths different explainable n-gram-based transformation paths."""
    paths = []
    for i in range(n_paths):
        random.seed(i)
        path = generate_one_ngram_path(
            free, support, max_n=max_n, randomize=True, allowed_attributes=allowed_attributes
        )
        paths.append(path)
    return paths


def full_token_perturbations_from_triangle(triangle_ids, sources_map, attributes, max_len_attribute_set,
                                           class_to_explain, lprefix, rprefix, predict_fn, num_paths = 3):
    triangle = __get_records(sources_map, triangle_ids, lprefix, rprefix)  # get triangle values
    support = triangle[2].copy()
    free = triangle[0].copy()
    prefix = support.index[0].split('_')[0] + '_'
    if prefix+'id' in attributes:
        attributes.remove(prefix+'id')
    if prefix+'id' in free:
        free.drop(prefix+'id')
    if prefix+'id' in support:
        support.drop(prefix+'id')

    allowed_attributes = set([a.split("__")[0]for a in attributes if not a.startswith('ltable_id') and not a.startswith('rtable_id') ]) # attributes need to be attributes
    all_paths = generate_multiple_ngram_paths(free, support, allowed_attributes=allowed_attributes, max_n=1, n_paths=num_paths)
    all_perturbation_dfs = []
    for path in all_paths:
        path = [a for a in path if all(elem in attributes for elem in a['altered_attributes'])] # we allow changes on allowed attributes only
        if len(path) > 0:
            all_perturbation_dfs.append(pd.DataFrame(path))
    if len(all_perturbation_dfs) == 0:
        for path in all_paths:
            path = [a for a in path if
                    any(elem in attributes for elem in a['altered_attributes'])] # we select paths that also contain changed to allowed attributes
            if len(path) > 0:
                all_perturbation_dfs.append(pd.DataFrame(path))
    perturbations_df = pd.concat(all_perturbation_dfs, axis=0)
    if perturbations_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    r2 = triangle[1].copy() # pivot record is untouched
    r2_df = pd.DataFrame([r2] * len(perturbations_df))
    if support.axes[0][0].startswith(rprefix):
        all_perturbations = pd.concat([perturbations_df.reset_index(drop=True), r2_df.reset_index(drop=True)], axis=1)
    else:
        all_perturbations = pd.concat([r2_df.reset_index(drop=True), perturbations_df.reset_index(drop=True)], axis=1)
    all_perturbations = all_perturbations.drop([f"{lprefix}id", f"{rprefix}id"], axis=1)

    triangle_values = 'triangle'
    all_perturbations[triangle_values] = ' '.join(triangle_ids)
    all_predictions = []
    all_flipped = []
    all_ranking = []
    for cur_step in range(1, max_len_attribute_set):
        predictions, curr_flipped_predictions, all_good, ranking = predict_step(all_perturbations,
                                                                               class_to_explain,
                                                                               cur_step,
                                                                               max_len_attribute_set,
                                                                               predict_fn, triangle_values)
        all_predictions.append(predictions)
        if len(curr_flipped_predictions) > 0:
            all_flipped.append(curr_flipped_predictions)
        if len(ranking) > 0:
            all_ranking.append(ranking)
        if all_good:
            # todo: compensate rankings
            break
    if len(all_flipped) == 0:
        all_flipped = pd.DataFrame()
    else:
        all_flipped = pd.concat(all_flipped)
    rankings = collect_rankings(all_ranking)
    return all_perturbations, pd.concat(all_predictions), all_flipped, rankings


def collect_rankings(all_ranking):
    rankings = {}
    for d in all_ranking:
        for k, v in d.items():
            rankings[k] = rankings.get(k, 0) + v
    return rankings


def predict_step(all_perturbations, class_to_explain, cur_step, max_len_attribute_set, predict_fn, triangle_values):
    all_good = False
    altered_attributes = 'altered_attributes'
    dropped_values = 'dropped_values'
    copied_values = 'copied_values'
    changed_span = 'changed_span'
    step = 'step'

    try:
        to_be_predicted = all_perturbations[all_perturbations[step] == cur_step]
        if len(to_be_predicted) == 0:
            return pd.DataFrame(), pd.DataFrame(), all_good, {}

        curr_perturbed_attr = to_be_predicted.altered_attributes.values
        to_be_predicted = to_be_predicted.drop([step, altered_attributes, dropped_values, copied_values, triangle_values, changed_span], axis=1)

        predictions = predict_fn(to_be_predicted)

        predictions = pd.concat(
            [predictions, all_perturbations[[altered_attributes, dropped_values, copied_values, triangle_values]]],
            axis=1)

        proba = predictions[['nomatch_score', 'match_score']].values
        curr_flipped_predictions = predictions[proba[:, class_to_explain] < 0.5]
        ranking = get_attribute_ranking(proba, curr_perturbed_attr, class_to_explain)

        if len(curr_flipped_predictions) == len(to_be_predicted):
            logging.info(f'skipped predictions at depth >= {max_len_attribute_set}')
            all_good = True
        else:
            logging.debug(f'predicted depth {max_len_attribute_set}')

        return predictions, curr_flipped_predictions, all_good, ranking
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return pd.DataFrame(), pd.DataFrame(), all_good, {}


def multiple_token_perturbations_depth(triangle_ids, sources_map, attributes, max_len_attribute_set,
                                       class_to_explain, lprefix, rprefix, predict_fn, subsequences: bool = True):
    all_good = False
    triangle = __get_records(sources_map, triangle_ids, lprefix, rprefix)  # get triangle values
    support = triangle[2].copy()
    free = triangle[0].copy()
    prefix = support.index[0].split('_')[0]
    filtered_attributes = [a for a in attributes if a.startswith(prefix)]

    # generate power set of token-attributes
    if subsequences:
        affected_ats_lists = list(ngrams(filtered_attributes, max_len_attribute_set))
    else:
        affected_ats_lists = list(_powerset(filtered_attributes, max_len_attribute_set, max_len_attribute_set))

    perturbations = []
    perturbed_attributes = []
    droppedValues = []
    copiedValues = []

    for affected_at_list in affected_ats_lists:
        affected_attributes_counts = {}
        for affected_at in affected_at_list:
            affected_a = affected_at.split('__')[0]
            if affected_a in affected_attributes_counts:
                affected_attributes_counts[affected_a] += 1
            else:
                affected_attributes_counts[affected_a] = 1

        replacements_list = []
        for k, v in affected_attributes_counts.items():
            tokens = [f"{k}__{r}" for r in str(support[k]).split(' ')]
            if subsequences:
                replacements = list(ngrams(tokens, v))
            else:
                replacements = list(_powerset(tokens, v, v))
            replacements_list.append(replacements)

        if len(replacements_list) == 1:
            substitutions = replacements_list[0]
        else:
            flat_replacements = list(chain.from_iterable(replacements_list))
            if subsequences:
                substitutions = list(ngrams(flat_replacements, len(affected_at_list)))
            else:
                substitutions = list(combinations(flat_replacements, len(affected_at_list)))

        for subst in substitutions:
            newRecord = free.copy()
            dv, cv, affected_attributes = [], [], []
            subst_dict = {}
            for e in subst:
                if tuple == type(e):
                    e = str(e[0])
                att, tok = e.split('__')
                if att in subst_dict:
                    subst_dict[att].append(tok)
                else:
                    subst_dict[att] = [tok]

            for tbc in affected_at_list:  # iterate over the attribute_token items
                affected_attribute, affected_token = tbc.split('__')
                if affected_attribute in support.index and affected_attribute in subst_dict and subst_dict[
                    affected_attribute]:
                    replacement_token = subst_dict[affected_attribute].pop(0)
                    tokens = str(newRecord[affected_attribute]).split(" ")
                    new_record_value = ' '.join(
                        replacement_token if token == affected_token else token for token in tokens)
                    newRecord[affected_attribute] = new_record_value
                    dv.append(affected_token)
                    cv.append(replacement_token)
                    affected_attributes.append(tbc)

            if len(dv) == max_len_attribute_set:
                droppedValues.append(dv)
                copiedValues.append(cv)
                perturbations.append(newRecord)
                perturbed_attributes.append(affected_at_list)

    perturbations_df = pd.DataFrame(perturbations)
    if perturbations_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), all_good, {}

    r2 = triangle[1].copy() # pivot record is untouched
    r2_df = pd.DataFrame([r2] * len(perturbations_df))
    if support.axes[0][0].startswith(rprefix):
        all_perturbations = pd.concat([perturbations_df.reset_index(drop=True), r2_df.reset_index(drop=True)], axis=1)
    else:
        all_perturbations = pd.concat([r2_df.reset_index(drop=True), perturbations_df.reset_index(drop=True)], axis=1)
    all_perturbations = all_perturbations.drop([f"{lprefix}id", f"{rprefix}id"], axis=1)

    altered_attributes = 'altered_attributes'
    dropped_values = 'dropped_values'
    copied_values = 'copied_values'
    triangle_values = 'triangle'

    all_perturbations[('%s' % altered_attributes)] = perturbed_attributes
    all_perturbations[('%s' % dropped_values)] = droppedValues
    all_perturbations[copied_values] = copiedValues
    all_perturbations[triangle_values] = ' '.join(triangle_ids)

    if len(all_perturbations) > 100:
        logging.info(f'perturbations_df size: {len(all_perturbations)}')

        # Calculate similarity scores
        similarities = []
        for _, row in all_perturbations.iterrows():
            text1 = ' '.join(str(v) for v in triangle[0].values)
            text2 = ' '.join(str(v) for v in triangle[1].values)
            row_text = ' '.join(str(v) for v in row.values)
            sim1 = nltk.jaccard_distance(set(text1.split()), set(row_text.split()))
            sim2 = nltk.jaccard_distance(set(text2.split()), set(row_text.split()))
            similarities.append((sim1 + sim2) / 2)

        # Sort by similarity and keep top 100
        all_perturbations['similarity'] = similarities
        pert_len = max(10, min(100, int(len(all_perturbations) / 10)))
        if class_to_explain:
            all_perturbations = all_perturbations.nsmallest(pert_len,'similarity')
        else:
            all_perturbations = all_perturbations.nlargest(pert_len, 'similarity')
        all_perturbations = all_perturbations.drop('similarity', axis=1)
        logging.info(f'trimmed perturbations_df size: {len(all_perturbations)}')

    currPerturbedAttr = all_perturbations.altered_attributes.values
    try:
        predictions = predict_fn(
            all_perturbations.drop([altered_attributes, dropped_values, copied_values, triangle_values], axis=1))
        predictions = pd.concat(
            [predictions, all_perturbations[[altered_attributes, dropped_values, copied_values, triangle_values]]],
            axis=1)

        proba = predictions[['nomatch_score', 'match_score']].values
        curr_flippedPredictions = predictions[proba[:, class_to_explain] < 0.5]
        ranking = get_attribute_ranking(proba, currPerturbedAttr, class_to_explain)

        if len(curr_flippedPredictions) == len(perturbations_df):
            logging.info(f'skipped predictions at depth >= {max_len_attribute_set}')
            all_good = True
        else:
            logging.debug(f'predicted depth {max_len_attribute_set}')

        return all_perturbations, predictions, curr_flippedPredictions, all_good, ranking
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return all_perturbations, pd.DataFrame(), pd.DataFrame(), all_good, {}


def get_row_string(fr, pr):
    for col in ['ltable_id', 'id', 'rtable_id']:
        if col in fr:
            fr = fr.drop(col)
        if col in pr:
            pr = pr.drop(col)
    row = '\t'.join([' '.join(fr.astype(str).values), ' '.join(pr.astype(str).values), '0'])
    return row


def check_properties(triangle, sourcesMap, predict_fn):
    try:
        t1 = triangle[0].split('@')
        t2 = triangle[1].split('@')
        t3 = triangle[2].split('@')
        if int(t1[0]) == 0:
            u = pd.DataFrame(sourcesMap.get(int(t1[0])).iloc[int(t1[1])]).transpose()
            v = pd.DataFrame(sourcesMap.get(int(t2[0])).iloc[int(t2[1])]).transpose()
            w1 = pd.DataFrame(sourcesMap.get(int(t3[0])).iloc[int(t3[1])]).transpose()
            u1 = u.copy()
            v1 = v.copy()
            w = w1.copy()

        else:
            u = pd.DataFrame(sourcesMap.get(int(t2[0])).iloc[int(t2[1])]).transpose()
            v = pd.DataFrame(sourcesMap.get(int(t1[0])).iloc[int(t1[1])]).transpose()
            w = pd.DataFrame(sourcesMap.get(int(t3[0])).iloc[int(t3[1])]).transpose()
            u1 = u.copy()
            v1 = v.copy()
            w1 = w.copy()

        _renameColumnsWithPrefix('ltable_', u)
        _renameColumnsWithPrefix('rtable_', u1)
        _renameColumnsWithPrefix('rtable_', v)
        _renameColumnsWithPrefix('ltable_', v1)
        _renameColumnsWithPrefix('rtable_', w)
        _renameColumnsWithPrefix('ltable_', w1)

        records = pd.concat([
            # identity
            pd.concat([u.reset_index(), u1.reset_index()], axis=1),
            pd.concat([v1.reset_index(), v.reset_index()], axis=1),
            pd.concat([w1.reset_index(), w.reset_index()], axis=1),

            # symmetry
            pd.concat([u.reset_index(), v.reset_index()], axis=1),
            pd.concat([v1.reset_index(), u1.reset_index()], axis=1),
            pd.concat([u.reset_index(), w.reset_index()], axis=1),
            pd.concat([w1.reset_index(), u1.reset_index()], axis=1),
            pd.concat([v1.reset_index(), w.reset_index()], axis=1),
            pd.concat([w1.reset_index(), v.reset_index()], axis=1),

            # transitivity
            pd.concat([u.reset_index(), v.reset_index()], axis=1),
            pd.concat([v1.reset_index(), w.reset_index()], axis=1),
            pd.concat([u.reset_index(), w.reset_index()], axis=1)
        ])

        predictions = np.argmax(predict_fn(records)[['nomatch_score', 'match_score']].values, axis=1)

        identity = predictions[0] == 1 and predictions[1] == 1 and predictions[2] == 1

        symmetry = predictions[3] == predictions[4] and predictions[5] == predictions[6] \
                   and predictions[7] == predictions[8]

        p1 = predictions[9]
        p2 = predictions[10]
        p3 = predictions[11]

        matches = 0
        non_matches = 0
        if p1 == 1:
            matches += 1
        else:
            non_matches += 1
        if p2 == 1:
            matches += 1
        else:
            non_matches += 1
        if p3 == 1:
            matches += 1
        else:
            non_matches += 1
        transitivity = matches == 3 or non_matches == 3 or (matches == 1 and non_matches == 2)

        return identity, symmetry, transitivity
    except:
        return False, False, False

def token_triangle_process(all_triangles, attributes, class_to_explain, predict_fn, sourcesMap, lprefix,
                           rprefix, num_threads=-1, max_combinations=-1, num_paths:int = 3):
    results = Parallel(n_jobs=num_threads, prefer='threads')(
        delayed(full_token_perturbations_from_triangle)(
            triangle, sourcesMap, attributes, max_combinations, class_to_explain, lprefix, rprefix, predict_fn, num_paths
        ) for triangle in tqdm(all_triangles)
    )

    pert_df, pred_df, cfp_df, ranking = zip(*results)

    perturbations_df = pd.concat(pert_df, ignore_index=True)
    predictions = pd.concat(pred_df, ignore_index=True)
    curr_flipped_predictions = pd.concat(cfp_df, ignore_index=True)
    rankings = collect_rankings(ranking)

    return perturbations_df, predictions, curr_flipped_predictions, rankings


def lattice_stratified_process(depth, all_triangles, attributes, class_to_explain, predict_fn, sourcesMap, lprefix,
                               rprefix, num_threads=-1):
    results = Parallel(n_jobs=num_threads, prefer='threads')(
        delayed(multiple_token_perturbations_depth)(
            triangle, sourcesMap, attributes, depth, class_to_explain, lprefix, rprefix, predict_fn
        ) for triangle in tqdm(all_triangles)
    )

    pert_df, pred_df, cfp_df, all_good, ranking = zip(*results)

    perturbations_df = pd.concat(pert_df, ignore_index=True)
    predictions = pd.concat(pred_df, ignore_index=True)
    curr_flippedPredictions = pd.concat(cfp_df, ignore_index=True)
    all_good = all(all_good)
    ranking = functools.reduce(lambda d1, d2: {**d1, **d2}, ranking)

    return perturbations_df, predictions, curr_flippedPredictions, all_good, ranking


def perturb_predict_token(pair: pd.DataFrame, all_triangles: list, tokenlevel_attributes: list, class_to_explain: int,
                          predict_fn, sources_map: dict, lprefix: str, rprefix: str, summarizer,
                          tf_idf_filter: bool = False, num_threads: int = -1, early_stop: bool = True):
    fr = sources_map[0][sources_map[0].ltable_id == int(pair.ltable_id)].iloc[0]
    pr = sources_map[1][sources_map[1].rtable_id == int(pair.rtable_id)].iloc[0]

    row_text = get_row_string(fr, pr)

    if tf_idf_filter and summarizer is not None:
        transformed_row_text = summarizer.transform(row_text.lower(), max_len=SML)
        tokenlevel_attributes = [
            ca for ca in tokenlevel_attributes if ca.split('__')[1].lower() in transformed_row_text
        ]
    else:
        transformed_row_text = row_text

    token_combinations = min(len(tokenlevel_attributes), len(transformed_row_text.split(' ')) if tf_idf_filter else len(row_text.split(' ')))

    all_predictions = pd.DataFrame()
    rankings = []
    flipped_predictions = []
    all_good = False
    flipped = False

    tot_flipped = 0
    for a in range(1, token_combinations):
        if early_stop and a > 3 and flipped or all_good:
            print(f'{tot_flipped} flipped predictions found @depth={a}')
            break
        print(f'depth-{a}')

        pert_df, pred_df, cfp_df, all_good, ranking = lattice_stratified_process(
            a, all_triangles, tokenlevel_attributes, class_to_explain, predict_fn, sources_map, lprefix, rprefix,
            num_threads=num_threads
        )
        new_flipped_size = len(cfp_df)
        print(f'{new_flipped_size} new flipped predictions found!')
        tot_flipped += new_flipped_size
        if tot_flipped > 0:
            flipped = True

        flipped_predictions.append(cfp_df)
        all_predictions = pd.concat([pred_df, all_predictions], ignore_index=True)
        rankings.append(ranking)

        if new_flipped_size == len(pert_df):
            logging.info(f'skipped predictions at depth >= {a}')
            all_good = True
        else:
            logging.debug(f'predicted depth {a}')

    flipped_predictions_df = pd.concat(flipped_predictions, ignore_index=True) if flipped_predictions else pd.DataFrame()

    return flipped_predictions_df, rankings, all_predictions


def triangle_parallel_token_explain(pair: pd.DataFrame, all_triangles: list, tokenlevel_attributes: list, class_to_explain: int,
                                    predict_fn, sources_map: dict, lprefix: str, rprefix: str, num_paths:int = 3, num_threads: int = -1):
    fr = sources_map[0][sources_map[0].ltable_id == int(pair.ltable_id)].iloc[0]
    pr = sources_map[1][sources_map[1].rtable_id == int(pair.rtable_id)].iloc[0]

    row_text = get_row_string(fr, pr)

    token_combinations = min(len(tokenlevel_attributes), len(row_text.split(' ')))

    all_predictions = pd.DataFrame()
    rankings = []
    flipped_predictions = []

    pert_df, pred_df, cfp_df, ranking = token_triangle_process(
        all_triangles, tokenlevel_attributes, class_to_explain, predict_fn, sources_map, lprefix, rprefix,
        num_threads=num_threads, max_combinations=token_combinations, num_paths=num_paths
    )

    flipped_predictions.append(cfp_df)
    all_predictions = pd.concat([pred_df, all_predictions], ignore_index=True)
    rankings.append(ranking)

    flipped_predictions_df = pd.concat(flipped_predictions, ignore_index=True) if flipped_predictions else pd.DataFrame()

    return flipped_predictions_df, rankings, all_predictions



def process_triangle(triangle: tuple, attributes: list, class_to_explain: int, predict_fn, sourcesMap: dict,
                     lprefix: str, rprefix: str):
    # take the original triangle
    max_len = 4  # len(attributes)
    all_subsets = list(_powerset(attributes, 1, max_len))
    token_rankings = []
    token_flippedPredictions = []
    predictions_list = []
    for attr_length in range(1, len(all_subsets)):
        currentTokenPerturbations = token_perturbations_from_triangle(triangle, sourcesMap, attributes, max_len,
                                                                      class_to_explain, lprefix, rprefix)
        # currentTokenPerturbations = createPerturbationsFromTriangle(triangle, sourcesMap, attributes, attr_length, class_to_explain, lprefix, rprefix)
        currPerturbedAttr = currentTokenPerturbations[['altered_attributes', 'altered_tokens']].apply(
            lambda x: ':'.join(x.dropna().astype(str)), axis=1).values
        predictions = predict_fn(currentTokenPerturbations)
        predictions = predictions.drop(columns=['altered_attributes', 'altered_tokens'])
        proba = predictions[['nomatch_score', 'match_score']].values
        curr_flippedPredictions = currentTokenPerturbations[proba[:, class_to_explain] < 0.5]
        token_flippedPredictions.append(curr_flippedPredictions)
        token_ranking = get_attribute_ranking(proba, currPerturbedAttr, class_to_explain)
        token_rankings.append(token_ranking)
        predictions_list.append(predictions)
    return pd.DataFrame(predictions_list), token_flippedPredictions, token_rankings


def explain_samples(dataset: pd.DataFrame, sources: list, predict_fn: callable, lprefix, rprefix,
                    class_to_explain: int, attr_length: int, check: bool = False,
                    discard_bad: bool = False, return_top: bool = False, persist_predictions: bool = False,
                    token: bool = False, two_step_token: bool = False, use_nec: bool = True,
                    filter_features: list = None):
    _renameColumnsWithPrefix(lprefix, sources[0]) # <--- fix this!!
    _renameColumnsWithPrefix(rprefix, sources[1]) # <--- fix this!!

    # all_triangles, sourcesMap = getMixedTriangles(dataset, sources)
    all_triangles, sourcesMap = get_triangles(dataset, sources)
    pair = dataset.iloc[[0]]
    if two_step_token:
        attributes = [col for col in list(sources[0]) if col not in [lprefix + 'id']]
        attributes += [col for col in list(sources[1]) if col not in [rprefix + 'id']]
        if filter_features is not None:
            attributes = list(set(attributes).intersection(set(filter_features)))
            attr_length = len(attributes)

        if len(all_triangles) > 0:
            attribute_ps, _, attribute_pn = attribute_level_expl(all_triangles, attr_length, attributes,
                                                                 check, class_to_explain, dataset,
                                                                 discard_bad, lprefix, persist_predictions,
                                                                 predict_fn, rprefix, sourcesMap)
            if use_nec:
                top_k_attr = 2
                sorted_pns = sorted(attribute_pn.items(), key=lambda kv: kv[1], reverse=True)
                topl = []
                topr = []
                spidx = 0
                while len(topl) < top_k_attr or len(topr) < top_k_attr:
                    c_attr = sorted_pns[spidx][0]
                    if c_attr.startswith(lprefix) and len(topl) < top_k_attr:
                        topl.append(c_attr)
                    if c_attr.startswith(rprefix) and len(topr) < top_k_attr:
                        topr.append(c_attr)
                    spidx += 1
                combs = topl + topr
            else:
                series = cf_summary(attribute_ps)
                combs = []
                for sc in series.index:
                    for scc in sc.split('/'):
                        combs.append(scc)
                combs = set(combs)

            record = pd.DataFrame(dataset.iloc[0]).T

            attributes = []
            for column in record.columns:
                if column not in ['label', 'id', lprefix + 'id', rprefix + 'id'] and column in combs:
                    tokens = str(record[column].values[0]).split(' ')
                    for t in tokens:
                        attributes.append(column + '__' + t)
            attr_length = len(attributes)

            if len(all_triangles) > 0:
                saliency, filtered_exp, flipped_predictions, all_triangles = token_level_expl(pair, all_triangles,
                                                                                              attr_length,
                                                                                              attributes,
                                                                                              class_to_explain, lprefix,
                                                                                              persist_predictions,
                                                                                              predict_fn, return_top,
                                                                                              rprefix, sourcesMap)
                return saliency, filtered_exp, flipped_predictions, all_triangles
            else:
                logging.warning(f'empty triangles !?')
                return dict(), pd.DataFrame(), pd.DataFrame(), []

    if token:
        record = pd.DataFrame(dataset.iloc[0]).T
        # we need to map records from series of attributes into series of tokens, attribute names are mapped to "original" token names
        attributes = []
        for column in record.columns:
            if column not in ['label', 'id', lprefix + 'id', rprefix + 'id']:
                tokens = str(record[column].values[0]).split(' ')
                for t in tokens:
                    attributes.append(column + '__' + t)
        if filter_features is not None:
            attributes_new = []
            for f in filter_features:
                if f in attributes:
                    attributes_new.append(f)
            attributes = attributes_new
        attr_length = len(attributes)

        if len(all_triangles) > 0 and attr_length > 0:
            return token_level_expl(pair, all_triangles, attr_length, attributes, class_to_explain, lprefix,
                                    persist_predictions, predict_fn, return_top, rprefix, sourcesMap)
        else:
            logging.warning(f'empty triangles !?')
            return dict(), pd.DataFrame(), pd.DataFrame(), []


    else:
        attributes = [col for col in list(sources[0]) if col not in [lprefix + 'id']]
        attributes += [col for col in list(sources[1]) if col not in [rprefix + 'id']]
        if filter_features is not None:
            attributes_new = []
            for f in filter_features:
                if f in attributes:
                    attributes_new.append(f)
                elif 'ltable_'+f in attributes:
                    attributes_new.append('ltable_'+f)
                elif 'rtable_'+f in attributes:
                    attributes_new.append('rtable_'+f)
            attributes = attributes_new

        if len(all_triangles) > 0:
            explanation, flipped_predictions, saliency = attribute_level_expl(all_triangles, attr_length, attributes,
                                                                              check, class_to_explain, dataset,
                                                                              discard_bad, lprefix, persist_predictions,
                                                                              predict_fn, rprefix, sourcesMap)

            if len(explanation) > 0:
                if len(flipped_predictions) > 0:
                    flipped_predictions['attr_count'] = flipped_predictions.altered_attributes.astype(str) \
                        .str.split(',').str.len()
                    flipped_predictions = flipped_predictions.sort_values(by=['attr_count'])
                if return_top:
                    series = cf_summary(explanation)
                    filtered_exp = series
                else:
                    filtered_exp = explanation

                return saliency, filtered_exp, flipped_predictions, all_triangles
            else:
                return dict(), pd.DataFrame(), pd.DataFrame(), []
        else:
            logging.warning(f'empty triangles !?')
            return dict(), pd.DataFrame(), pd.DataFrame(), []


def token_level_expl(pair, all_triangles, attr_length, attributes, class_to_explain, lprefix, persist_predictions,
                     predict_fn, return_top, rprefix, sourcesMap, num_paths=3):
    flipped_predictions, rankings, all_predictions = triangle_parallel_token_explain(pair, all_triangles, attributes,
                                                                                     class_to_explain, predict_fn, sourcesMap,
                                                                                     lprefix, rprefix, num_paths=num_paths)
    if persist_predictions:
        all_predictions.to_csv(f'predictions-{attr_length}.csv')
    explanation = aggregate_rankings(rankings, len_triangles=len(all_triangles), attr_length=attr_length)

    if 'altered_attributes' in all_predictions.columns:
        all_predictions['altered_attributes'] = all_predictions['altered_attributes'].astype(str).apply(
            lambda x: x.replace("'", '').replace('(', '').replace(',)', '').replace(', ', '/').replace(')', ''))
    else:
        print('No altered_attributes')
        print(all_predictions)
    flips = len(flipped_predictions)
    if flips > 0:
        perturb_count = all_predictions.groupby('altered_attributes').size()
        for att in explanation.index:
            if att in perturb_count:
                explanation[att] = explanation[att] / perturb_count[att]
            else:
                print(f'{att} not found in {perturb_count}')
                print(flipped_predictions)
        saliency = dict()
        for ranking in rankings:
            for k, v in ranking.items():
                for a in k:
                    if a not in attributes:
                        saliency[a] = 0
                        continue
                    if a not in saliency:
                        saliency[a] = 0
                    if flips > 0:
                        saliency[a] += v / flips

        flipped_predictions['attr_count'] = flipped_predictions.altered_attributes.astype(str) \
                .str.split(',').str.len()
        flipped_predictions = flipped_predictions.sort_values(by=['attr_count'])
        if return_top:
            series = cf_summary(explanation)
            filtered_exp = series
        else:
            filtered_exp = explanation

        return saliency, filtered_exp, flipped_predictions, all_triangles
    else:
        logging.warning(f'empty explanation !?')
        return dict(), pd.DataFrame(), pd.DataFrame(), []


def attribute_level_expl(allTriangles, attr_length, attributes, check, class_to_explain, dataset, discard_bad, lprefix,
                         persist_predictions, predict_fn, rprefix, sourcesMap):
    flipped_predictions, rankings, all_predictions = perturb_predict(allTriangles, attributes, check,
                                                                     class_to_explain, discard_bad,
                                                                     attr_length, predict_fn, sourcesMap, lprefix,
                                                                     rprefix)
    if persist_predictions:
        all_predictions.to_csv('predictions.csv')
    explanation = aggregate_rankings(rankings, len_triangles=len(allTriangles), attr_length=attr_length)
    flips = len(flipped_predictions)
    if len(attributes) == attr_length:
        flips += len(allTriangles) # account for top of the lattice structure
    saliency = dict()
    for a in dataset.columns:
        if (a.startswith(lprefix) or a.startswith(rprefix)) and not (a == lprefix + 'id' or a == rprefix + 'id'):
            saliency[a] = 0.0
    if flips != 0:
        for ranking in rankings:
            for k, v in ranking.items():
                for a in k:
                    if a not in attributes:
                        saliency[a] = 0.0
                    else:
                        saliency[a] += v / flips
    return explanation, flipped_predictions, saliency


def cf_summary(explanation):
    if len(explanation) > 0:
        sorted_attr_pairs = explanation.sort_values(ascending=False)
        explanations = sorted_attr_pairs.loc[sorted_attr_pairs.values == sorted_attr_pairs.values.max()]
        filtered = [i for i in explanations.keys() if
                    not any(all(c in i for c in b) and len(b) < len(i) for b in explanations.keys())]
        filtered_exp = {}
        for te in filtered:
            filtered_exp[te] = explanations[te]
        series = pd.Series(index=filtered_exp.keys(), data=filtered_exp.values())
    else:
        series = pd.Series()
    return series


def lattice_stratified_attribute(all_triangles, class_to_explain, sources_map, attributes, no_combinations,
                                 lprefix, rprefix, attr_length, predict_fn):
    all_predictions = pd.DataFrame()
    perturbations = []
    curr_flipped_predictions = pd.DataFrame()
    ranking = {}
    for triangle in tqdm(all_triangles):
        try:
            current_perturbations = create_perturbations_from_triangle(triangle, sources_map, attributes,
                                                                       no_combinations, lprefix, rprefix)
            current_perturbations['triangle'] = ' '.join(triangle)
            perturbations.append(current_perturbations)
        except:
            pass
    try:
        perturbations_df = pd.concat(perturbations, ignore_index=True)
    except:
        perturbations_df = pd.DataFrame(perturbations)
    if len(perturbations_df) > 0 and 'altered_attributes' in perturbations_df.columns:
        curr_perturbed_attr = perturbations_df.altered_attributes.values
        if no_combinations != attr_length:
            for c in ['ltable_id', 'rtable_id']:
                if c in perturbations_df.columns:
                    perturbations_df = perturbations_df.drop([c], axis=1)

            predictions = predict_fn(
                perturbations_df.drop(['altered_attributes', 'droppedValues', 'copiedValues', 'triangle'], axis=1))
            predictions = pd.concat(
                [predictions, perturbations_df[['altered_attributes', 'droppedValues', 'copiedValues', 'triangle']]],
                axis=1)
            all_predictions = pd.concat([all_predictions, predictions])
            proba = predictions[['nomatch_score', 'match_score']].values

            curr_flipped_predictions = predictions[proba[:, class_to_explain] < 0.5]
        else:
            proba = pd.DataFrame(columns=['nomatch_score', 'match_score'])

            if class_to_explain == 0:
                proba.loc[:, 'nomatch_score'] = np.zeros([len(perturbations_df)])
                proba.loc[:, 'match_score'] = np.ones([len(perturbations_df)])
            else:
                proba.loc[:, 'match_score'] = np.zeros([len(perturbations_df)])
                proba.loc[:, 'nomatch_score'] = np.ones([len(perturbations_df)])

            curr_flipped_predictions = pd.concat([perturbations_df.copy(), proba], axis=1)
            proba = proba.values

        ranking = get_attribute_ranking(proba, curr_perturbed_attr, class_to_explain)

    return curr_flipped_predictions, ranking, all_predictions


def perturb_predict(all_triangles, attributes, check, class_to_explain, discard_bad, attr_length, predict_fn,
                    sources_map, lprefix, rprefix, method="parallel", num_threads=-1):
    if method == "parallel":
        flipped_predictions_df, rankings, all_predictions = zip(
            *Parallel(n_jobs=num_threads, prefer='threads')(
                delayed(lattice_stratified_attribute)(all_triangles, class_to_explain, sources_map, attributes,
                                                      no_combinations, lprefix, rprefix, attr_length, predict_fn)
                for no_combinations in tqdm(range(1, attr_length))))

        all_predictions = pd.concat(list(all_predictions))
        try:
            flipped_predictions_df = pd.concat(list(flipped_predictions_df))
        except:
            flipped_predictions_df = pd.DataFrame()
        return flipped_predictions_df, rankings, all_predictions
    elif method == "monotonicity":
        all_predictions = pd.DataFrame()
        rankings = []
        transitivity = True
        flipped_predictions = []
        # lattice stratified predictions
        all_good = False
        for a in range(1, attr_length):
            t_i = 0
            perturbations = []
            for triangle in tqdm(all_triangles):
                try:
                    if check:
                        identity, symmetry, transitivity = check_properties(triangle, sources_map, predict_fn)
                        all_triangles[t_i] = all_triangles[t_i] + (identity, symmetry, transitivity,)
                    if check and discard_bad and not transitivity:
                        continue
                    current_perturbations = create_perturbations_from_triangle(triangle, sources_map, attributes, a,
                                                                               lprefix, rprefix)
                    current_perturbations['triangle'] = ' '.join(triangle)
                    perturbations.append(current_perturbations)
                except:
                    all_triangles[t_i] = all_triangles[t_i] + (False, False, False,)
                    pass
                t_i += 1

            try:
                perturbations_df = pd.concat(perturbations, ignore_index=True)
            except:
                perturbations_df = pd.DataFrame(perturbations)
            if len(perturbations_df) == 0 or 'altered_attributes' not in perturbations_df.columns:
                continue
            curr_perturbed_attr = perturbations_df.altered_attributes.values
            if a != attr_length and not all_good:
                predictions = predict_fn(
                    perturbations_df.drop(['altered_attributes', 'droppedValues', 'copiedValues', 'triangle'], axis=1))
                predictions = pd.concat(
                    [predictions, perturbations_df[['altered_attributes', 'droppedValues', 'copiedValues', 'triangle']]],
                    axis=1)
                all_predictions = pd.concat([all_predictions, predictions])
                proba = predictions[['nomatch_score', 'match_score']].values

                curr_flipped_predictions = predictions[proba[:, class_to_explain] < 0.5]
            else:
                proba = pd.DataFrame(columns=['nomatch_score', 'match_score'])

                if class_to_explain == 0:
                    proba.loc[:, 'nomatch_score'] = np.zeros([len(perturbations_df)])
                    proba.loc[:, 'match_score'] = np.ones([len(perturbations_df)])
                else:
                    proba.loc[:, 'match_score'] = np.zeros([len(perturbations_df)])
                    proba.loc[:, 'nomatch_score'] = np.ones([len(perturbations_df)])

                curr_flipped_predictions = pd.concat([perturbations_df.copy(), proba], axis=1)
                proba = proba.values

            flipped_predictions.append(curr_flipped_predictions)
            ranking = get_attribute_ranking(proba, curr_perturbed_attr, class_to_explain)
            rankings.append(ranking)

            if len(curr_flipped_predictions) == len(perturbations_df):
                logging.info(f'skipped predictions at depth {a}')
                all_good = True
            else:
                logging.debug(f'predicted depth {a}')
        try:
            flipped_predictions_df = pd.concat(flipped_predictions, ignore_index=True)
        except:
            flipped_predictions_df = pd.DataFrame(flipped_predictions)
        return flipped_predictions_df, rankings, all_predictions
    else:
        rankings = []
        transitivity = True
        flipped_predictions = []
        t_i = 0
        perturbations = []
        for triangle in tqdm(all_triangles):
            try:
                if check:
                    identity, symmetry, transitivity = check_properties(triangle, sources_map, predict_fn)
                    all_triangles[t_i] = all_triangles[t_i] + (identity, symmetry, transitivity,)
                if check and discard_bad and not transitivity:
                    continue
                current_perturbations = create_perturbations_from_triangle(triangle, sources_map, attributes,
                                                                           attr_length, lprefix, rprefix)
                perturbations.append(current_perturbations)
            except:
                all_triangles[t_i] = all_triangles[t_i] + (False, False, False,)
                pass
            t_i += 1
        try:
            perturbations_df = pd.concat(perturbations, ignore_index=True)
        except:
            perturbations_df = pd.DataFrame(perturbations)
        curr_perturbed_attr = perturbations_df.altered_attributes.values
        predictions = predict_fn(perturbations_df)
        predictions = predictions.drop(columns=['altered_attributes'])
        proba = predictions[['nomatch_score', 'match_score']].values
        curr_flipped_predictions = perturbations_df[proba[:, class_to_explain] < 0.5]
        flipped_predictions.append(curr_flipped_predictions)
        ranking = get_attribute_ranking(proba, curr_perturbed_attr, class_to_explain)
        rankings.append(ranking)
        try:
            flipped_predictions_df = pd.concat(flipped_predictions, ignore_index=True)
        except:
            flipped_predictions_df = pd.DataFrame(flipped_predictions)
        return flipped_predictions_df, rankings, predictions


# for each prediction, if the original class is flipped, set the rank of the altered attributes to 1
def get_attribute_ranking(proba: np.ndarray, altered_attributes: list, original_class: int):
    attribute_ranking = {k: 0 for k in altered_attributes}
    for i, prob in enumerate(proba):
        if float(prob[original_class]) < 0.5:
            attribute_ranking[altered_attributes[i]] += 1
    return attribute_ranking


# MaxLenAttributeSet is the max len of perturbed attributes we want to consider
# for each ranking, sum  the rank of each altered attribute
# then normalize the aggregated rank wrt the no. of triangles
def aggregate_rankings(ranking_l: list, len_triangles: int, attr_length: int):
    aggregateRanking = defaultdict(int)
    for ranking in ranking_l:
        for altered_attr in ranking.keys():
            if attr_length == -1 or len(altered_attr) <= attr_length:
                aggregateRanking[altered_attr] += ranking[altered_attr]
    aggregateRanking_normalized = {k: (v / len_triangles) for (k, v) in aggregateRanking.items()}

    alteredAttr = list(map(lambda t: "/".join(t), aggregateRanking_normalized.keys()))
    return pd.Series(data=list(aggregateRanking_normalized.values()), index=alteredAttr)

def aggregate_rankings_simple(ranking_l: list, len_triangles: int):
    aggregateRanking = defaultdict(int)
    for ranking in ranking_l:
        for altered_attr in ranking.keys():
            aggregateRanking[altered_attr] += ranking[altered_attr]
    aggregateRanking_normalized = {k: (v / len_triangles) for (k, v) in aggregateRanking.items()}

    return pd.Series(aggregateRanking_normalized)