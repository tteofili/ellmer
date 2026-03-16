import functools
import json
import random
import time

import numpy as np
import openai
import pandas as pd
from tqdm import tqdm


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
    return pd.concat(xcs, axis=0)


def get_tuples(xc):
    elt = dict()
    ert = dict()
    for c in xc.columns:
        if c in ['ltable_id', 'rtable_id']:
            continue
        if c.startswith('ltable_'):
            elt[str(c).replace('ltable_', '')] = xc[c].astype(str).values[0]
        if c.startswith('rtable_'):
            ert[str(c).replace('rtable_', '')] = xc[c].astype(str).values[0]
    return elt, ert


def text_to_data(answer, llm_fn):
    template = ("transform the following content into a json with entries for: the matching prediction (key = 'prediction'),"
                "the saliency explanation (key = 'saliency_explanation') as a dictionary, the counterfactual explanation "
                "(key = 'counterfactual_explanation') as a dictionary."
                "the matching prediction has to be either 1 for matching or 0 for non-matching."
                "the saliency explanation is a dictionary with features as keys and saliency scores as values (e.g., "
                "{'ltable_abc':0.1, 'ltable_cde':0.3, 'rtable_abc':0.1, 'rtable_cde':0.3})"
                "the counterfactual explanation is a dictionary with features as keys and counterfactual values as values"
                "e.g., {'ltable_abc':'foo bar', 'ltable_cde':'lorem ipsum', 'rtable_abc':'foo ban', 'rtable_cde':'lorem ipsum'})"
                "in case of features with the same name, add 'ltable_' prefix to the former and 'rtable_' prefix to the latter."
                "return only the json, here's the content: \"content\"")
    json_str_answer = "{'prediction': null, 'saliency_explanation': null, 'counterfactual_explanation': null}"
    try:
        json_str_answer = llm_fn.invoke(template.replace("content", answer))
        json_str_answer = json_str_answer.content
    except:
        try:
            json_str_answer = llm_fn(template.replace("content", answer))
        except:
            pass
    if json_str_answer.startswith('```json'):
        json_str_answer = json_str_answer.replace('```json','').replace('```','')
    answer_dict = json.loads(json_str_answer)
    if 'prediction' in answer_dict:
        prediction = answer_dict['prediction']
    elif 'matching' in answer_dict:
        prediction = answer_dict['matching']
    elif 'is_match' in answer_dict:
        prediction = answer_dict['is_match']
    else:
        prediction = text_to_match(answer, llm_fn)
    saliency = answer_dict['saliency_explanation']
    cf = answer_dict['counterfactual_explanation']
    return prediction, saliency, cf


def text_to_match(answer, llm_fn, n=0):
    summarized = 0
    idks = 0
    no_match_score = 0
    match_score = 0
    answer = answer.strip().lower()
    if answer.startswith("yes") or answer.endswith("yes") or answer == "1" or answer == 'matching':
        match_score = 1
    elif answer.startswith("no") or answer.endswith("no") or answer == "0" or answer == 'non-matching':
        no_match_score = 1
    elif n == 0:
        template = "summarize the following sentence as a 'matching' or 'non-matching': \"response\""
        try:
            summarized_answer = llm_fn(template.replace("response", answer))
        except:
            try:
                summarized_answer = llm_fn.invoke(template.replace("response", answer))
                summarized_answer = summarized_answer.content
            except:
                summarized_answer = "false"
        summarized += 1
        snms, sms = text_to_match(summarized_answer, llm_fn, n=1)
        if snms == 0 and sms == 0:
            idks += 1
            no_match_score = 1
        else:
            no_match_score = snms
            match_score = sms
    return no_match_score, match_score


@functools.lru_cache(maxsize=64)
def read_prompt(file_path: str):
    with open(file_path) as file:
        lines = [tuple(line.rstrip().split('::')) for line in file]
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

def _row_from_index(df_by_id, id_val):
    row = df_by_id.loc[id_val]
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def merge_sources(table, left_prefix, right_prefix, left_source, right_source, copy_from_table, ignore_from_table,
                  robust: bool = False, samples: int = -1):
    ignore_column = copy_from_table + ignore_from_table
    left_idx = left_source.set_index('id')
    right_idx = right_source.set_index('id')
    rows_list = []

    for i, row in tqdm(table.iterrows()):
        leftid = row[left_prefix + 'id']
        rightid = row[right_prefix + 'id']
        l_tuple = _row_from_index(left_idx, leftid).copy()
        r_tuple = _row_from_index(right_idx, rightid).copy()
        for ic in ignore_column:
            if ic in l_tuple.index:
                l_tuple = l_tuple.drop([ic])
            if ic in r_tuple.index:
                r_tuple = r_tuple.drop([ic])
        new_row = get_row(l_tuple, r_tuple, lprefix=left_prefix, rprefix=right_prefix)
        new_row['label'] = row['label']
        rows_list.append(new_row)

        if robust:
            try:
                sym_new_row = {column: row[column] for column in copy_from_table}
                for id_val, source_idx, prefix in [
                    (rightid, right_idx, left_prefix),
                    (leftid, left_idx, right_prefix),
                ]:
                    r = _row_from_index(source_idx, id_val)
                    for column in source_idx.columns:
                        if column not in ignore_column:
                            sym_new_row[prefix + column] = r[column]
                rows_list.append(pd.DataFrame([sym_new_row]))
            except Exception:
                pass

            try:
                lcopy_row = {column: row[column] for column in copy_from_table}
                for id_val, source_idx, prefix in [
                    (leftid, left_idx, left_prefix),
                    (leftid, left_idx, right_prefix),
                ]:
                    r = _row_from_index(source_idx, id_val)
                    for column in source_idx.columns:
                        if column not in ignore_column:
                            lcopy_row[prefix + column] = r[column]
                lcopy_row['label'] = 1
                rows_list.append(pd.DataFrame([lcopy_row]))
            except Exception:
                pass

            try:
                rcopy_row = {column: row[column] for column in copy_from_table}
                for id_val, source_idx, prefix in [
                    (rightid, right_idx, left_prefix),
                    (rightid, right_idx, right_prefix),
                ]:
                    r = _row_from_index(source_idx, id_val)
                    for column in source_idx.columns:
                        if column not in ignore_column:
                            rcopy_row[prefix + column] = r[column]
                rcopy_row['label'] = 1
                rows_list.append(pd.DataFrame([rcopy_row]))
            except Exception:
                pass

        if i == samples:
            break

    if not rows_list:
        return pd.DataFrame(columns={col: table[col].dtype for col in copy_from_table})
    return pd.concat(rows_list, ignore_index=True)


def get_row(r1, r2, lprefix='ltable_', rprefix='rtable_'):
    r1_df = pd.DataFrame(data=[r1.values], columns=r1.index)
    r2_df = pd.DataFrame(data=[r2.values], columns=r2.index)
    r1_df.columns = list(map(lambda col: lprefix + col, r1_df.columns))
    r2_df.columns = list(map(lambda col: rprefix + col, r2_df.columns))
    r1r2 = pd.concat([r1_df, r2_df], axis=1)
    return r1r2