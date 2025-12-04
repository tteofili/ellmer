import logging
import math
import re
from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from ellmer.post_hoc.utils import diff, get_row
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def get_original_prediction(r1, r2, predict_fn):
    r1r2 = get_row(r1, r2)
    return predict_fn(r1r2)[['nomatch_score', 'match_score']].values[0]


def support_predictions(r1: pd.Series, r2: pd.Series, lsource: pd.DataFrame,
                        rsource: pd.DataFrame, predict_fn, lprefix, rprefix, num_triangles: int = 100,
                        class_to_explain: int = None, max_predict: int = -1,
                        use_w: bool = True, use_q: bool = True, use_all: bool = False, llm=None):
    '''
    generate a pd.DataFrame of support predictions to be used to generate open triangles.
    :param r1: the "left" record
    :param r2: the "right" record
    :param lsource: the "left" data source
    :param rsource: the "right" data source
    :param predict_fn: the ER model prediction function
    :param lprefix: the prefix of attributes from the "left" table
    :param rprefix: the prefix of attributes from the "right" table
    :param num_triangles: number of open triangles to be used to generate the explanation
    :param class_to_explain: the class to be explained
    :param max_predict: the maximum number of predictions to be performed by the ER model to generate the requested
        number of open triangles
    :param use_w: whether to use left open triangles
    :param use_q: whether to use right open triangles
    :param use_all: whether to use all possible records in the existing data sources to create support records, not
        stopping when _num_triangles_ records have been found
    :param llm: the llm to explain, used to eventually generate synthetic records
    :return: a pd.DataFrame of record pairs with one record from the original prediction and one record yielding an
        opposite prediction by the ER model
    '''
    r1r2 = get_row(r1, r2)
    original_prediction = predict_fn(r1r2)[['nomatch_score', 'match_score']].values[0]

    r1r2['id'] = "0@" + str(r1r2[lprefix + 'id'].values[0]) + "#" + "1@" + str(r1r2[rprefix + 'id'].values[0])

    find_positives, support = get_support(class_to_explain, lsource, max_predict,
                                          original_prediction, predict_fn, r1, r2, rsource,
                                          use_w, use_q, lprefix, rprefix, num_triangles, use_all=use_all, llm=llm)

    copies_left = pd.DataFrame()
    copies_right = pd.DataFrame()
    if len(support) < num_triangles:
        try:
            copies, copies_left, copies_right = expand_copies(lprefix, lsource, r1, r2, rprefix, rsource)
            find_positives2, support2 = get_support(class_to_explain, copies_right, max_predict,
                                                    original_prediction, predict_fn, r1, r2, copies_left,
                                                    use_w, use_q, lprefix, rprefix, num_triangles, use_all=use_all)
            if len(support2) > 0:
                support = pd.concat([support, support2])
        except:
            pass

    if len(support) > 0:
        if len(support) > num_triangles:
            support = pd.concat([support[:int(num_triangles / 2)], support[-int(num_triangles / 2):]], axis=0)
        else:
            logging.warning(f'could find {str(len(support))} triangles of the {str(num_triangles)} requested')

        support['label'] = list(map(lambda predictions: int(round(predictions)),
                                    support.match_score.values))
        support = support.drop(['match_score', 'nomatch_score'], axis=1)
        if class_to_explain == None:
            r1r2['label'] = np.argmax(original_prediction)
        else:
            r1r2['label'] = class_to_explain
        support_pairs = pd.concat([r1r2, support], ignore_index=True)
        return support_pairs, copies_left, copies_right
    else:
        logging.warning('no triangles found')
        return pd.DataFrame(), copies_left, copies_right

WORD = re.compile(r'\w+')


def find_similarities(test_df: pd.DataFrame, strict: bool):
    lprefix = 'ltable_'
    rprefix = 'rtable_'
    ignore_columns = ['id']

    l_columns = [col for col in list(test_df) if (col.startswith(lprefix)) and (col not in ignore_columns)]
    r_columns = [col for col in list(test_df) if col.startswith(rprefix) and (col not in ignore_columns)]

    l_string_test_df = test_df[l_columns].astype('str').agg(' '.join, axis=1)
    r_string_test_df = test_df[r_columns].astype('str').agg(' '.join, axis=1)
    label_df = test_df['label']

    # vectorized similarity calculation
    vectorizer = CountVectorizer(token_pattern=r'\w+')
    # fit on all text to ensure vocabulary consistency
    all_text = pd.concat([l_string_test_df, r_string_test_df])
    vectorizer.fit(all_text)

    l_vectors = vectorizer.transform(l_string_test_df)
    r_vectors = vectorizer.transform(r_string_test_df)

    # normalize vectors to unit length for cosine similarity
    l_vectors = normalize(l_vectors)
    r_vectors = normalize(r_vectors)

    # row-wise dot product
    sim_scores = np.asarray(l_vectors.multiply(r_vectors).sum(axis=1)).flatten()
    sim_df = pd.Series(sim_scores)

    merged_string = pd.concat([l_string_test_df, r_string_test_df, label_df], ignore_index=True, axis=1)

    tuples_ls_df = pd.concat([merged_string, sim_df], ignore_index=True, axis=1)

    lpos_df = tuples_ls_df[tuples_ls_df[2] == 1]
    lneg_df = tuples_ls_df[tuples_ls_df[2] == 0]

    theta_mean_std_max_strict = lpos_df[3].mean()
    theta_mean_std_min_strict = lneg_df[3].mean()

    if strict:
        theta_mean_std_max_strict = theta_mean_std_max_strict + lpos_df[3].std()
        theta_mean_std_min_strict = theta_mean_std_min_strict - lneg_df[3].std()

    return theta_mean_std_min_strict, theta_mean_std_max_strict

# calculate similarity between two text vectors
def get_cosine(text1, text2):
    vec1 = text_to_vector(text1)
    vec2 = text_to_vector(text2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


def find_candidates(record, source, min_similarity, find_positives):
    record2text = " ".join([val for k, val in record.to_dict().items() if k not in ['id']])
    source_without_id = source.copy()
    source_without_id = source_without_id.drop(['id'], axis=1)
    source_ids = source.id.values

    # vectorized candidate finding
    source_docs = source_without_id.astype(str).agg(' '.join, axis=1)
    vectorizer = CountVectorizer(token_pattern=r'\w+')
    all_docs = [record2text] + source_docs.tolist()
    tfidf_matrix = vectorizer.fit_transform(all_docs)

    cosine_sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    candidates = []
    for idx, score in enumerate(cosine_sims):
        if find_positives:
            if score >= min_similarity:
                candidates.append((record['id'], source_ids[idx]))
        else:
            if score < min_similarity:
                candidates.append((record['id'], source_ids[idx]))
    return pd.DataFrame(candidates, columns=['ltable_id', 'rtable_id'])


def find_candidates_predict(record, source, find_positives, predict_fn, num_candidates, lj=True, scored: bool = True,
                            max_predict=-1, lprefix='ltable_', rprefix='rtable_', batched: bool = True,
                            num_threads: int = -1, llm=None):
    # optimize by filtering source first before creating full pair dataframe
    source_copy = source.copy()

    if max_predict > 0:
        source_copy = source_copy.sample(frac=1)[:max_predict]

    if scored:
        record_text = record_to_text(record)
        # use vectorized cosine similarity
        source_texts = source_copy.astype(str).agg(' '.join, axis=1).tolist()
        vectorizer = CountVectorizer(token_pattern=r'\w+')
        all_docs = [record_text] + source_texts
        matrix = vectorizer.fit_transform(all_docs)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

        source_copy['__score'] = scores
        source_copy = source_copy.sort_values(by='__score', ascending=not find_positives)
        source_copy = source_copy.drop(['__score'], axis=1)

    source_copy = source_copy[:num_candidates * 100]

    if lj:
        records = pd.DataFrame([record] * len(source_copy))
        copy = source_copy.copy()
        records.columns = list(map(lambda col: lprefix + col, records.columns))
        copy.columns = list(map(lambda col: rprefix + col, copy.columns))
        records.index = copy.index
        samples = pd.concat([records, copy], axis=1)
    else:
        copy = source_copy.copy()
        records = pd.DataFrame([record] * len(source_copy))
        records.index = copy.index
        copy.columns = list(map(lambda col: lprefix + col, copy.columns))
        records.columns = list(map(lambda col: rprefix + col, records.columns))
        samples = pd.concat([copy, records], axis=1)

    result = pd.DataFrame()
    if batched:
        batch = num_candidates * 4
        max_splits = 1 + int(len(samples) / batch)
        max_iters = min(max_splits, 50)
        i = 0
        while len(result) < num_candidates and i < max_iters:
            batch_samples = []
            while i < max_splits:
                start = batch * i
                end = batch * (i + 1)
                batch_sample = samples[start:end]
                batch_samples.append(batch_sample)
                i += 1

            s_zipped = zip(Parallel(n_jobs=num_threads, prefer='threads')(
                delayed(find_counter_predict)(bs, find_positives, predict_fn)
                for bs in tqdm(batch_samples, disable=False)))
            s_list = [x[0] for x in s_zipped]
            result = pd.concat(s_list, axis=0)
            if len(result) > num_candidates:
                break
    else:
        predicted = predict_fn(samples)
        if find_positives:
            out = predicted[predicted["match_score"] > 0.5]
        else:
            out = predicted[predicted["match_score"] < 0.5]
        if len(out) > 0:
            result = pd.concat([result, out], axis=0)
    #result = augment_with_llm(find_positives, lj, llm, num_candidates, predict_fn, record, result, source, lprefix, rprefix)
    return result


def augment_with_llm(find_positives, lj, llm, num_candidates, predict_fn, record, result, source, lprefix, rprefix):
    if len(result) < num_candidates and llm is not None:
        try:
            template = f'''
            you can only return headless CSV outputs
            given the following record, generate 5 more records that refer to the same entity while using different text:
            {record}
            format them as a CSV file
            return only the CSV output
            '''
            raw_answer = llm.invoke(template)
            content = raw_answer.content
            with open('synth_df.csv', mode='w') as writer:
                writer.write(content)
            synthetic_records_df = pd.read_csv('synth_df.csv', header=None, on_bad_lines='skip')
            if lj:
                records = pd.DataFrame([record] * len(synthetic_records_df))
                copy = synthetic_records_df.copy()
                copy.columns = list(map(lambda col: rprefix + col, source.columns))
                records.columns = list(map(lambda col: lprefix + col, records.columns))
                records.index = copy.index
                samples = pd.concat([records, copy], axis=1)
            else:
                copy = synthetic_records_df.copy()
                copy.columns = list(map(lambda col: lprefix + col, source.columns))
                records = pd.DataFrame([record] * len(synthetic_records_df))
                records.index = copy.index
                records.columns = list(map(lambda col: rprefix + col, records.columns))
                samples = pd.concat([copy, records], axis=1)
            predicted = predict_fn(samples)
            if find_positives:
                out = predicted[predicted["match_score"] > 0.5]
            else:
                out = predicted[predicted["match_score"] < 0.5]
            if len(out) > 0:
                result = pd.concat([result, out], axis=0)
        except Exception as e:
            print(f'error: {e}')
            pass
    return result


def find_counter_predict(batch_sample, find_positives, predict_fn):
    try:
        predicted = predict_fn(batch_sample)
    except:
        predicted = batch_sample.copy()
        predicted['match_score'] = 0.5
        predicted['nomatch_score'] = 0.5
    if find_positives:
        out = predicted[predicted["match_score"] > 0.5]
    else:
        out = predicted[predicted["match_score"] < 0.5]
    return out


def record_to_text(record, ignored_columns=['id', 'ltable_id', 'rtable_id', 'label']):
    return " ".join([str(val) for k, val in record.to_dict().items() if k not in [ignored_columns]])


def generate_subsequences(lsource, rsource, max=-1):
    new_records_left_df = pd.DataFrame()
    for i in np.arange(len(lsource[:max])):
        r = lsource.iloc[i]
        nr_df = pd.DataFrame(generate_modified(r, start_id=len(new_records_left_df) + len(lsource)))
        if len(nr_df) > 0:
            nr_df.columns = lsource.columns
            new_records_left_df = pd.concat([new_records_left_df, nr_df])
    new_records_right_df = pd.DataFrame()
    for i in np.arange(len(rsource[:max])):
        r = rsource.iloc[i]
        nr_df = pd.DataFrame(generate_modified(r, start_id=len(new_records_right_df) + len(rsource)))
        if len(nr_df) > 0:
            nr_df.columns = rsource.columns
            new_records_right_df = pd.concat([new_records_right_df, nr_df])
    return new_records_left_df, new_records_right_df


def get_support(class_to_explain, lsource, max_predict, original_prediction, predict_fn, r1, r2,
                rsource, use_w, use_q, lprefix, rprefix, num_triangles, use_all: bool = False, llm=None):
    candidates4r1 = pd.DataFrame()
    candidates4r2 = pd.DataFrame()
    num_candidates = num_triangles
    if class_to_explain == None:
        findPositives = bool(original_prediction[0] > original_prediction[1])
    else:
        findPositives = bool(0 == int(class_to_explain))
    if use_q:
        candidates4r1 = find_candidates_predict(r1, rsource, findPositives, predict_fn, num_candidates,
                                                batched=not use_all, lj=True, max_predict=max_predict, lprefix=lprefix,
                                                rprefix=rprefix, llm=llm)
    if use_w:
        candidates4r2 = find_candidates_predict(r2, lsource, findPositives, predict_fn, num_candidates,
                                                batched=not use_all, lj=False, max_predict=max_predict, lprefix=lprefix,
                                                rprefix=rprefix, llm=llm)

    max_len = min(len(candidates4r1), len(candidates4r2))
    if max_len == 0:
        max_len = max(len(candidates4r1), len(candidates4r2))

    if len(candidates4r1) > max_len:
        candidates4r1 = candidates4r1.sample(n=max_len)
    if len(candidates4r2) > max_len:
        candidates4r2 = candidates4r2.sample(n=max_len)
    candidates = pd.concat([candidates4r1, candidates4r2]).sample(frac=1)

    neighborhood = pd.DataFrame()
    if len(candidates) > 0:
        candidates['id'] = "0@" + candidates[lprefix + 'id'].astype(str) + "#" + "1@" + candidates[
            rprefix + 'id'].astype(str)
        if findPositives:
            neighborhood = candidates[candidates.match_score >= 0.5].copy()
        else:
            neighborhood = candidates[candidates.match_score < 0.5].copy()

    return findPositives, neighborhood


def generate_modified(record, start_id: int = 0):
    new_copies = []
    t_len = len(record)
    copy = record.copy()
    for t in range(t_len):
        attr_value = str(copy.get(t))
        values = attr_value.split()
        for cut in range(1, len(values)):
            for new_val in [" ".join(values[cut:]),
                            " ".join(values[:cut])]:  # generate new values with prefix / suffix dropped
                new_copy = record.copy()
                new_copy[t] = new_val  # substitute the new value with missing prefix / suffix on the target attribute
                if start_id > 0:
                    new_copy['id'] = len(new_copies) + start_id
                new_copies.append(new_copy)
    return new_copies


WORD = re.compile(r'\w+')


def cs(text1, text2):
    vec1 = Counter(WORD.findall(text1))
    vec2 = Counter(WORD.findall(text2))
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def expand_copies(lprefix, lsource, r1, r2, rprefix, rsource):
    generated_df = pd.DataFrame()
    new_copies_left = []
    new_copies_right = []
    left = True
    for record in [r1, r2]:
        r1_df = pd.DataFrame(data=[record.values], columns=record.index)
        r2_df = pd.DataFrame(data=[record.values], columns=record.index)
        r1_df.columns = list(map(lambda col: 'ltable_' + col, r1_df.columns))
        r2_df.columns = list(map(lambda col: 'rtable_' + col, r2_df.columns))
        r1r2c = pd.concat([r1_df, r2_df], axis=1)

        original = r1r2c.iloc[0].copy()
        t_len = int(len(r1r2c.columns) / 2)
        # only used for reporting
        r1r2c['diff'] = ''
        r1r2c['attr_name'] = ''
        r1r2c['attr_pos'] = ''
        copy = original.copy()
        for t in range(t_len):
            if left:
                t = t_len + t
            attr_value = str(copy.get(t))
            values = attr_value.split()
            for cut in range(1, len(values)):
                for new_val in [" ".join(values[cut:]),
                                " ".join(values[:cut])]:  # generate new values with prefix / suffix dropped
                    new_copy = original.copy()
                    new_copy[
                        t] = new_val  # substitute the new value with missing prefix / suffix on the target attribute
                    if left:
                        prefix = rprefix
                        new_id = len(new_copies_left) + len(rsource)
                        idn = 'rtable_id'
                    else:
                        prefix = lprefix
                        idn = 'ltable_id'
                        new_id = len(new_copies_right) + len(lsource)

                    new_record = pd.DataFrame(new_copy).transpose().filter(regex='^' + prefix).iloc[0]
                    new_record[idn] = new_id
                    new_copy[idn] = new_id
                    if left:
                        new_copies_left.append(new_record)
                    else:
                        new_copies_right.append(new_record)

                    # only used for reporting
                    new_copy['diff'] = diff(attr_value, new_val)
                    new_copy['attr_name'] = r1r2c.columns[t]
                    new_copy['attr_pos'] = t

                    # r1r2c = r1r2c.append(new_copy, ignore_index=True)
                    r1r2c = pd.concat([r1r2c, pd.DataFrame([new_copy])], ignore_index=True)

        if left:
            r1r2c['id'] = "0@" + r1r2c[lprefix + 'id'].astype(str) + "#" + "1@" + r1r2c[
                rprefix + 'id'].astype(str)
            left = False
        else:
            r1r2c['id'] = "0@" + r1r2c[lprefix + 'id'].astype(str) + "#" + "1@" + r1r2c[
                rprefix + 'id'].astype(str)

        generated_df = pd.concat([generated_df, r1r2c], axis=0)
    generated_records_left_df = pd.DataFrame(new_copies_left).rename(columns=lambda x: x[len(lprefix):])
    generated_records_right_df = pd.DataFrame(new_copies_right).rename(columns=lambda x: x[len(rprefix):])

    return generated_df, generated_records_left_df, generated_records_right_df


def get_neighbors(find_positives, predict_fn, r1r2c, report: bool = False):
    original = r1r2c.copy()
    try:
        r1r2c = r1r2c.drop(columns=['diff', 'attr_name', 'attr_pos'])
    except:
        pass

    unlabeled_predictions = predict_fn(r1r2c)
    if report:
        try:
            report = pd.concat([original, unlabeled_predictions['match_score']], axis=1)
            report.to_csv('experiments/diffs.csv', mode='a')
        except:
            pass
    if find_positives:
        neighborhood = unlabeled_predictions[unlabeled_predictions.match_score >= 0.5].copy()
    else:
        neighborhood = unlabeled_predictions[unlabeled_predictions.match_score < 0.5].copy()
    return neighborhood
