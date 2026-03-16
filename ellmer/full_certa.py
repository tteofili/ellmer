from collections import Counter

import math
import re

from ellmer.explainer import BaseLLMExplainer


class FullCerta(BaseLLMExplainer):

    def __init__(self, explanation_granularity, delegate, certa, num_triangles=10, max_predict=-1):
        self.explanation_granularity = explanation_granularity
        self.certa = certa
        self.num_triangles = num_triangles
        self.delegate = delegate
        self.predict_fn = lambda x: self.delegate.predict(x)
        self.max_predict = max_predict

    def predict_and_explain(self, ltuple, rtuple, verbose: bool = False):
        pae = self.delegate.predict_tuples(ltuple, rtuple)
        prediction = pae
        saliency_explanation = None
        cf_explanation = None
        if pae is not None:
            ltuple_series = self.get_row(ltuple, self.certa.lsource, prefix="ltable_")
            rtuple_series = self.get_row(rtuple, self.certa.rsource, prefix="rtable_")
            saliency_df, cf_summary, cfs, _, _, _ = self.certa.explain(ltuple_series, rtuple_series, self.predict_fn,
                                                                       token="token" == self.explanation_granularity,
                                                                       num_triangles=self.num_triangles,
                                                                       max_predict=self.max_predict, llm=self.delegate.llm)
            saliency_explanation = saliency_df.to_dict('list')
            if len(cfs) > 0:
                cf_explanation = cfs.drop(
                    ['altered_attributes', 'dropped_values', 'copied_values', 'triangle', 'attr_count'],
                    axis=1).iloc[0].T.to_dict()
            else:
                cf_explanation = {}
        return {"prediction": prediction, "saliency": saliency_explanation, "cf": cf_explanation}

    def get_row(self, row, df, prefix=''):
        # Build lookup dict: strip prefix from row keys, values as single-element lists for isin()
        rc = {k.replace(prefix, ""): [v] for k, v in row.items()}

        # Step 1: exact match on all columns (excluding id)
        if 'id' in df.columns:
            match_cols = df.drop(['id'], axis=1)
        elif prefix + 'id' in df.columns:
            match_cols = df.drop([prefix + 'id'], axis=1)
        else:
            match_cols = df
        exact = df[match_cols.isin(rc).all(axis=1)]
        if len(exact) == 1:
            return exact.iloc[0]
        if len(exact) > 1:
            return exact.iloc[0]

        # Step 2: column-by-column filter
        result = df.copy()
        for k, v in rc.items():
            if k not in result.columns:
                continue
            result_new = result[result[k] == v[0]]
            if len(result_new) == 1:
                return result_new.iloc[0]
            if len(result_new) == 0:
                continue
            result = result_new

        # Step 3: final tie-break when multiple rows remain
        if len(result) > 1:
            print(f'warning: found more than 1 item!({len(result)})')
            filtered_df = df.copy()
            for c in df.columns:
                if c not in filtered_df.columns or c not in rc:
                    continue
                new_filtered_df = filtered_df.loc[filtered_df[c].isin(rc[c])]
                if len(new_filtered_df) == 1:
                    return new_filtered_df.iloc[0]
                if len(new_filtered_df) > 0:
                    filtered_df = new_filtered_df
            if len(filtered_df) > 0:
                filtered_d_df = filtered_df.drop_duplicates()
                if len(filtered_d_df) > 1:
                    print(f'warning: found more than 1 filtered item!({len(filtered_d_df)})')
                    print(filtered_d_df)
                    print('getting first one')
                return filtered_d_df.iloc[0]

        return result.iloc[0]

    def record_to_text(self, record, ignored_columns=['id', 'ltable_id', 'rtable_id', 'label']):
        return " ".join([str(val) for k, val in record.to_dict().items() if k not in [ignored_columns]])

    def cs(self, text1, text2):
        WORD = re.compile(r'\w+')
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

    def count_predictions(self):
        return self.delegate.count_predictions()

    def count_tokens(self):
        return self.delegate.count_tokens()