import operator
import random
from langchain.chains import LLMChain

from ellmer.full_certa import FullCerta


class HybridCerta(FullCerta):
    def __init__(self, explanation_granularity, pred_delegate, certa, ellmers, num_draws=1, num_triangles=10,
                 combine: str = 'freq', top_k: int = -1):
        self.explanation_granularity = explanation_granularity
        self.certa = certa
        self.num_triangles = num_triangles
        self.delegate = pred_delegate
        self.ellmers = ellmers
        self.predict_fn = lambda x: self.delegate.predict(x)
        self.num_draws = num_draws
        self.combine = combine
        self.top_k = top_k

    def predict_and_explain(self, ltuple, rtuple, max_predict: int = -1, verbose: bool = False):
        satisfied = False
        saliency_explanation = {}
        cf_explanation = {}
        filter_features = []
        tri = []

        if 'attribute' == self.explanation_granularity:
            no_features = len(ltuple) + len(rtuple)
        elif 'token' == self.explanation_granularity:
            no_features = len(str(ltuple).split(' ')) + len(str(rtuple).split(' '))
        else:
            raise ValueError('invalid explanation granularity')
        if self.top_k < 0:
            top_k = int(no_features * 0.25)
        else:
            top_k = self.top_k
        its = 0
        support_samples = None
        pae_dicts = []
        for e in self.ellmers:
            for _ in range(self.num_draws):
                full_answer = e.predict_and_explain(ltuple, rtuple)
                pae_dict = full_answer['saliency']
                pae_dicts.append(pae_dict)

        pae = self.delegate.predict_tuples(ltuple, rtuple)
        prediction = pae
        ltuple_series = self.get_row(ltuple, self.certa.lsource, prefix="ltable_")
        rtuple_series = self.get_row(rtuple, self.certa.rsource, prefix="rtable_")
        num_triangles = self.num_triangles

        while not satisfied:
            if self.combine == 'freq':
                # get most frequent features from the self-explanations
                filter_features = []
                for se in pae_dicts:
                    if type(se) is dict:
                        try:
                            fse = {k: v for k, v in se.items() if v > 0}
                            if len(fse) > 0:
                                sorted_attributes_dict = sorted(fse.items(), key=operator.itemgetter(1), reverse=True)[:top_k]
                                top_features = [f[0] for f in sorted_attributes_dict]
                                filter_features = filter_features + top_features
                        except:
                            pass
                fc = {}
                for f in filter_features:
                    if f in fc:
                        fc[f] = fc[f] + 1
                    else:
                        fc[f] = 1
                sorted_fc = sorted(fc.items(), key=operator.itemgetter(1), reverse=True)[:top_k]
                filter_features = [sfc[0] for sfc in sorted_fc]
            elif self.combine == 'union':
                # get all features from the self-explanations
                filter_features = []
                for se in pae_dicts:
                    sorted_attributes_dict = sorted(se.items(), key=operator.itemgetter(1), reverse=True)[:top_k]
                    top_features = [f[0] for f in sorted_attributes_dict]
                    for tf in top_features:
                        if tf not in filter_features:
                            filter_features.append(tf)
            elif self.combine == 'intersection':
                # get recurring features only from the self-explanations
                filter_features = set()
                for se in pae_dicts:
                    sorted_attributes_dict = sorted(se.items(), key=operator.itemgetter(1), reverse=True)[:top_k]
                    top_features = set([f[0] for f in sorted_attributes_dict])
                    if len(filter_features) == 0:
                        filter_features = top_features
                    filter_features = filter_features.intersection(top_features)
                filter_features = list(filter_features)
            elif self.combine == 'random':
                filter_features = set()
                for se in pae_dicts:
                    for _ in range(top_k):
                        filter_features.add(random.choice(se)[0])
                filter_features = list(filter_features)
            else:
                raise ValueError("Unknown combination method")
            if len(filter_features) > 0:
                if 'token' == self.explanation_granularity:
                    token_attributes_filtered = []
                    for ff in filter_features:
                        if ff.startswith('ltable_') and '__' in ff:
                            token_attributes_filtered.append(ff)
                        elif ff.startswith('rtable_') and '__' in ff:
                            token_attributes_filtered.append(ff)
                        else:
                            for ic in ltuple.keys():
                                if ff in ltuple[ic].split(' '):
                                    token_attributes_filtered.append(ic + '__' + ff)
                            for ic in rtuple.keys():
                                if ff in rtuple[ic].split(' '):
                                    token_attributes_filtered.append(ic + '__' + ff)
                    filter_features = token_attributes_filtered
                    if len(filter_features) == 0:
                        print(f'filtered out all features from {filter_features}')
                        break
                # regenerate support_samples, when empty
                if support_samples is not None and len(support_samples) == 0:
                    support_samples = None
                    num_triangles *= 2
                saliency_df, cf_summary, cfs, tri, _, support_samples = self.certa.explain(ltuple_series, rtuple_series,
                                                                                           self.predict_fn,
                                                                                           token="token" == self.explanation_granularity,
                                                                                           num_triangles=num_triangles,
                                                                                           max_predict=max_predict,
                                                                                           filter_features=filter_features,
                                                                                           support_samples=support_samples,
                                                                                           two_step_token=False)
                if len(saliency_df) > 0 and len(cf_summary) > 0:
                    saliency_explanation = saliency_df.to_dict('list')
                    if len(cfs) > 0:
                        cf_explanation = cfs.drop(
                            ['altered_attributes', 'dropped_values', 'copied_values', 'triangle', 'attr_count'],
                            axis=1).iloc[0].T.to_dict()
                    print(saliency_explanation)
                    aggregated_pn = 0
                    for sev in saliency_explanation.values():
                        if type(sev) == list:
                            aggregated_pn += sev[0]
                        else:
                            aggregated_pn += sev
                    if aggregated_pn >= 0.1 and max(cf_summary.to_dict().values()) > 0:
                        satisfied = True
            if 'attribute' == self.explanation_granularity:
                top_k += 1
            elif 'token' == self.explanation_granularity:
                top_k += 5
            its += 1
            if satisfied or top_k == no_features or its == 10:
                top_k -= 1
                break
            else:
                print(f"hybrid iteration={its} with top_k={top_k}")
        return {"prediction": prediction, "saliency": saliency_explanation, "cf": cf_explanation,
                "filter_features": filter_features, "self_explanations": pae_dicts, "top_k": top_k, "iterations": its,
                "triangles": len(tri)}

    def count_predictions(self):
        return self.delegate.count_predictions()

    def count_tokens(self):
        return self.delegate.count_tokens()