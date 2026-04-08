"""Lightweight base class for LLM explainers (no torch / transformers import chain)."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

import ellmer.utils


class BaseLLMExplainer:

    pred_count = 0
    tokens = 0

    def predict_and_explain(self, ltuple, rtuple):
        prediction = self.predict_tuples(ltuple, rtuple)
        saliency, cf = self.explain(ltuple, rtuple, prediction)
        return {"prediction": prediction, "saliency": saliency, "cf": cf}

    def predict_tuples(self, ltuple, rtuple):
        return False

    def predict(self, x, mojito=False):
        xcs = []
        ranged = range(len(x))
        for idx in tqdm(ranged, disable=False):
            xc = x.iloc[[idx]].copy()
            ltuple, rtuple = ellmer.utils.get_tuples(xc)
            matching = self.predict_tuples(ltuple, rtuple)
            if matching:
                xc["nomatch_score"] = 0
                xc["match_score"] = 1
            else:
                xc["nomatch_score"] = 1
                xc["match_score"] = 0
            if mojito:
                full_df = np.dstack((xc["nomatch_score"], xc["match_score"])).squeeze()
                xc = full_df
            xcs.append(xc)
            self.pred_count += len(x)
        return pd.concat(xcs, axis=0)

    def explain(self, ltuple, rtuple, prediction):
        return None, None

    def evaluation(self, data_df):
        predictions = self.predict(data_df)
        predictions = predictions["match_score"].astype(int).values
        labels = data_df["label"].astype(int).values
        return f1_score(y_true=labels, y_pred=predictions)

    def count_predictions(self):
        return self.pred_count

    def count_tokens(self):
        return self.tokens
