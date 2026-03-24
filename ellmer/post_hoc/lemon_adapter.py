"""Build LEMON ``records_a`` / ``records_b`` and a ``predict_proba`` wrapper around ellmer's ``predict_fn``."""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import pandas as pd


def ltuple_rtuple_to_lemon_frames(
    ltuple: dict,
    rtuple: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Single pair -> LEMON inputs (index = rid, one row each).
    Dict keys are unprefixed column names (as in ``ellmer.utils.get_tuples``).
    """
    records_a = pd.DataFrame([dict(ltuple)]).convert_dtypes()
    records_b = pd.DataFrame([dict(rtuple)]).convert_dtypes()
    records_a.index = pd.Index([0], name="rid")
    records_b.index = pd.Index([0], name="rid")
    record_id_pairs = pd.DataFrame({"a.rid": [0], "b.rid": [0]})
    return records_a, records_b, record_id_pairs


def make_lemon_predict_proba(
    predict_fn: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    l_prefix: str = "ltable_",
    r_prefix: str = "rtable_",
) -> Callable:
    """
    Wrap CERTA-style ``predict_fn(df)`` returning columns ``nomatch_score``, ``match_score``.

    LEMON's ``predict_proba(records_a, records_b, record_id_pairs)`` should return a 1-D array
    of **match** probabilities (class index 1), consistent with ``lemon._lemon_utils``.
    """

    def predict_proba(records_a, records_b, record_id_pairs, show_progress=False, attr_strings=None):
        _ = show_progress
        _ = attr_strings

        rows = []
        for _, rid_row in record_id_pairs.iterrows():
            ar = int(rid_row["a.rid"])
            br = int(rid_row["b.rid"])
            a = records_a.loc[ar]
            b = records_b.loc[br]
            row = {}
            for c in a.index:
                row[l_prefix + str(c)] = a[c]
            for c in b.index:
                row[r_prefix + str(c)] = b[c]
            rows.append(row)

        batch = pd.DataFrame(rows)
        pred = predict_fn(batch)
        match = pred["match_score"].astype(float).values
        return np.asarray(match, dtype=float)

    predict_proba.__name__ = "predict_proba_ellmer"
    return predict_proba
