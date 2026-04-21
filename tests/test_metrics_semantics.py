"""
Contract tests for quantitative metrics semantics (faithfulness curve, CF validity).

These encode the intended behavior so refactors to formulas stay checked.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.metrics import auc

from ellmer.metrics import get_cf_metrics, get_faithfulness, get_validity


# --- CF validity: |match_score(CF) - original_prediction| ----------------------------


def test_validity_is_absolute_gap_to_original_class():
    """
    get_validity compares the predictor's score on the counterfactual row(s) to the
    original instance's discrete prediction (0/1). For binary labels, |p - y_orig| is
    large when the CF pushes the score toward the opposite class (flip), small when it
    matches the original class.
    """
    cf = [{"ltable_a": "x", "rtable_a": "y"}]

    assert get_validity(
        lambda _df: pd.DataFrame([{"match_score": 0.0}]), cf, original=1
    ) == pytest.approx(abs(0.0 - 1.0)), "CF with score 0 vs original 1 → full gap (flip)"

    assert get_validity(
        lambda _df: pd.DataFrame([{"match_score": 1.0}]), cf, original=1
    ) == pytest.approx(0.0)

    assert get_validity(
        lambda _df: pd.DataFrame([{"match_score": 0.0}]), cf, original=1
    ) == pytest.approx(1.0)


def test_validity_original_zero_prefers_high_match_score_on_cf():
    """Original non-match (0): a CF scored near 1 is farther from 0 than one near 0."""
    assert get_validity(
        lambda _df: pd.DataFrame([{"match_score": 1.0}]), [{}], original=0
    ) == pytest.approx(1.0)
    assert get_validity(
        lambda _df: pd.DataFrame([{"match_score": 0.0}]), [{}], original=0
    ) == pytest.approx(0.0)


def test_get_cf_metrics_validity_mean_over_examples():
    """Mean validity averages per-row get_validity (same weight per included row)."""
    test_df = pd.DataFrame(
        [
            {"ltable_a": "a0", "rtable_a": "b0", "label": 1},
            {"ltable_a": "a1", "rtable_a": "b1", "label": 0},
        ]
    )
    rows = [
        {"id": 0, "prediction": 1, "cfs": [{"ltable_a": "x", "rtable_a": "b0"}]},
        {"id": 1, "prediction": 0, "cfs": [{"ltable_a": "a1", "rtable_a": "y"}]},
    ]

    def predict_fn(df: pd.DataFrame) -> pd.DataFrame:
        # Deterministic: pretend classifier returns fixed scores per test case
        if df.iloc[0]["ltable_a"] == "x":
            return pd.DataFrame([{"match_score": 0.0}])
        return pd.DataFrame([{"match_score": 1.0}])

    out = get_cf_metrics(
        ["c"],
        predict_fn=predict_fn,
        base_dir="",
        test_set_df=test_df,
        results_by_name={"c": rows},
    )
    v0 = abs(0.0 - 1)
    v1 = abs(1.0 - 0)
    assert out["c"]["validity"] == pytest.approx((v0 + v1) / 2.0)


def test_get_validity_uses_first_predicted_match_score_only():
    """get_validity uses predict_fn(...)['match_score'].values[0] (first row of model output)."""
    calls: list[pd.DataFrame] = []

    def predict_fn(df: pd.DataFrame) -> pd.DataFrame:
        calls.append(df.copy())
        # Two CF candidates: only the first output score must affect validity
        return pd.DataFrame([{"match_score": 0.25}, {"match_score": 0.99}])

    cfs = [
        {"ltable_a": "first"},
        {"ltable_a": "second"},
    ]
    v = get_validity(predict_fn, cfs, original=1)
    assert len(calls) == 1
    assert calls[0].shape[0] == 2
    assert v == pytest.approx(abs(0.25 - 1))


# --- Faithfulness: AUC over mask fractions × eval_fn(masked dataset) -------------------


def test_faithfulness_constant_eval_fn_equals_sklearn_auc_of_flat_curve():
    """Faithfulness is auc(thresholds, [eval_fn(...)] per threshold); constant eval → flat curve."""
    rows = [
        {"id": 0, "prediction": 1, "saliency": {"ltable_a": 0.5, "rtable_a": 0.5}},
    ]
    test_df = pd.DataFrame([{"ltable_a": "a", "rtable_a": "b", "label": 1}])
    c = 0.37
    aucs = get_faithfulness(
        ["m"],
        eval_fn=lambda df: c,
        base_dir="",
        test_set_df=test_df,
        results_by_name={"m": rows},
    )
    thresholds = [0.1, 0.33, 0.5, 0.7, 0.9]
    assert aucs["m"] == pytest.approx(auc(thresholds, [c] * len(thresholds)))


def test_faithfulness_thresholds_and_attr_len_determine_top_k():
    """
    attr_len = len(columns) - 2 (reserving label + match or similar).
    top_k = max(1, int(threshold * attr_len)) attributes masked per row, by descending saliency.
    """
    # Three attribute columns (+ label + match) so int(0.9 * 3) == 2 and top_k can exceed 1.
    test_set_df = pd.DataFrame(
        [
            {
                "ltable_a": "A",
                "ltable_b": "B",
                "rtable_a": "C",
                "label": 0,
                "match": 0,
            }
        ]
    )
    results_by_name = {
        "m": [
            {
                "id": 0,
                "prediction": 0,
                "saliency": {"ltable_a": 1.0, "ltable_b": 0.0, "rtable_a": 0.0},
            }
        ]
    }
    scores: list[float] = []

    def eval_fn(masked_df: pd.DataFrame) -> float:
        # Record whether top salient column was cleared (ltable_a first)
        cleared = masked_df.iloc[0]["ltable_a"] == ""
        scores.append(1.0 if cleared else 0.0)
        return scores[-1]

    get_faithfulness(["m"], eval_fn, base_dir="", test_set_df=test_set_df, results_by_name=results_by_name)
    assert len(scores) == 5
    attr_len = 3
    expected_top_k = [max(1, int(t * attr_len)) for t in [0.1, 0.33, 0.5, 0.7, 0.9]]
    assert expected_top_k == [1, 1, 1, 2, 2]
    # With k>=1 ltable_a masked first → all 1.0
    assert all(s == 1.0 for s in scores)


def test_faithfulness_masks_highest_saliency_attributes_first():
    """At k=1, the attribute with strictly higher saliency is masked first (descending sort)."""
    test_set_df = pd.DataFrame(
        [{"ltable_a": "keep", "rtable_a": "B", "label": 1, "match": 1}]
    )
    results_by_name = {
        "m": [
            {
                "id": 0,
                "prediction": 1,
                "saliency": {"ltable_a": 0.1, "rtable_a": 0.9},
            }
        ]
    }

    def eval_fn(masked_df: pd.DataFrame) -> float:
        return 1.0 if masked_df.iloc[0]["rtable_a"] == "" else 0.0

    aucs = get_faithfulness(
        ["m"], eval_fn, base_dir="", test_set_df=test_set_df, results_by_name=results_by_name
    )
    thresholds = [0.1, 0.33, 0.5, 0.7, 0.9]
    assert aucs["m"] == pytest.approx(auc(thresholds, [1.0] * len(thresholds)))

