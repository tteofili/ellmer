from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from ellmer.llm_output_parse import parse_cf_tsv_or_json, parse_saliency_response
from ellmer.metrics import get_cf_metrics, get_faithfulness
from ellmer.selfexplainer import SelfExplainer


def test_parse_saliency_response_falls_back_to_first_json_object():
    text = "analysis...\nHere is json: {'saliency_explanation': {'ltable_a': 0.7}}\n"
    out = parse_saliency_response(text)
    assert out["ltable_a"] == 0.7


def test_parse_cf_handles_empty_record_after_gracefully():
    out = parse_cf_tsv_or_json('CF_JSON: {"record_after": {}}', {"x": "1"}, {"y": "2"})
    assert out == {}


@patch.object(SelfExplainer, "_invoke")
@patch("ellmer.selfexplainer.ellmer.utils.read_prompt")
def test_ptse_returns_dicts_and_presence_flags_even_without_saliency_cf(mock_read_prompt, mock_invoke):
    mock_read_prompt.return_value = [("system", "s")]
    mock_invoke.return_value = "yes"
    llm = MagicMock()
    se = SelfExplainer(
        model_type="delegate",
        delegate=llm,
        explanation_granularity="attribute",
        prompts={"ptse": {"er": "dummy_er_prompt.txt"}},
    )
    out = se.predict_and_explain({"a": "1"}, {"b": "2"})
    assert isinstance(out["saliency"], dict)
    assert isinstance(out["cf"], dict)
    assert out["saliency_present"] is False
    assert out["cf_present"] is False


def test_metrics_include_stats_for_missing_saliency_and_cf():
    results_by_name = {
        "m": [{"id": 0, "prediction": 1, "saliency": "bad"}],
        "c": [{"id": 0, "prediction": 1, "cfs": [{}]}],
    }
    test_df = pd.DataFrame([{"ltable_a": "x", "rtable_a": "y", "label": 1, "match": 1}])

    aucs, faith_stats = get_faithfulness(
        ["m"],
        eval_fn=lambda df: 0.0,
        base_dir="",
        test_set_df=test_df,
        results_by_name={"m": results_by_name["m"]},
        include_stats=True,
    )
    assert "m" in faith_stats
    assert faith_stats["m"]["n_missing_or_invalid_saliency"] >= 1
    assert isinstance(aucs, dict)

    cf_rows, cf_stats = get_cf_metrics(
        ["c"],
        predict_fn=lambda df: pd.DataFrame([{"match_score": 1.0}]),
        base_dir="",
        test_set_df=test_df,
        results_by_name={"c": results_by_name["c"]},
        include_stats=True,
    )
    assert "c" in cf_rows
    assert "c" in cf_stats
    assert "n_skipped" in cf_stats["c"]


def test_get_faithfulness_row_indices_subset():
    """Faithfulness on a row subset matches full evaluation when eval_fn is constant."""
    rows = [
        {"id": 0, "prediction": 1, "saliency": {"ltable_a": 0.9, "rtable_a": 0.1}},
        {"id": 1, "prediction": 0, "saliency": {"ltable_a": 0.2, "rtable_a": 0.8}},
        {"id": 2, "prediction": 1, "saliency": {"ltable_a": 0.5, "rtable_a": 0.5}},
    ]
    test_df = pd.DataFrame(
        [
            {"ltable_a": "a0", "rtable_a": "b0", "label": 1},
            {"ltable_a": "a1", "rtable_a": "b1", "label": 0},
            {"ltable_a": "a2", "rtable_a": "b2", "label": 1},
        ]
    )
    full = get_faithfulness(
        ["m"],
        eval_fn=lambda df: 0.5,
        base_dir="",
        test_set_df=test_df,
        results_by_name={"m": rows},
    )
    sub_aucs, st = get_faithfulness(
        ["m"],
        eval_fn=lambda df: 0.5,
        base_dir="",
        test_set_df=test_df,
        results_by_name={"m": rows},
        row_indices=[0, 2],
        include_stats=True,
    )
    assert abs(full["m"] - sub_aucs["m"]) < 1e-9
    assert st["m"]["n_total"] == 2


def test_get_cf_metrics_row_indices_aligns_ids():
    test_df = pd.DataFrame(
        [
            {"ltable_a": "a0", "rtable_a": "b0", "label": 1},
            {"ltable_a": "a1", "rtable_a": "b1", "label": 0},
        ]
    )
    data = [
        {"id": 0, "prediction": 1, "cfs": [[{"ltable_a": "x", "rtable_a": "b0"}]]},
        {"id": 1, "prediction": 0, "cfs": [[{"ltable_a": "a1", "rtable_a": "y"}]]},
    ]
    out = get_cf_metrics(
        ["c"],
        predict_fn=lambda df: pd.DataFrame([{"match_score": 1.0}]),
        base_dir="",
        test_set_df=test_df,
        results_by_name={"c": data},
        row_indices=[1],
    )
    assert "c" in out
    assert out["c"]["validity"] >= 0.0

