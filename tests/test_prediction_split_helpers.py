"""Tests for correctness indices and aggregate fast-path split rows."""

from ellmer.experiment_paths import ResultPathInfo
from ellmer.metrics_json_rows import fast_rows_from_payload, prediction_indices


def test_prediction_indices_basic():
    data = [
        {"prediction": 1, "label": 1},
        {"prediction": 0, "label": 1},
        {"prediction": 0, "label": 0},
    ]
    all_idx, correct, incorrect = prediction_indices(data)
    assert all_idx == [0, 1, 2]
    assert correct == [0, 2]
    assert incorrect == [1]


def test_fast_rows_from_metrics_by_prediction_split():
    info = ResultPathInfo(
        model_type="m",
        model_name="mn",
        granularity="attribute",
        session="s",
        run_id=0,
        dataset="d",
        explainer_key="zs_sample",
        results_path="/x",
    )
    payload = {
        "metrics": {"faithfulness": {"zs_sample": 0.5}, "counterfactual_metrics": {"zs_sample": {"validity": 0.1}}},
        "metrics_by_prediction_split": {
            "all": {
                "faithfulness": {"zs_sample": 0.5},
                "counterfactual_metrics": {"zs_sample": {"validity": 0.1, "proximity": 0.2, "sparsity": 0.3, "diversity": 0.0}},
            },
            "correct": {
                "faithfulness": {"zs_sample": 0.6},
                "counterfactual_metrics": {"zs_sample": {"validity": 0.4, "proximity": 0.5, "sparsity": 0.6, "diversity": 0.0}},
            },
            "incorrect": {
                "faithfulness": {"zs_sample": 0.4},
                "counterfactual_metrics": {"zs_sample": {"validity": 0.0, "proximity": 0.0, "sparsity": 0.0, "diversity": 0.0}},
            },
        },
    }
    rows = fast_rows_from_payload(payload, info)
    assert len(rows) == 3
    splits = {r["prediction_split"]: r["faithfulness_auc"] for r in rows}
    assert splits["all"] == 0.5
    assert splits["correct"] == 0.6
    assert splits["incorrect"] == 0.4
    assert all(r["source"] == "json_metrics" for r in rows)
