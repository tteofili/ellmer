from __future__ import annotations

import pandas as pd

from ellmer.llm_output_parse import to_numeric_saliency_map
from ellmer.metrics import get_faithfulness
from ellmer.post_hoc.lemon_masked import lemon_explanation_to_saliency_dict


def test_to_numeric_saliency_map_preserves_signed_values():
    d = to_numeric_saliency_map(
        {
            "a": {"saliency": -0.4},
            "b": {"saliency_score": [0.2]},
            "c": [-0.1],
            "d": "0.5",
        }
    )
    assert d["a"] == -0.4
    assert d["b"] == 0.2
    assert d["c"] == -0.1
    assert d["d"] == 0.5


def test_faithfulness_uses_signed_descending_order_for_non_match_predictions():
    # Under signed-default policy, we always rank by descending raw saliency.
    # For prediction=0, this must still pick +0.2 above -0.9.
    test_set_df = pd.DataFrame(
        [{"ltable_a": "A", "rtable_a": "B", "label": 0, "match": 0}]
    )
    results_by_name = {
        "signed_model": [
            {"id": 0, "prediction": 0, "saliency": {"ltable_a": -0.9, "rtable_a": 0.2}}
        ]
    }

    def eval_fn(masked_df: pd.DataFrame) -> float:
        # returns 1 only if ltable_a got masked
        return 1.0 if masked_df.iloc[0]["ltable_a"] == "" else 0.0

    aucs = get_faithfulness(
        ["signed_model"], eval_fn, base_dir="", test_set_df=test_set_df, results_by_name=results_by_name
    )
    assert aucs["signed_model"] == 0.0


def test_lemon_signed_orientation_kept():
    class _Att:
        def __init__(self, weight, positions):
            self.weight = weight
            self.positions = positions

    class _TokSeq:
        def __getitem__(self, i):
            if i == 0:
                return "foo"
            raise IndexError

    class _Exp:
        pass

    exp = _Exp()
    exp.attributions = [_Att(-0.25, [("a", "title", "val", 0)])]
    exp.string_representation = {("a", "title", "val"): _TokSeq()}
    d = lemon_explanation_to_saliency_dict(exp, saliency_granularity="token")
    assert d["ltable_title__foo"][0] < 0

