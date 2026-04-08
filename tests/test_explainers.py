"""Unit tests for ellmer explainers (mocked CERTA / LLM; no network).

Full CERTA/hybrid tests require a dev environment with ``pip install -e .`` (all
``install_requires``, including ``certa`` and PyTorch). Lightweight tests run with
``pandas`` only.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

_CERTA_IMPORT_ERROR = ""
try:
    from ellmer.explainer import BaseLLMExplainer
    from ellmer.full_certa import FullCerta
    from ellmer.hybrid import HybridCerta
    from ellmer.hybrid_lemon_minun import HybridLemonMinun
except ImportError as e:  # pragma: no cover - environment-specific
    _CERTA_IMPORT_ERROR = str(e)
    BaseLLMExplainer = None  # type: ignore[misc,assignment]
    FullCerta = None  # type: ignore[misc,assignment]
    HybridCerta = None  # type: ignore[misc,assignment]
    HybridLemonMinun = None  # type: ignore[misc,assignment]

_SELF_IMPORT_ERROR = ""
try:
    from ellmer.selfexplainer import ICLSelfExplainer, SelfExplainer
except ImportError as e:  # pragma: no cover
    _SELF_IMPORT_ERROR = str(e)
    SelfExplainer = None  # type: ignore[misc,assignment]
    ICLSelfExplainer = None  # type: ignore[misc,assignment]

skip_certa_stack = unittest.skipUnless(
    not _CERTA_IMPORT_ERROR,
    f"requires full ellmer deps (import error: {_CERTA_IMPORT_ERROR})",
)

skip_self_stack = unittest.skipUnless(
    not _SELF_IMPORT_ERROR,
    f"requires selfexplainer deps (import error: {_SELF_IMPORT_ERROR})",
)


def _tables():
    """Minimal DeepMatcher-style tables and tuple dicts."""
    lsource = pd.DataFrame([{"id": 1, "title": "apple", "price": "10"}])
    rsource = pd.DataFrame([{"id": 2, "title": "apple", "price": "12"}])
    ltuple = {"id": 1, "title": "apple", "price": "10"}
    rtuple = {"id": 2, "title": "apple", "price": "12"}
    return lsource, rsource, ltuple, rtuple


class MockDelegate:
    """Delegate for FullCerta / Hybrid with predict_fn contract."""

    pred_count = 0
    tokens = 0

    def __init__(self):
        self.llm = MagicMock()

    def predict_tuples(self, ltuple, rtuple):
        return True

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        self.pred_count += len(df)
        return pd.DataFrame(
            {"nomatch_score": [0.1] * len(df), "match_score": [0.9] * len(df)}
        )

    def count_predictions(self):
        return self.pred_count

    def count_tokens(self):
        return self.tokens


class MockCertaExplainer:
    def __init__(self, lsource, rsource):
        self.lsource = lsource
        self.rsource = rsource

    def explain(self, *args, **kwargs):
        saliency_df = pd.DataFrame([[0.4, 0.3]], columns=["ltable_title", "ltable_price"])
        cf_summary = pd.Series({"set_a": 0.5, "set_b": 0.4})
        cfs = pd.DataFrame(
            {
                "altered_attributes": ["ltable_title"],
                "dropped_values": [""],
                "copied_values": [""],
                "triangle": ["t0"],
                "attr_count": [1],
                "ltable_title": ["banana"],
            }
        )
        tri = []
        support = pd.DataFrame()
        return saliency_df, cf_summary, cfs, tri, None, support


class MockEllmer:
    def __init__(self, saliency: dict):
        self._saliency = saliency

    def predict_and_explain(self, ltuple, rtuple):
        return {"prediction": True, "saliency": self._saliency, "cf": {}}


class TestExplanationMask(unittest.TestCase):
    """Mask helpers used by LEMON/Minun hybrid (no CERTA import)."""

    def test_mask_from_top_saliency_uses_absolute_scores(self):
        from ellmer.post_hoc.explanation_mask import ExplanationMask, mask_from_top_saliency

        sal = {"a": [-0.8], "b": [0.3], "c": [0.1]}
        m = mask_from_top_saliency(sal, "attribute", top_k=2, universe={"a", "b", "c"})
        self.assertIsInstance(m, ExplanationMask)
        picked = set(m.allowed)
        self.assertIn("a", picked)
        self.assertEqual(len(picked), 2)


class TestMinunCandidateHelpers(unittest.TestCase):
    def test_generate_candidates_neg_identity_first(self):
        from ellmer.post_hoc.minun_cf import generate_candidates_from_one_attribute_neg

        c = generate_candidates_from_one_attribute_neg("a b", "a c")
        self.assertEqual(c[0], "a b")


class TestLemonSaliencyDict(unittest.TestCase):
    """LEMON -> ellmer saliency mapping (pandas-only import path)."""

    def test_scores_normalized_to_unit_magnitude(self):
        from ellmer.post_hoc.lemon_masked import lemon_explanation_to_saliency_dict

        att = MagicMock()
        att.weight = -0.25
        att.positions = [("a", "title", "val", 0)]
        exp = MagicMock()
        exp.attributions = [att]

        class _TokSeq:
            def __getitem__(self, i):
                if i == 0:
                    return "foo"
                raise IndexError

        exp.string_representation = {("a", "title", "val"): _TokSeq()}
        d = lemon_explanation_to_saliency_dict(exp, saliency_granularity="token")
        self.assertIn("ltable_title__foo", d)
        self.assertAlmostEqual(d["ltable_title__foo"][0], -1.0)


@skip_certa_stack
class TestBaseLLMExplainer(unittest.TestCase):
    def test_default_predict_and_explain_returns_false_prediction_and_none_rest(self):
        e = BaseLLMExplainer()
        out = e.predict_and_explain({}, {})
        self.assertIn("prediction", out)
        self.assertFalse(out["prediction"])
        self.assertIsNone(out.get("saliency"))
        self.assertIsNone(out.get("cf"))


@skip_certa_stack
class TestFullCerta(unittest.TestCase):
    def test_predict_and_explain_shapes(self):
        lsource, rsource, ltuple, rtuple = _tables()
        delegate = MockDelegate()
        certa = MockCertaExplainer(lsource, rsource)
        fc = FullCerta("attribute", delegate, certa, num_triangles=2)
        out = fc.predict_and_explain(ltuple, rtuple)
        self.assertTrue(out["prediction"])
        self.assertIsInstance(out["saliency"], dict)
        self.assertIn("ltable_title", out["saliency"])
        self.assertIsInstance(out["cf"], dict)


@skip_certa_stack
class TestHybridCerta(unittest.TestCase):
    def test_freq_requires_repeated_feature_in_topk(self):
        lsource, rsource, ltuple, rtuple = _tables()
        delegate = MockDelegate()
        certa = MockCertaExplainer(lsource, rsource)
        e1 = MockEllmer({"ltable_title": 0.9, "ltable_price": 0.05})
        e2 = MockEllmer({"ltable_title": 0.8, "rtable_price": 0.05})
        hy = HybridCerta(
            "attribute",
            delegate,
            certa,
            ellmers=[e1, e2],
            num_draws=1,
            num_triangles=2,
            combine="freq",
            top_k=2,
        )
        out = hy.predict_and_explain(ltuple, rtuple, max_predict=10)
        self.assertIn("filter_features", out)
        self.assertIn("ltable_title", out["filter_features"])
        self.assertIsInstance(out["saliency"], dict)

    def test_single_ellmer_yields_empty_freq_mask(self):
        lsource, rsource, ltuple, rtuple = _tables()
        delegate = MockDelegate()
        certa = MockCertaExplainer(lsource, rsource)
        e1 = MockEllmer({"ltable_title": 0.9})
        hy = HybridCerta(
            "attribute",
            delegate,
            certa,
            ellmers=[e1],
            num_draws=1,
            num_triangles=2,
            combine="freq",
            top_k=3,
        )
        out = hy.predict_and_explain(ltuple, rtuple, max_predict=10)
        self.assertEqual(out["filter_features"], [])


@skip_certa_stack
class TestHybridLemonMinun(unittest.TestCase):
    @patch("ellmer.hybrid_lemon_minun.minun_counterfactual")
    @patch("ellmer.hybrid_lemon_minun.run_lemon_lime_masked")
    def test_returns_expected_keys(self, mock_lemon, mock_minun):
        lsource, rsource, ltuple, rtuple = _tables()
        delegate = MockDelegate()
        certa = MockCertaExplainer(lsource, rsource)

        def lemon_side(lt, rt, pfn, mask, **kwargs):
            if mask is None:
                return {"ltable_title": [0.5], "ltable_price": [0.2]}, MagicMock()
            return {"ltable_title": [1.0]}, MagicMock()

        mock_lemon.side_effect = lemon_side
        mock_minun.return_value = ({"ltable_title": "x"}, 3)

        hy = HybridLemonMinun(
            "attribute",
            delegate,
            certa,
            lem_num_features=2,
            lem_num_samples=5,
            top_k=2,
            combine="freq",
        )
        out = hy.predict_and_explain(ltuple, rtuple)
        self.assertIn("prediction", out)
        self.assertIn("saliency", out)
        self.assertIn("cf", out)
        self.assertIn("filter_features", out)
        self.assertIn("self_explanations", out)
        mock_minun.assert_called()
        self.assertGreaterEqual(mock_lemon.call_count, 2)


@skip_self_stack
class TestSelfExplainer(unittest.TestCase):
    @patch.object(SelfExplainer, "_invoke")
    @patch("ellmer.selfexplainer.ellmer.utils.read_prompt")
    def test_ptse_er_only_match(self, mock_read_prompt, mock_invoke):
        mock_read_prompt.return_value = [("system", "You match records."), ("human", "{feature}\n{ltuple}\n{rtuple}")]
        mock_invoke.return_value = "yes, these records are matching"
        llm = MagicMock()
        se = SelfExplainer(
            model_type="delegate",
            delegate=llm,
            explanation_granularity="attribute",
            prompts={"ptse": {"er": "dummy_er_prompt.txt"}},
        )
        pred = se.predict_tuples({"a": "1"}, {"b": "2"})
        self.assertEqual(pred, 1)
        mock_invoke.assert_called()


@skip_self_stack
class TestICLSelfExplainer(unittest.TestCase):
    @patch("ellmer.selfexplainer.FewShotChatMessagePromptTemplate")
    @patch("ellmer.selfexplainer.ChatPromptTemplate")
    @patch("ellmer.selfexplainer.ellmer.utils.read_prompt")
    def test_predict_and_explain_parses_structured_answer(self, mock_read, mock_chat_cls, mock_fewshot):
        _, _, ltuple, rtuple = _tables()
        mock_read.return_value = [("system", "ex")]
        mock_fewshot.return_value = MagicMock()

        answer = (
            'prediction:1, saliency: {"ltable_title": 0.9}, counterfactual: {"ltable_title": "pear"} trail'
        )
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = MagicMock(content=answer)
        fake_prompt = MagicMock()
        fake_prompt.__or__ = MagicMock(return_value=fake_chain)
        mock_chat_cls.from_messages.return_value = fake_prompt

        icl = object.__new__(ICLSelfExplainer)
        icl.examples = []
        icl.llm = MagicMock()
        icl.prompts = {
            "fs": "ellmer/prompts/fs1.txt",
            "input": "record1:\n{ltuple}\n record2:\n{rtuple}\n",
        }
        icl.pred_count = 0
        icl.tokens = 0

        out = ICLSelfExplainer.predict_and_explain(icl, ltuple, rtuple)

        self.assertEqual(out["prediction"], "1")
        self.assertIn("ltable_title", out.get("saliency", {}))
        self.assertIn("ltable_title", out.get("cf", {}))


class TestLlmOutputParse(unittest.TestCase):
    """Staged CoT output parsers (no LLM)."""

    def test_extract_tagged_block(self):
        from ellmer.llm_output_parse import extract_tagged_block

        text = "noise\nBEGIN_SALIENCY\na\t1.0\nEND_SALIENCY\ntrailer"
        self.assertEqual(extract_tagged_block(text, "BEGIN_SALIENCY", "END_SALIENCY"), "a\t1.0")

    def test_parse_tsv_scores_skips_comments(self):
        from ellmer.llm_output_parse import parse_tsv_scores

        block = "# hdr\nltable_x\t0.5\n\nrtable_y\t-0.25\nbadline\n"
        d = parse_tsv_scores(block)
        self.assertAlmostEqual(d["ltable_x"], 0.5)
        self.assertAlmostEqual(d["rtable_y"], -0.25)
        self.assertEqual(len(d), 2)

    def test_parse_prediction_line(self):
        from ellmer.llm_output_parse import parse_prediction_line

        self.assertEqual(
            parse_prediction_line("<thinking>x</thinking>\nPREDICTION: 1\n"),
            1,
        )
        self.assertEqual(parse_prediction_line("PREDICTION: 0"), 0)
        self.assertEqual(parse_prediction_line("prefix\nPREDICTION: 1\ntrailer"), 1)

    def test_parse_saliency_tsv_and_json(self):
        from ellmer.llm_output_parse import parse_saliency_response

        tsv = "blah\nBEGIN_SALIENCY\nltable_title\t0.9\nEND_SALIENCY\n"
        self.assertAlmostEqual(parse_saliency_response(tsv)["ltable_title"], 0.9)

        js = 'talk\nSALIENCY_JSON: {"ltable_a": 0.2, "rtable_b": 0.8}\n'
        sal = parse_saliency_response(js)
        self.assertAlmostEqual(sal["ltable_a"], 0.2)
        self.assertAlmostEqual(sal["rtable_b"], 0.8)

    def test_normalize_saliency_nested(self):
        from ellmer.llm_output_parse import normalize_saliency_dict

        d = normalize_saliency_dict({"k": {"saliency": 0.5}, "j": {"saliency_score": [0.1]}})
        self.assertAlmostEqual(d["k"], 0.5)
        self.assertAlmostEqual(d["j"], 0.1)

    def test_parse_cf_tsv(self):
        from ellmer.llm_output_parse import parse_cf_tsv_or_json

        lt = {"id": "1", "title": "a"}
        rt = {"id": "2", "title": "b"}
        text = "BEGIN_CF\nltable_title\tpear\nEND_CF"
        cf = parse_cf_tsv_or_json(text, lt, rt)
        self.assertEqual(cf["ltable_title"], "pear")


@skip_self_stack
class TestStagedSelfExplainer(unittest.TestCase):
    @patch.object(SelfExplainer, "_invoke")
    @patch("ellmer.selfexplainer.ellmer.utils.read_prompt")
    def test_ptse_staged_three_calls(self, mock_read_prompt, mock_invoke):
        mock_read_prompt.return_value = [
            ("system", "s"),
            ("human", "{ltuple}\n{rtuple}\n{prediction_int}\n{prediction_label}\n{feature}"),
        ]
        mock_invoke.side_effect = [
            "<thinking>t</thinking>\nPREDICTION: 1\n",
            "BEGIN_SALIENCY\nltable_title\t0.9\nEND_SALIENCY\n",
            "BEGIN_CF\nltable_title\tz\nEND_CF\n",
        ]
        llm = MagicMock()
        se = SelfExplainer(
            model_type="delegate",
            delegate=llm,
            explanation_granularity="attribute",
            prompts={
                "ptse_staged": {
                    "er": "dummy_er.txt",
                    "saliency": "dummy_sal.txt",
                    "cf": "dummy_cf.txt",
                }
            },
        )
        lt, rt = {"title": "a"}, {"title": "b"}
        out = se.predict_and_explain(lt, rt)
        self.assertEqual(out["prediction"], 1)
        self.assertAlmostEqual(out["saliency"]["ltable_title"], 0.9)
        self.assertEqual(out["cf"]["ltable_title"], "z")
        self.assertEqual(mock_invoke.call_count, 3)
        self.assertIn("llm_time", out)
        self.assertIsInstance(out["conversation"], list)


if __name__ == "__main__":
    unittest.main()
