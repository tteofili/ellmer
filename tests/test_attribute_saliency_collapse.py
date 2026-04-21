"""Tests for attribute-level saliency: token-key collapse and PASE attribute prompt contract."""
import os
import unittest

from ellmer.llm_output_parse import collapse_saliency_token_keys_to_attributes


class TestCollapseTokenSaliency(unittest.TestCase):
    def test_max_abs_picks_largest_magnitude(self):
        sal = {
            "ltable_Foo__$token1": 0.25,
            "ltable_Foo__$token2": -0.9,
            "rtable_Bar__$token1": 0.1,
        }
        out = collapse_saliency_token_keys_to_attributes(sal, aggregation="max_abs")
        self.assertEqual(out["ltable_Foo"], -0.9)
        self.assertEqual(out["rtable_Bar"], 0.1)

    def test_merges_plain_and_token_keys_same_base(self):
        sal = {
            "ltable_X": 0.5,
            "ltable_X__$token1": 0.8,
        }
        out = collapse_saliency_token_keys_to_attributes(sal, aggregation="max_abs")
        self.assertEqual(len(out), 1)
        self.assertEqual(out["ltable_X"], 0.8)

    def test_sum_aggregation(self):
        sal = {
            "ltable_Foo__$token1": 0.2,
            "ltable_Foo__$token2": 0.3,
        }
        out = collapse_saliency_token_keys_to_attributes(sal, aggregation="sum")
        self.assertAlmostEqual(out["ltable_Foo"], 0.5)

    def test_non_token_keys_unchanged(self):
        sal = {"ltable_Attr": 0.7, "rtable_Attr": 0.2}
        out = collapse_saliency_token_keys_to_attributes(sal)
        self.assertEqual(out, sal)

    def test_empty(self):
        self.assertEqual(collapse_saliency_token_keys_to_attributes({}), {})
        self.assertEqual(collapse_saliency_token_keys_to_attributes(None), {})


class TestConstrained16AttributePrompt(unittest.TestCase):
    def test_saliency_example_has_no_token_suffix(self):
        root = os.path.join(os.path.dirname(__file__), "..", "ellmer", "prompts", "constrained16_attribute.txt")
        root = os.path.normpath(root)
        with open(root, encoding="utf-8") as f:
            text = f.read()
        # Saliency block must not show per-token key pattern from legacy constrained16
        self.assertNotIn("__$token1", text)
        self.assertIn("ltable_$attribute1:", text)
        self.assertIn("do not use any", text.lower())


if __name__ == "__main__":
    unittest.main()
