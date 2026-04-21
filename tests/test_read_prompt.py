"""Tests for ellmer.utils.read_prompt (multi-line role::body and :: in body).

Use an editable install so imports match the repo: ``pip install -e .`` from the project root.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ellmer.utils import read_prompt


class TestReadPrompt(unittest.TestCase):
    def setUp(self):
        read_prompt.cache_clear()

    def tearDown(self):
        read_prompt.cache_clear()

    def test_cot_staged_er_merges_continuations(self):
        root = Path(__file__).resolve().parent.parent
        path = root / "ellmer" / "prompts" / "cot_staged_er.txt"
        msgs = read_prompt(str(path))
        self.assertTrue(all(len(m) == 2 for m in msgs))
        roles = [m[0] for m in msgs]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

        first_user = msgs[1][1]
        self.assertIn("Use chain-of-thought", first_user)
        self.assertIn("After your reasoning", first_user)

        last_user = msgs[3][1]
        self.assertIn("Granularity for later explanations", last_user)
        self.assertIn("record1:", last_user)
        self.assertIn("{ltuple}", last_user)

    def test_cot_staged_saliency_and_cf_parse(self):
        root = Path(__file__).resolve().parent.parent
        for name in ("cot_staged_saliency.txt", "cot_staged_cf.txt"):
            path = root / "ellmer" / "prompts" / name
            msgs = read_prompt(str(path))
            self.assertTrue(all(len(m) == 2 for m in msgs), msg=name)
            self.assertGreater(len(msgs), 0, msg=name)
            for role, body in msgs:
                self.assertIn(role, ("system", "user", "assistant", "human"))
                self.assertIsInstance(body, str)

    def test_body_may_contain_double_colon(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("user::Say :: hello and :: world\n")
            f.flush()
            path = f.name
        try:
            msgs = read_prompt(path)
            self.assertEqual(msgs, [("user", "Say :: hello and :: world")])
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
