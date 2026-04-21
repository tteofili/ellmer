"""Tests for sklearn compatibility shims (lemon-explain + newer scikit-learn)."""
from __future__ import annotations

import unittest

import numpy as np


class TestLemonOneHotCompat(unittest.TestCase):
    def test_sparse_kw_maps_to_sparse_output(self):
        import sklearn.preprocessing as spp

        from ellmer.post_hoc.sklearn_compat import apply_lemon_onehot_encoder_compat

        apply_lemon_onehot_encoder_compat()
        X = np.array([[0, 1], [1, 0], [2, 1]])
        categories = [[0, 1, 2]] * X.shape[1]
        out = spp.OneHotEncoder(
            categories=categories, drop="first", sparse=False
        ).fit_transform(X)
        self.assertEqual(out.shape[0], 3)


if __name__ == "__main__":
    unittest.main()
