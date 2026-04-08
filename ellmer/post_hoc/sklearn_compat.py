"""
Shims for sklearn API drift used by optional third-party code (e.g. lemon-explain).

``lemon._lemon`` uses ``OneHotEncoder(..., sparse=False)``. scikit-learn 1.2+ renamed
that argument to ``sparse_output``; 1.4+ removed ``sparse`` entirely.
"""
from __future__ import annotations

import functools
import inspect

_applied = False


def apply_lemon_onehot_encoder_compat() -> None:
    """Map legacy ``sparse=`` to ``sparse_output=`` on ``sklearn.preprocessing.OneHotEncoder``."""
    global _applied
    if _applied:
        return

    import sklearn.preprocessing as spp

    base = spp.OneHotEncoder
    sig = inspect.signature(base.__init__)
    if "sparse_output" not in sig.parameters:
        _applied = True
        return

    if getattr(base.__init__, "_ellmer_sparse_compat", False):
        _applied = True
        return

    _orig_init = base.__init__

    @functools.wraps(_orig_init)
    def _compat_init(self, *args, **kwargs):
        if "sparse" in kwargs and "sparse_output" not in kwargs:
            kwargs["sparse_output"] = kwargs.pop("sparse")
        return _orig_init(self, *args, **kwargs)

    _compat_init._ellmer_sparse_compat = True  # type: ignore[attr-defined]
    base.__init__ = _compat_init
    _applied = True
