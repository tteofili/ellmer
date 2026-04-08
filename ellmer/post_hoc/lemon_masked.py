"""
Masked LIME path for LEMON: restrict interpretable features to an ``ExplanationMask``.

Relies on private symbols from ``lemon._lemon`` (same major version as ``lemon-explain`` on PyPI).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ellmer.post_hoc.explanation_mask import ExplanationMask
from ellmer.post_hoc.sklearn_compat import apply_lemon_onehot_encoder_compat


class FilteredInterpretablePair:
    """View onto ``_InterpretableRecordPair`` keeping only a subset of feature indices."""

    __slots__ = ("_allowed", "_irp")

    def __init__(self, irp, allowed_indices: List[int]):
        self._irp = irp
        self._allowed = list(allowed_indices)

    def __len__(self):
        return len(self._allowed)

    def get_all_pos(self, i: int):
        return self._irp.get_all_pos(self._allowed[i])

    def get_first_pos(self, i: int):
        return self._irp.get_first_pos(self._allowed[i])

    def get_value(self, i: int):
        return self._irp.get_value(self._allowed[i])

    @property
    def record_pair(self):
        return self._irp.record_pair

    @property
    def string_representation(self):
        return self._irp.string_representation


def _lemon_feature_allowed(
    irp,
    idx: int,
    mask: Optional[ExplanationMask],
    prefix_a: str,
    prefix_b: str,
) -> bool:
    if mask is None:
        return True
    for pos in irp.get_all_pos(idx):
        source, attr, attr_or_val, j = pos
        ts_val = None
        if j is not None and attr_or_val == "val":
            ts = irp.string_representation.get((source, attr, "val"))
            if ts is not None and j < len(ts):
                ts_val = ts[j]
        if mask.allows_lemon_index(prefix_a, prefix_b, source, attr, attr_or_val, j, ts_val):
            return True
    return False


def apply_explanation_mask(
    irp,
    mask: Optional[ExplanationMask],
    prefix_a: str = "ltable_",
    prefix_b: str = "rtable_",
):
    if mask is None:
        return irp
    allowed = [i for i in range(len(irp)) if _lemon_feature_allowed(irp, i, mask, prefix_a, prefix_b)]
    if len(allowed) == len(irp):
        return irp
    return FilteredInterpretablePair(irp, allowed)


def _require_lemon():
    try:
        import lemon._lemon  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The 'lemon-explain' package is required. Install with: pip install lemon-explain"
        ) from e


def explain_lime_record_pair_masked(
    record_pair: pd.DataFrame,
    predict_proba: Callable,
    mask: Optional[ExplanationMask],
    *,
    num_features: int = 5,
    num_samples: Optional[int] = None,
    granularity: str = "tokens",
    token_representation: str = "record-bow",
    token_patterns="[^ ]+",
    estimate_potential: bool = True,
    explain_attrs: bool = False,
    random_state=None,
    show_progress: bool = False,
    prefix_a: str = "ltable_",
    prefix_b: str = "rtable_",
):
    _require_lemon()
    apply_lemon_onehot_encoder_compat()
    from lemon._lemon import (
        _InterpretableRecordPair,
        _InterpretableSamples,
        _create_explanation,
        _get_predictions,
        _lime,
    )

    if random_state is None:
        random_state = np.random.default_rng()
    elif isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)

    interpretable_record_pair = _InterpretableRecordPair(
        record_pair,
        granularity=granularity,
        token_representation=token_representation,
        features_a=True,
        features_b=True,
        features_attr=explain_attrs,
        features_val=True,
        token_regexes=token_patterns,
    )
    interpretable_record_pair = apply_explanation_mask(interpretable_record_pair, mask, prefix_a, prefix_b)

    if len(interpretable_record_pair) == 0:
        raw = float(
            np.array(
                predict_proba(
                    records_a=record_pair["a"].rename_axis(index="rid"),
                    records_b=record_pair["b"].rename_axis(index="rid"),
                    record_id_pairs=pd.DataFrame(
                        {"a.rid": [record_pair.index[0]], "b.rid": [record_pair.index[0]]}
                    ),
                )
            )[0]
        )
        return _create_explanation(
            interpretable_record_pair,
            coefs={},
            prediction_score=raw,
            dual_explanation=False,
            metadata={
                "r2_score": None,
                "granularity": granularity,
                "token_representation": token_representation,
                "masked": mask is not None,
            },
        )

    if num_samples is None:
        n = len(interpretable_record_pair)
        num_samples = max(min(30 * n, 3000), 500)

    samples = _InterpretableSamples(
        num_samples=num_samples,
        record_pair=interpretable_record_pair,
        random_state=random_state,
        perturb_injection=estimate_potential,
    )
    predictions = _get_predictions(
        samples.features(dummy_encode=False),
        interpretable_record_pair,
        predict_proba,
        random_state,
        show_progress=show_progress,
    )
    coefs, r2_score, _local_prediction = _lime(
        samples.features(),
        predictions,
        samples.distances,
        num_features,
        feature_group_size=(2 if estimate_potential else 1),
    )

    return _create_explanation(
        interpretable_record_pair,
        coefs,
        float(predictions[0]),
        dual_explanation=False,
        metadata={
            "r2_score": r2_score,
            "granularity": granularity,
            "token_representation": token_representation,
            "masked": mask is not None,
        },
    )


def build_lemon_record_pair(records_a: pd.DataFrame, records_b: pd.DataFrame) -> pd.DataFrame:
    ra = records_a.convert_dtypes()
    rb = records_b.convert_dtypes()
    return pd.concat((ra, rb), axis=1, keys=["a", "b"], names=["source", "attribute"])


def lemon_explanation_to_saliency_dict(
    exp,
    *,
    saliency_granularity: str,
    prefix_a: str = "ltable_",
    prefix_b: str = "rtable_",
) -> dict:
    """Map LEMON ``MatchingAttributionExplanation`` to ellmer/CERTA-style ``{feature: [weight]}``.

    Weights are **signed** and scaled by max-|weight| so every value lies in ``[-1, 1]`` (or is zero).
    Sign follows LIME semantics: positive supports the explained/predicted class, negative opposes it.
    """
    acc = {}

    def pos_key(pos: Tuple) -> Optional[str]:
        source, attr, attr_or_val, j = pos
        if attr_or_val == "attr":
            return None
        pre = prefix_a if source == "a" else prefix_b
        if saliency_granularity == "attribute" or j is None:
            return f"{pre}{attr}"
        val_obj = exp.string_representation.get((source, attr, "val"))
        if val_obj is None or isinstance(val_obj, str):
            return f"{pre}{attr}"
        try:
            tok = val_obj[j]
        except (IndexError, TypeError, KeyError):
            return f"{pre}{attr}"
        return f"{pre}{attr}__{tok}"

    for att in exp.attributions:
        w = float(att.weight)
        for pos in att.positions:
            key = pos_key(pos)
            if key is None:
                continue
            acc[key] = acc.get(key, 0.0) + w

    if not acc:
        return {}

    max_abs = max(abs(v) for v in acc.values())
    if max_abs > 0:
        acc = {k: v / max_abs for k, v in acc.items()}

    return {k: [float(v)] for k, v in acc.items()}


def run_lemon_lime_masked(
    ltuple: dict,
    rtuple: dict,
    predict_fn: Callable[[pd.DataFrame], pd.DataFrame],
    mask: Optional[ExplanationMask],
    *,
    explanation_granularity: str,
    num_features: int = 5,
    num_samples: Optional[int] = None,
    random_state: Optional[int] = None,
    show_progress: bool = False,
) -> Tuple[dict, object]:
    from ellmer.post_hoc.lemon_adapter import ltuple_rtuple_to_lemon_frames, make_lemon_predict_proba

    records_a, records_b, _ = ltuple_rtuple_to_lemon_frames(ltuple, rtuple)
    rp = build_lemon_record_pair(records_a, records_b)
    predict_proba = make_lemon_predict_proba(predict_fn)

    lemon_gran = "tokens" if explanation_granularity == "token" else "attributes"

    exp = explain_lime_record_pair_masked(
        rp,
        predict_proba,
        mask,
        num_features=num_features,
        num_samples=num_samples,
        granularity=lemon_gran,
        random_state=random_state,
        show_progress=show_progress,
    )
    saliency = lemon_explanation_to_saliency_dict(exp, saliency_granularity=explanation_granularity)
    return saliency, exp
