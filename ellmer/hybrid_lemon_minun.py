"""Hybrid explainer: LEMON (masked LIME) saliency + Minun-style counterfactual search."""
from __future__ import annotations

import operator
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from ellmer.full_certa import FullCerta
from ellmer.post_hoc.explanation_mask import ExplanationMask, all_attribute_keys, all_token_keys
from ellmer.post_hoc.lemon_masked import run_lemon_lime_masked
from ellmer.post_hoc.minun_cf import minun_counterfactual


class HybridLemonMinun(FullCerta):
    """
    Uses LEMON for local saliency and a Minun-style greedy/binary search for counterfactuals.
    Both stages share an ``ExplanationMask`` (modifiable attributes/tokens).

    Depends on the optional ``lemon-explain`` package.
    """

    def __init__(
        self,
        explanation_granularity,
        pred_delegate,
        certa,
        lem_num_features: int = 5,
        lem_num_samples: Optional[int] = None,
        cf_method: str = "greedy",
        cf_k: int = 10,
        cf_max_evals: int = 800,
        combine: str = "freq",
        top_k: int = -1,
        max_predict: int = -1,
        user_mask: Optional[ExplanationMask] = None,
        random_state: int = 0,
        num_triangles: int = 1,
    ):
        super().__init__(explanation_granularity, pred_delegate, certa, num_triangles=num_triangles, max_predict=max_predict)
        self.lem_num_features = lem_num_features
        self.lem_num_samples = lem_num_samples
        self.cf_method = cf_method
        self.cf_k = cf_k
        self.cf_max_evals = cf_max_evals
        self.combine = combine
        self.top_k = top_k
        self.user_mask = user_mask
        self.random_state = random_state
        if combine not in ("freq", "union", "intersection"):
            raise ValueError("combine must be 'freq', 'union', or 'intersection' for mask discovery")

    def _universe_keys(self, ltuple: dict, rtuple: dict):
        if self.explanation_granularity == "attribute":
            return all_attribute_keys(ltuple, rtuple)
        if self.explanation_granularity == "token":
            return all_token_keys(ltuple, rtuple)
        raise ValueError("invalid explanation granularity")

    def _initial_top_k(self, ltuple: dict, rtuple: dict) -> int:
        if self.top_k >= 0:
            return self.top_k
        if self.explanation_granularity == "attribute":
            no_features = len(ltuple) + len(rtuple)
        elif self.explanation_granularity == "token":
            no_features = len(str(ltuple).split(" ")) + len(str(rtuple).split(" "))
        else:
            raise ValueError("invalid explanation granularity")
        return max(1, int(no_features * 0.25))

    def _positive_class(self, prediction) -> int:
        if prediction is None:
            return 0
        if isinstance(prediction, bool):
            return 1 if prediction else 0
        try:
            return 1 if int(prediction) == 1 else 0
        except (TypeError, ValueError):
            return 1 if prediction else 0

    def predict_and_explain(self, ltuple, rtuple, max_predict: int = -1, verbose: bool = False):
        _ = max_predict
        _ = verbose
        prediction = self.delegate.predict_tuples(ltuple, rtuple)
        universe = self._universe_keys(ltuple, rtuple)
        top_k = self._initial_top_k(ltuple, rtuple)
        its = 0
        satisfied = False
        saliency_explanation = {}
        cf_explanation = {}
        pae_dicts: List[dict] = []
        discovery_done = False
        mask: Optional[ExplanationMask] = self.user_mask
        final_mask: Optional[ExplanationMask] = mask

        while not satisfied and its < 10:
            if mask is None:
                if not discovery_done:
                    sal_uncovered, _exp0 = run_lemon_lime_masked(
                        ltuple,
                        rtuple,
                        self.predict_fn,
                        None,
                        explanation_granularity=self.explanation_granularity,
                        num_features=self.lem_num_features,
                        num_samples=self.lem_num_samples,
                        random_state=self.random_state,
                        show_progress=False,
                    )
                    pae_dicts.append(sal_uncovered)
                    discovery_done = True
                mask = self._aggregate_masks_from_dicts(pae_dicts, top_k, universe)
                if len(mask.allowed) == 0:
                    fallback = list(universe)[: max(1, min(top_k, len(universe)))]
                    mask = ExplanationMask(self.explanation_granularity, frozenset(fallback))

            final_mask = mask

            pos_c = self._positive_class(prediction)
            # LEMON and Minun only share inputs (mask, tuples); run concurrently to overlap CPU + predict_fn work.
            rs = self.random_state + its + 17

            def _run_lemon():
                return run_lemon_lime_masked(
                    ltuple,
                    rtuple,
                    self.predict_fn,
                    mask,
                    explanation_granularity=self.explanation_granularity,
                    num_features=self.lem_num_features,
                    num_samples=self.lem_num_samples,
                    random_state=rs,
                    show_progress=False,
                )

            def _run_minun():
                return minun_counterfactual(
                    ltuple,
                    rtuple,
                    self.predict_fn,
                    mask,
                    pos_c,
                    method=self.cf_method,
                    k=self.cf_k,
                    max_evals=self.cf_max_evals,
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_lemon = pool.submit(_run_lemon)
                fut_minun = pool.submit(_run_minun)
                saliency_explanation, _exp = fut_lemon.result()
                cf_explanation, _neval = fut_minun.result()

            has_cf = len(cf_explanation) > 0
            # Saliency is max-abs normalized to [-1, 1]; require both a CF flip and some attribution mass.
            total_abs = 0.0
            for sev in saliency_explanation.values():
                total_abs += abs(float(sev[0]) if isinstance(sev, list) else float(sev))
            if has_cf and total_abs >= 0.1:
                satisfied = True

            if self.user_mask is not None:
                break

            if self.explanation_granularity == "attribute":
                top_k += 1
            else:
                top_k += 5
            its += 1
            # Next iteration: re-derive mask from the same discovery saliency with larger top_k
            if self.user_mask is None:
                mask = None
            no_cap = len(universe)
            if top_k >= no_cap or satisfied:
                break

        filter_features = list(final_mask.allowed) if final_mask is not None else []
        return {
            "prediction": prediction,
            "saliency": saliency_explanation,
            "cf": cf_explanation,
            "filter_features": filter_features,
            "self_explanations": pae_dicts,
            "top_k": top_k,
            "iterations": its,
            "triangles": 0,
        }

    @staticmethod
    def _score_val(v):
        if isinstance(v, list):
            return float(v[0]) if v else 0.0
        return float(v)

    def _aggregate_masks_from_dicts(self, dicts: List[dict], top_k: int, universe) -> ExplanationMask:
        if not dicts:
            return ExplanationMask(self.explanation_granularity, frozenset())

        if self.combine == "freq":
            fc = {}
            for se in dicts:
                if not isinstance(se, dict):
                    continue
                try:
                    fse = {k: v for k, v in se.items() if abs(self._score_val(v)) > 1e-12}
                    if not fse:
                        continue
                    sorted_attributes_dict = sorted(
                        fse.items(), key=lambda kv: abs(self._score_val(kv[1])), reverse=True
                    )[:top_k]
                    for f, _ in sorted_attributes_dict:
                        fc[f] = fc.get(f, 0) + 1
                except Exception:
                    continue
            sorted_fc = sorted(fc.items(), key=operator.itemgetter(1), reverse=True)
            sorted_fc = [(f, c) for f, c in sorted_fc if c > 1][:top_k]
            keys = [sfc[0] for sfc in sorted_fc]
            keys = [k for k in keys if k in universe]
            return ExplanationMask(self.explanation_granularity, frozenset(keys))

        if self.combine == "union":
            fc = {}
            for se in dicts:
                if not isinstance(se, dict):
                    continue
                try:
                    sorted_attributes_dict = sorted(
                        se.items(), key=lambda kv: abs(self._score_val(kv[1])), reverse=True
                    )[:top_k]
                    for f, _ in sorted_attributes_dict:
                        fc[f] = fc.get(f, 0) + 1
                except Exception:
                    continue
            sorted_fc = sorted(fc.items(), key=operator.itemgetter(1), reverse=True)
            sorted_fc = [(f, c) for f, c in sorted_fc if c > 1][:top_k]
            keys = [sfc[0] for sfc in sorted_fc if sfc[0] in universe]
            return ExplanationMask(self.explanation_granularity, frozenset(keys))

        # intersection: in every dict's top-k, and counted in more than one dict
        top_sets = []
        fc = {}
        for se in dicts:
            if not isinstance(se, dict):
                continue
            try:
                sorted_attributes_dict = sorted(
                    se.items(), key=lambda kv: abs(self._score_val(kv[1])), reverse=True
                )[:top_k]
                top_features = {f[0] for f in sorted_attributes_dict}
                top_features &= universe
                top_sets.append(top_features)
                for f in top_features:
                    fc[f] = fc.get(f, 0) + 1
            except Exception:
                continue
        if not top_sets:
            inter = set()
        else:
            inter = set.intersection(*top_sets)
            inter = {f for f in inter if fc.get(f, 0) > 1}
        return ExplanationMask(self.explanation_granularity, frozenset(inter))
