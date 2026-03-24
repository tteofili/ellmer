"""Unified mask for LEMON + Minun: which attributes/tokens post-hoc methods may perturb."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Set


@dataclass(frozen=True)
class ExplanationMask:
    """Keys use CERTA/ellmer conventions: ``ltable_*`` / ``rtable_*`` or ``col__token`` for token mode."""

    granularity: str  # "attribute" | "token"
    allowed: FrozenSet[str] = field(default_factory=frozenset)

    @staticmethod
    def normalize_key(key: str) -> str:
        k = key.strip()
        return k

    def with_keys(self, keys) -> "ExplanationMask":
        return ExplanationMask(
            self.granularity,
            frozenset(self.normalize_key(k) for k in keys),
        )

    def union_keys(self, keys) -> "ExplanationMask":
        s = set(self.allowed)
        s.update(self.normalize_key(k) for k in keys)
        return ExplanationMask(self.granularity, frozenset(s))

    def left_attribute_names(self) -> Set[str]:
        """Unprefixed left-column names permitted for Minun (left-entity edits)."""
        out: Set[str] = set()
        for k in self.allowed:
            if k.startswith("ltable_"):
                rest = k[len("ltable_") :]
                if self.granularity == "token" and "__" in rest:
                    rest = rest.split("__")[0]
                if rest != "id":
                    out.add(rest)
        return out

    def allows_lemon_index(self, prefix_a: str, prefix_b: str, source: str, attr: str, attr_or_val: str, j, ts_val) -> bool:
        """
        Whether a LEMON interpretable position may be perturbed.
        ``prefix_a``/``prefix_b`` are typically ``ltable_`` / ``rtable_``.
        ``j`` is token index or None (attribute value treated as one unit).
        ``ts_val`` is string token at j when applicable.
        """
        pre = prefix_a if source == "a" else prefix_b
        col = f"{pre}{attr}"
        if self.granularity == "attribute":
            return col in self.allowed
        # token granularity: keys like ltable_name__foo
        if j is None:
            return col in self.allowed
        tok = str(ts_val) if ts_val is not None else ""
        return f"{col}__{tok}" in self.allowed


_TOKEN_RE = re.compile(r"\S+")


def tokens_in_value(val: str):
    return _TOKEN_RE.findall(str(val))


def token_keys_for_record(prefix: str, tup: dict) -> Set[str]:
    keys: Set[str] = set()
    for attr, val in tup.items():
        if attr == "id":
            continue
        col = f"{prefix}{attr}"
        for tok in tokens_in_value(val):
            keys.add(f"{col}__{tok}")
    return keys


def all_attribute_keys(ltuple: dict, rtuple: dict) -> Set[str]:
    keys: Set[str] = set()
    for attr in ltuple:
        if attr != "id":
            keys.add(f"ltable_{attr}")
    for attr in rtuple:
        if attr != "id":
            keys.add(f"rtable_{attr}")
    return keys


def all_token_keys(ltuple: dict, rtuple: dict) -> Set[str]:
    return token_keys_for_record("ltable_", ltuple).union(token_keys_for_record("rtable_", rtuple))


def mask_from_top_saliency(saliency: dict, granularity: str, top_k: int, universe: Optional[Set[str]] = None) -> ExplanationMask:
    """Pick top_k keys by largest |saliency|; optionally intersect with ``universe``."""
    scored = []
    for k, v in saliency.items():
        if v is None:
            continue
        if isinstance(v, list):
            score = float(v[0]) if v else 0.0
        else:
            try:
                score = float(v)
            except (TypeError, ValueError):
                continue
        if abs(score) > 1e-12:
            scored.append((k, abs(score)))
    scored.sort(key=lambda x: x[1], reverse=True)
    picked = [k for k, _ in scored[:top_k]]
    if universe is not None:
        picked = [k for k in picked if k in universe]
    return ExplanationMask(granularity, frozenset(picked))
