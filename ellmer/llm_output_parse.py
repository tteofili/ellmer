"""Parse LLM outputs for staged CoT ER explainers (tagged blocks, TSV, prediction lines)."""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

PredictionType = Union[int, bool, None]


def extract_tagged_block(text: str, begin: str, end: str) -> Optional[str]:
    """Return content between the first ``begin`` and the first ``end`` after it (exclusive)."""
    if not text:
        return None
    i = text.find(begin)
    if i < 0:
        return None
    j = text.find(end, i + len(begin))
    if j < 0:
        return None
    return text[i + len(begin) : j].strip()


def _parse_float_loose(s: str) -> Optional[float]:
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def parse_tsv_scores(block: str) -> Dict[str, float]:
    """
    Parse lines ``key<TAB>score`` into a dict of string -> float.
    Skips empty lines and lines starting with ``#``.
    """
    out: Dict[str, float] = {}
    if not block:
        return out
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" not in line:
            continue
        key, rest = line.split("\t", 1)
        key = key.strip()
        val = _parse_float_loose(rest)
        if key and val is not None:
            out[key] = val
    return out


def parse_line_prefixed_json(text: str, prefix: str) -> Optional[dict]:
    """Find a line starting with ``prefix`` and parse the remainder as JSON object."""
    if not text:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(prefix):
            continue
        payload = line[len(prefix) :].strip()
        if not payload:
            continue
        try:
            obj = json.loads(payload)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _find_json_object_after_marker(text: str, marker: str) -> Optional[dict]:
    idx = text.find(marker)
    if idx < 0:
        return None
    sub = text[idx + len(marker) :].lstrip()
    if not sub.startswith("{"):
        brace = sub.find("{")
        if brace < 0:
            return None
        sub = sub[brace:]
    depth = 0
    end_i = -1
    for i, ch in enumerate(sub):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_i = i + 1
                break
    if end_i < 0:
        return None
    chunk = sub[:end_i]
    try:
        obj = json.loads(chunk)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(chunk)
            return obj if isinstance(obj, dict) else None
        except (SyntaxError, ValueError):
            return None


def _find_first_json_object(text: str) -> Optional[dict]:
    """Return first parseable JSON/Python-dict object found in text."""
    if not text or "{" not in text:
        return None
    start = text.find("{")
    while start >= 0:
        depth = 0
        end_i = -1
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_i = i + 1
                    break
        if end_i < 0:
            return None
        chunk = text[start:end_i]
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            try:
                obj = ast.literal_eval(chunk)
                if isinstance(obj, dict):
                    return obj
            except (SyntaxError, ValueError):
                pass
        start = text.find("{", start + 1)
    return None


def parse_saliency_response(text: str) -> Dict[str, Any]:
    """
    Prefer ``BEGIN_SALIENCY`` / ``END_SALIENCY`` TSV block; else ``SALIENCY_JSON:`` line or inline object.
    Normalizes nested ``{key: {saliency: n}}`` to flat floats where applicable.
    """
    if not text:
        return {}
    block = extract_tagged_block(text, "BEGIN_SALIENCY", "END_SALIENCY")
    raw: Dict[str, Any] = {}
    if block is not None:
        raw = parse_tsv_scores(block)
    if not raw:
        lj = parse_line_prefixed_json(text, "SALIENCY_JSON:")
        if lj:
            if "saliency_explanation" in lj and isinstance(lj["saliency_explanation"], dict):
                raw = lj["saliency_explanation"]
            else:
                raw = lj
    if not raw:
        jd = _find_json_object_after_marker(text, "SALIENCY_JSON:")
        if jd:
            if "saliency_explanation" in jd and isinstance(jd["saliency_explanation"], dict):
                raw = jd["saliency_explanation"]
            else:
                raw = jd
    if not raw:
        jfirst = _find_first_json_object(text)
        if jfirst:
            if "saliency_explanation" in jfirst and isinstance(jfirst["saliency_explanation"], dict):
                raw = jfirst["saliency_explanation"]
            else:
                raw = jfirst
    return normalize_saliency_dict(raw)


def parse_prediction_line(text: str, llm_fn: Optional[Callable[..., Any]] = None) -> PredictionType:
    """
    Parse ``PREDICTION: 0`` or ``PREDICTION: 1`` (last occurrence wins).
    Falls back to ``ellmer.utils.text_to_match`` when no marker is found.
    """
    if not text:
        return 0
    matches = list(re.finditer(r"(?im)^\s*PREDICTION:\s*([01])\s*$", text))
    if matches:
        return int(matches[-1].group(1))
    # tolerate "PREDICTION: 1" inline
    matches2 = list(re.finditer(r"(?i)PREDICTION:\s*([01])\b", text))
    if matches2:
        return int(matches2[-1].group(1))
    if llm_fn is not None:
        try:
            _, match_score = _text_to_match_safe(text, llm_fn)
            return int(match_score) if match_score else 0
        except Exception:
            pass
    return 0


def _text_to_match_safe(answer: str, llm_fn: Callable[..., Any]) -> Tuple[int, int]:
    import ellmer.utils

    return ellmer.utils.text_to_match(answer, llm_fn)


def parse_cf_tsv_or_json(
    text: str,
    ltuple: dict,
    rtuple: dict,
) -> Dict[str, Any]:
    """
    Prefer ``BEGIN_CF`` / ``END_CF`` TSV ``field<TAB>value`` (merged flat record).
    Else JSON after ``CF_JSON:`` or braced object; reuse key-normalization rules from legacy ptse.
    """
    if not text:
        return {}
    block = extract_tagged_block(text, "BEGIN_CF", "END_CF")
    cf_explanation: Dict[str, Any] = {}
    if block is not None:
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" not in line:
                continue
            k, v = line.split("\t", 1)
            k = k.strip()
            if k:
                cf_explanation[k] = v.strip()
    if not cf_explanation:
        lj = parse_line_prefixed_json(text, "CF_JSON:")
        if lj:
            cf_explanation = _normalize_cf_inner(lj, ltuple, rtuple)
    if not cf_explanation:
        jd = _find_json_object_after_marker(text, "CF_JSON:")
        if jd:
            cf_explanation = _normalize_cf_inner(jd, ltuple, rtuple)
    if not cf_explanation and "{" in text:
        jfirst = _find_first_json_object(text)
        if jfirst:
            cf_explanation = _normalize_cf_inner(jfirst, ltuple, rtuple)
    return cf_explanation


def _normalize_cf_inner(cf_dict: Any, ltuple: dict, rtuple: dict) -> Dict[str, Any]:
    if not isinstance(cf_dict, dict):
        return {}
    keys = list(cf_dict.keys())
    if "record_after" in keys:
        cf_explanation = dict(cf_dict["record_after"])
        if cf_explanation:
            k0 = next(iter(cf_explanation.keys()))
            # Legacy: cf_explanation | other_record (other overwrites on key clash)
            if str(k0).startswith("rtable_"):
                cf_explanation = {**cf_explanation, **dict(ltuple)}
            else:
                cf_explanation = {**cf_explanation, **dict(rtuple)}
        return cf_explanation
    if "counterfactual_record" in keys:
        return dict(cf_dict["counterfactual_record"])
    if "counterfactual_explanation" in keys:
        return dict(cf_dict["counterfactual_explanation"])
    if "counterfactual" in keys:
        v = cf_dict["counterfactual"]
        return dict(v) if isinstance(v, dict) else {}
    if "record1" in keys and "record2" in keys:
        r1 = dict(cf_dict["record1"])
        r2 = dict(cf_dict["record2"])
        for k in list(r1.keys()):
            if not str(k).startswith("ltable_"):
                r1["ltable_" + str(k)] = r1.pop(k)
        for k in list(r2.keys()):
            if not str(k).startswith("rtable_"):
                r2["rtable_" + str(k)] = r2.pop(k)
        return {**r1, **r2}
    return dict(cf_dict)


def normalize_saliency_dict(saliency: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten ``{k: {'saliency': x}}`` / ``saliency_score``; coerce numeric values to float."""
    if not saliency:
        return {}
    out: Dict[str, Any] = {}
    for k, v in saliency.items():
        if isinstance(v, dict):
            if "saliency" in v:
                inner = v["saliency"]
            elif "saliency_score" in v:
                inner = v["saliency_score"]
            else:
                inner = v
        else:
            inner = v
        if isinstance(inner, (list, tuple)) and inner:
            try:
                out[k] = float(inner[0])
            except (TypeError, ValueError):
                out[k] = inner
        else:
            try:
                out[k] = float(inner)
            except (TypeError, ValueError):
                if inner is not None:
                    out[k] = inner
    return out


def saliency_score_value(v: Any, default: float = 0.0) -> float:
    """Extract a numeric saliency score from scalar/list wrappers, preserving sign."""
    try:
        if isinstance(v, (list, tuple)):
            if not v:
                return default
            return float(v[0])
        if isinstance(v, dict):
            if "saliency" in v:
                return float(v["saliency"])
            if "saliency_score" in v:
                inner = v["saliency_score"]
                if isinstance(inner, (list, tuple)):
                    return float(inner[0]) if inner else default
                return float(inner)
        return float(v)
    except (TypeError, ValueError, IndexError, KeyError):
        return default


def to_numeric_saliency_map(saliency: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Coerce a saliency mapping to ``dict[str, float]`` while preserving signs.

    Interpretation for signed explainers (self/LIME-style): positive scores support
    the explained/predicted class; negative scores oppose it.
    """
    if not isinstance(saliency, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in saliency.items():
        out[k] = saliency_score_value(v, default=0.0)
    return out


# Keys like ``ltable_Beer_Name__$token1`` (per-token saliency under attribute runs).
_TOKEN_SUFFIX_KEY = re.compile(r"^(.+?)__\$token\d+$")


def collapse_saliency_token_keys_to_attributes(
    saliency: Optional[Dict[str, Any]],
    *,
    aggregation: str = "max_abs",
) -> Dict[str, Any]:
    """
    Merge per-token saliency keys (``...__$tokenN``) into one key per attribute.

    Used when ``explanation_granularity`` is ``attribute`` but the model still outputs
    token-shaped keys (e.g. legacy PASE prompts or noncompliant completions).

    **max_abs** (default): for each attribute base key, keep the **stored value** of the
    token whose score has largest absolute magnitude (sign preserved).

    **sum**: sum scalar scores per base key (outputs floats).
    """
    if not saliency or not isinstance(saliency, dict):
        return {}
    groups: Dict[str, List[Tuple[Any, float]]] = defaultdict(list)
    for k, v in saliency.items():
        sk = str(k)
        m = _TOKEN_SUFFIX_KEY.match(sk)
        base = m.group(1) if m else sk
        groups[base].append((v, saliency_score_value(v, default=0.0)))

    out: Dict[str, Any] = {}
    for base, items in groups.items():
        if aggregation == "sum":
            out[base] = sum(s for _, s in items)
        else:
            best_v, _ = max(items, key=lambda t: abs(t[1]))
            out[base] = best_v
    return out
