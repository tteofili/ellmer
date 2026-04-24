"""Tests for ``out_claude`` eval.csv ingest."""

from __future__ import annotations

from pathlib import Path

import pytest

from ellmer.out_claude_eval_ingest import load_out_claude_long

_REPO = Path(__file__).resolve().parents[1]
_CLAUDE = _REPO / "out" / "claude"


@pytest.mark.skipif(
    not _CLAUDE.is_dir() or not any(_CLAUDE.rglob("eval.csv")),
    reason="out/claude eval exports not present",
)
def test_load_out_claude_long_default_is_correct_incorrect_only() -> None:
    df = load_out_claude_long(_CLAUDE)
    assert not df.empty
    assert set(df["prediction_split"].unique()) <= {"correct", "incorrect"}


@pytest.mark.skipif(
    not _CLAUDE.is_dir() or not any(_CLAUDE.rglob("eval.csv")),
    reason="out/claude eval exports not present",
)
def test_load_out_claude_long_can_include_all_split() -> None:
    df_pair = load_out_claude_long(_CLAUDE)
    df_all = load_out_claude_long(
        _CLAUDE, prediction_splits=("all", "correct", "incorrect")
    )
    assert not df_all.empty
    assert "all" in set(df_all["prediction_split"].unique())
    assert len(df_all) > len(df_pair)
