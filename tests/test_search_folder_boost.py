"""Tests for the folder-prefix ranking-boost helper."""

from __future__ import annotations

from dataclasses import dataclass

from markdown_vault_mcp.managers.search import _apply_folder_boost


@dataclass
class _Row:
    """Stand-in for FTSResult / _SemanticRow / _GroupableFTS."""

    path: str
    folder: str
    score: float


def test_none_and_empty_weights_are_identity():
    rows = [_Row("a.md", "sessions", 1.0), _Row("b.md", "", 0.5)]
    for weights in (None, {}):
        out = _apply_folder_boost(rows, weights=weights)
        assert [r.path for r in out] == ["a.md", "b.md"]
        assert [r.score for r in out] == [1.0, 0.5]


def test_exact_folder_match_scales_score():
    rows = [_Row("s.md", "sessions", 1.0)]
    out = _apply_folder_boost(rows, weights={"sessions": 0.5})
    assert out[0].score == 0.5


def test_subfolder_matches_prefix():
    rows = [_Row("s.md", "sessions/2026/07", 1.0)]
    out = _apply_folder_boost(rows, weights={"sessions": 0.5})
    assert out[0].score == 0.5


def test_prefix_is_boundary_matched_not_string_prefix():
    """'Project' must never match 'Projects' (boundary semantics)."""
    rows = [_Row("p.md", "Projects", 1.0)]
    out = _apply_folder_boost(rows, weights={"Project": 0.1})
    assert out[0].score == 1.0


def test_longest_matching_prefix_wins():
    rows = [_Row("d.md", "sessions/archive/old", 1.0)]
    out = _apply_folder_boost(rows, weights={"sessions": 2.0, "sessions/archive": 0.25})
    assert out[0].score == 0.25


def test_no_match_is_weight_one():
    rows = [_Row("n.md", "notes", 0.8)]
    out = _apply_folder_boost(rows, weights={"sessions": 0.5})
    assert out[0].score == 0.8


def test_result_is_resorted_descending():
    rows = [_Row("s.md", "sessions", 1.0), _Row("c.md", "curated", 0.8)]
    out = _apply_folder_boost(rows, weights={"sessions": 0.5})
    assert [r.path for r in out] == ["c.md", "s.md"]
    assert [r.score for r in out] == [0.8, 0.5]


def test_promoting_weight_lifts_folder():
    rows = [_Row("s.md", "sessions", 1.0), _Row("c.md", "curated", 0.8)]
    out = _apply_folder_boost(rows, weights={"curated": 2.0})
    assert [r.path for r in out] == ["c.md", "s.md"]
    assert out[0].score == 1.6


def test_input_rows_are_not_mutated():
    rows = [_Row("s.md", "sessions", 1.0)]
    _ = _apply_folder_boost(rows, weights={"sessions": 0.5})
    assert rows[0].score == 1.0


def test_negative_score_is_left_untouched():
    """A demoting weight must not promote a negative (cosine) score."""
    rows = [_Row("neg.md", "sessions", -0.4)]
    out = _apply_folder_boost(rows, weights={"sessions": 0.5})
    assert out[0].score == -0.4


def test_zero_score_is_left_untouched():
    rows = [_Row("z.md", "sessions", 0.0)]
    out = _apply_folder_boost(rows, weights={"sessions": 2.0})
    assert out[0].score == 0.0
