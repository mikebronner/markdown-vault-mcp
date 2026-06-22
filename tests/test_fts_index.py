"""Unit tests for FTSIndex (fts_index.py)."""

from __future__ import annotations

import datetime
import json
import sqlite3
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from markdown_vault_mcp.fts_index import FTSIndex, _json_default, should_optimize
from markdown_vault_mcp.types import Chunk, FTSResult, ParsedNote

# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


def make_note(
    path: str = "test.md",
    title: str = "Test",
    frontmatter: dict | None = None,
    chunks: list[Chunk] | None = None,
    content_hash: str = "abc123",
    modified_at: float = 1000.0,
) -> ParsedNote:
    """Create a ParsedNote for testing.

    Args:
        path: Relative document path including ``.md`` extension.
        title: Document title.
        frontmatter: Frontmatter metadata dict. Defaults to ``{}``.
        chunks: List of chunks. Defaults to a single generic chunk.
        content_hash: Hash string stored in the note.
        modified_at: Modification timestamp.

    Returns:
        A fully-populated :class:`ParsedNote` suitable for indexing.
    """
    if chunks is None:
        chunks = [
            Chunk(heading="Test", heading_level=1, content="Test content", start_line=0)
        ]
    return ParsedNote(
        path=path,
        frontmatter=frontmatter or {},
        title=title,
        chunks=chunks,
        content_hash=content_hash,
        modified_at=modified_at,
    )


# ---------------------------------------------------------------------------
# Helpers for building a tagged index in multiple tests
# ---------------------------------------------------------------------------

_INDEXED_FIELDS = ["cluster", "topics", "genre"]


def _tagged_index() -> FTSIndex:
    """Return a fresh in-memory index with the standard indexed fields."""
    return FTSIndex(":memory:", indexed_frontmatter_fields=_INDEXED_FIELDS)


# ===========================================================================
# Tests
# ===========================================================================


class TestBuildFromNotes:
    def test_build_from_notes_returns_total_chunk_count(self) -> None:
        """build_from_notes returns the total number of chunks indexed."""
        idx = FTSIndex(":memory:")
        notes = [
            make_note(
                "a.md",
                chunks=[
                    Chunk(heading="H1", heading_level=1, content="alpha", start_line=0),
                    Chunk(heading="H2", heading_level=2, content="beta", start_line=5),
                ],
            ),
            make_note(
                "b.md",
                chunks=[
                    Chunk(heading="B1", heading_level=1, content="gamma", start_line=0),
                ],
            ),
            make_note(
                "c.md",
                chunks=[
                    Chunk(heading="C1", heading_level=1, content="delta", start_line=0),
                    Chunk(
                        heading="C2", heading_level=2, content="epsilon", start_line=3
                    ),
                    Chunk(heading="C3", heading_level=2, content="zeta", start_line=6),
                ],
            ),
        ]
        total = idx.build_from_notes(notes)
        assert total == 6


class TestSearch:
    def test_search_returns_fts_results(self) -> None:
        """search() returns FTSResult objects for matching terms."""
        idx = FTSIndex(":memory:")
        idx.upsert_note(
            make_note(
                "dragons.md",
                title="Dragons",
                chunks=[
                    Chunk(
                        heading="Overview",
                        heading_level=1,
                        content="Dragons breathe fire and hoard treasure.",
                        start_line=0,
                    )
                ],
            )
        )
        results = idx.search("dragons")
        assert len(results) >= 1
        assert all(isinstance(r, FTSResult) for r in results)
        paths = {r.path for r in results}
        assert "dragons.md" in paths

    def test_search_bm25_ranking_orders_by_relevance(self) -> None:
        """More-relevant documents score higher than less-relevant ones."""
        idx = FTSIndex(":memory:")
        # "python" appears many times in high.md, once in low.md
        idx.upsert_note(
            make_note(
                "high.md",
                title="High relevance",
                chunks=[
                    Chunk(
                        heading=None,
                        heading_level=0,
                        content="python python python python python programming",
                        start_line=0,
                    )
                ],
            )
        )
        idx.upsert_note(
            make_note(
                "low.md",
                title="Low relevance",
                chunks=[
                    Chunk(
                        heading=None,
                        heading_level=0,
                        content="python is mentioned once here among other words",
                        start_line=0,
                    )
                ],
            )
        )
        results = idx.search("python", limit=10)
        assert len(results) == 2
        high_result = next(r for r in results if r.path == "high.md")
        low_result = next(r for r in results if r.path == "low.md")
        assert high_result.score > low_result.score

    def test_search_with_folder_filter(self) -> None:
        """folder= filter returns only documents under that folder."""
        idx = FTSIndex(":memory:")
        idx.upsert_note(
            make_note(
                "Journal/2024-01.md",
                title="January",
                chunks=[
                    Chunk(
                        heading=None,
                        heading_level=0,
                        content="today I went for a walk",
                        start_line=0,
                    )
                ],
            )
        )
        idx.upsert_note(
            make_note(
                "Projects/alpha.md",
                title="Alpha",
                chunks=[
                    Chunk(
                        heading=None,
                        heading_level=0,
                        content="today the project started",
                        start_line=0,
                    )
                ],
            )
        )
        results = idx.search("today", folder="Journal")
        assert len(results) == 1
        assert results[0].path == "Journal/2024-01.md"
        assert results[0].folder == "Journal"

    def test_search_with_tag_filters(self) -> None:
        """filters= restricts results to documents matching the tag pair."""
        idx = _tagged_index()
        idx.upsert_note(
            make_note(
                "fiction/story.md",
                title="Story",
                frontmatter={"cluster": "fiction"},
                chunks=[
                    Chunk(
                        heading=None,
                        heading_level=0,
                        content="once upon a time",
                        start_line=0,
                    )
                ],
            )
        )
        idx.upsert_note(
            make_note(
                "nonfiction/essay.md",
                title="Essay",
                frontmatter={"cluster": "nonfiction"},
                chunks=[
                    Chunk(
                        heading=None,
                        heading_level=0,
                        content="once upon a time there were facts",
                        start_line=0,
                    )
                ],
            )
        )
        results = idx.search("once", filters={"cluster": "fiction"})
        assert len(results) == 1
        assert results[0].path == "fiction/story.md"

    def test_search_multiple_filters_anded(self) -> None:
        """Multiple filter entries are ANDed — only docs matching ALL pass."""
        idx = _tagged_index()
        # Matches cluster=fiction but not genre=horror
        idx.upsert_note(
            make_note(
                "a.md",
                frontmatter={"cluster": "fiction", "genre": "romance"},
                chunks=[
                    Chunk(
                        heading=None,
                        heading_level=0,
                        content="love story",
                        start_line=0,
                    )
                ],
            )
        )
        # Matches both cluster=fiction AND genre=horror
        idx.upsert_note(
            make_note(
                "b.md",
                frontmatter={"cluster": "fiction", "genre": "horror"},
                chunks=[
                    Chunk(
                        heading=None,
                        heading_level=0,
                        content="scary story",
                        start_line=0,
                    )
                ],
            )
        )
        # Matches genre=horror but not cluster=fiction
        idx.upsert_note(
            make_note(
                "c.md",
                frontmatter={"cluster": "nonfiction", "genre": "horror"},
                chunks=[
                    Chunk(
                        heading=None,
                        heading_level=0,
                        content="true horror story",
                        start_line=0,
                    )
                ],
            )
        )
        results = idx.search("story", filters={"cluster": "fiction", "genre": "horror"})
        assert len(results) == 1
        assert results[0].path == "b.md"

    def test_search_empty_query_returns_empty_results(self) -> None:
        """search() returns an empty list for an empty query string (no exception)."""
        idx = FTSIndex(":memory:")
        idx.upsert_note(
            make_note(
                "a.md",
                chunks=[
                    Chunk(
                        heading=None,
                        heading_level=0,
                        content="hello world",
                        start_line=0,
                    )
                ],
            )
        )
        results = idx.search("")
        assert results == []

    def test_search_malformed_fts5_syntax_returns_empty_results(self) -> None:
        """search() returns an empty list for malformed FTS5 syntax (no exception)."""
        idx = FTSIndex(":memory:")
        idx.upsert_note(
            make_note(
                "a.md",
                chunks=[
                    Chunk(
                        heading=None,
                        heading_level=0,
                        content="hello world",
                        start_line=0,
                    )
                ],
            )
        )
        # Unclosed quote is invalid FTS5 syntax
        results = idx.search('"unclosed quote')
        assert results == []

    def test_search_invalid_fts5_column_returns_empty_results(self) -> None:
        """search() returns an empty list for an invalid FTS5 column reference."""
        idx = FTSIndex(":memory:")
        idx.upsert_note(
            make_note(
                "a.md",
                chunks=[
                    Chunk(
                        heading=None,
                        heading_level=0,
                        content="hello world",
                        start_line=0,
                    )
                ],
            )
        )
        # FTS5 column filters for non-existent columns raise OperationalError
        results = idx.search("nonexistent_column:value")
        assert results == []

    def test_search_non_fts5_operational_error_propagates(self) -> None:
        """Non-FTS5 OperationalError (e.g. DB lock) must propagate, not return []."""
        from unittest.mock import MagicMock

        idx = FTSIndex(":memory:")
        # Replace the real connection with a mock that raises a non-FTS5 DB error.
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.OperationalError("database is locked")
        idx._conn = mock_conn

        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            idx.search("hello")


class TestUpsert:
    def test_upsert_note_replaces_existing(self) -> None:
        """upsert_note removes old content and makes new content searchable."""
        idx = FTSIndex(":memory:")
        idx.upsert_note(
            make_note(
                "replace.md",
                chunks=[
                    Chunk(
                        heading=None,
                        heading_level=0,
                        content="the old unique word xylophone",
                        start_line=0,
                    )
                ],
                content_hash="old",
            )
        )
        # Sanity: old content is searchable before upsert.
        assert len(idx.search("xylophone")) == 1

        idx.upsert_note(
            make_note(
                "replace.md",
                chunks=[
                    Chunk(
                        heading=None,
                        heading_level=0,
                        content="entirely new content kazoo",
                        start_line=0,
                    )
                ],
                content_hash="new",
            )
        )
        assert idx.search("xylophone") == []
        results = idx.search("kazoo")
        assert len(results) == 1
        assert results[0].path == "replace.md"


class TestFrontmatterSerialization:
    def test_date_in_frontmatter_stored_as_iso_string(self) -> None:
        """datetime.date in frontmatter is stored as ISO string."""
        idx = FTSIndex(":memory:")
        note = make_note(
            "dated.md",
            frontmatter={"created": datetime.date(2024, 1, 15), "title": "Dated"},
        )
        idx.upsert_note(note)
        row = idx.get_note("dated.md")
        assert row is not None
        fm = json.loads(row["frontmatter_json"])
        assert fm["created"] == "2024-01-15"

    def test_datetime_in_frontmatter_stored_as_iso_string(self) -> None:
        """datetime.datetime in frontmatter is stored as ISO string."""
        idx = FTSIndex(":memory:")
        note = make_note(
            "timestamped.md",
            frontmatter={"updated": datetime.datetime(2024, 6, 15, 12, 30, 0)},
        )
        idx.upsert_note(note)
        row = idx.get_note("timestamped.md")
        assert row is not None
        fm = json.loads(row["frontmatter_json"])
        assert fm["updated"] == "2024-06-15T12:30:00"

    def test_time_in_frontmatter_stored_as_iso_string(self) -> None:
        """datetime.time in frontmatter is stored as ISO string."""
        idx = FTSIndex(":memory:")
        note = make_note(
            "timed.md",
            frontmatter={"starts_at": datetime.time(15, 30, 0)},
        )
        idx.upsert_note(note)
        row = idx.get_note("timed.md")
        assert row is not None
        fm = json.loads(row["frontmatter_json"])
        assert fm["starts_at"] == "15:30:00"

    def test_json_default_raises_for_unsupported_types(self) -> None:
        """_json_default raises TypeError for non-date types."""
        with pytest.raises(TypeError, match="set"):
            _json_default({1, 2, 3})


class TestDelete:
    def test_delete_by_path_removes_search_results(self) -> None:
        """delete_by_path makes the note unsearchable."""
        idx = FTSIndex(":memory:")
        idx.upsert_note(
            make_note(
                "gone.md",
                chunks=[
                    Chunk(
                        heading=None,
                        heading_level=0,
                        content="the unique word fjord",
                        start_line=0,
                    )
                ],
            )
        )
        assert len(idx.search("fjord")) == 1

        deleted = idx.delete_by_path("gone.md")
        assert deleted == 1
        assert idx.search("fjord") == []

    def test_delete_cascades_to_sections_and_tags(self) -> None:
        """Deleting a document removes its sections and tags from the DB."""
        idx = _tagged_index()
        idx.upsert_note(
            make_note(
                "cascade.md",
                frontmatter={"cluster": "fiction"},
                chunks=[
                    Chunk(
                        heading="Ch1", heading_level=1, content="content", start_line=0
                    ),
                    Chunk(heading="Ch2", heading_level=2, content="more", start_line=5),
                ],
            )
        )

        # Verify sections and tags exist before deletion.
        conn = idx._conn
        sec_count = conn.execute(
            "SELECT COUNT(*) FROM sections WHERE document_id IN "
            "(SELECT id FROM documents WHERE path = ?)",
            ("cascade.md",),
        ).fetchone()[0]
        assert sec_count == 2

        tag_count = conn.execute(
            "SELECT COUNT(*) FROM document_tags WHERE document_id IN "
            "(SELECT id FROM documents WHERE path = ?)",
            ("cascade.md",),
        ).fetchone()[0]
        assert tag_count == 1

        idx.delete_by_path("cascade.md")

        # Documents row is gone — CASCADE should have cleared child rows.
        doc_count = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE path = ?", ("cascade.md",)
        ).fetchone()[0]
        assert doc_count == 0

        # Sections and tags for the deleted document must also be gone.
        orphan_secs = conn.execute(
            "SELECT COUNT(*) FROM sections WHERE document_id NOT IN "
            "(SELECT id FROM documents)"
        ).fetchone()[0]
        assert orphan_secs == 0

        orphan_tags = conn.execute(
            "SELECT COUNT(*) FROM document_tags WHERE document_id NOT IN "
            "(SELECT id FROM documents)"
        ).fetchone()[0]
        assert orphan_tags == 0


class TestListFolders:
    def test_list_folders_returns_sorted_distinct_values(self) -> None:
        """list_folders() returns all distinct folder values in sorted order."""
        idx = FTSIndex(":memory:")
        idx.upsert_note(make_note("Journal/jan.md"))
        idx.upsert_note(make_note("Journal/feb.md"))
        idx.upsert_note(make_note("Projects/alpha.md"))
        idx.upsert_note(make_note("root.md"))

        folders = idx.list_folders()
        assert folders == sorted(set(folders))
        assert "Journal" in folders
        assert "Projects" in folders
        assert "" in folders  # root document
        # No duplicates.
        assert len(folders) == len(set(folders))


class TestListFieldValues:
    def test_list_field_values_returns_distinct_values(self) -> None:
        """list_field_values() returns distinct tag values for a field."""
        idx = _tagged_index()
        idx.upsert_note(make_note("a.md", frontmatter={"cluster": "fiction"}))
        idx.upsert_note(make_note("b.md", frontmatter={"cluster": "nonfiction"}))
        idx.upsert_note(make_note("c.md", frontmatter={"cluster": "fiction"}))

        values = idx.list_field_values("cluster")
        assert sorted(values) == ["fiction", "nonfiction"]
        # No duplicates.
        assert len(values) == len(set(values))


class TestTagIndexing:
    def test_tag_indexing_scalar_creates_one_row(self) -> None:
        """A scalar frontmatter value produces exactly one document_tags row."""
        idx = _tagged_index()
        idx.upsert_note(make_note("scalar.md", frontmatter={"cluster": "fiction"}))
        rows = idx._conn.execute(
            "SELECT tag_value FROM document_tags WHERE tag_key = 'cluster'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "fiction"

    def test_tag_indexing_list_deduplicates(self) -> None:
        """A list frontmatter value creates one row per distinct item."""
        idx = _tagged_index()
        idx.upsert_note(make_note("list.md", frontmatter={"topics": ["a", "b", "a"]}))
        rows = idx._conn.execute(
            "SELECT tag_value FROM document_tags WHERE tag_key = 'topics' "
            "ORDER BY tag_value"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0][0] == "a"
        assert rows[1][0] == "b"

    def test_tag_indexing_complex_value_skipped(self) -> None:
        """A nested dict frontmatter value is NOT promoted to document_tags."""
        idx = _tagged_index()
        idx.upsert_note(
            make_note("complex.md", frontmatter={"cluster": {"key": "val"}})
        )
        rows = idx._conn.execute(
            "SELECT COUNT(*) FROM document_tags WHERE tag_key = 'cluster'"
        ).fetchone()
        assert rows[0] == 0


class TestGetNote:
    def test_get_note_returns_correct_dict(self) -> None:
        """get_note() returns a dict with the expected keys and values."""
        idx = FTSIndex(":memory:")
        note = make_note(
            "Journal/entry.md",
            title="My Entry",
            frontmatter={"date": "2024-01-01"},
            content_hash="deadbeef",
            modified_at=9999.0,
        )
        idx.upsert_note(note)

        result = idx.get_note("Journal/entry.md")
        assert result is not None
        assert result["path"] == "Journal/entry.md"
        assert result["title"] == "My Entry"
        assert result["folder"] == "Journal"
        assert result["content_hash"] == "deadbeef"
        assert result["modified_at"] == pytest.approx(9999.0)

    def test_get_note_not_found_returns_none(self) -> None:
        """get_note() returns None for a path that was never indexed."""
        idx = FTSIndex(":memory:")
        assert idx.get_note("nonexistent.md") is None


class TestListChunks:
    def test_list_chunks_returns_all_chunks_with_document_metadata(self) -> None:
        """list_chunks() joins sections with document path/title/folder."""
        idx = FTSIndex(":memory:")
        idx.upsert_note(
            make_note(
                "Journal/entry.md",
                title="My Entry",
                chunks=[
                    Chunk(heading=None, heading_level=0, content="intro", start_line=0),
                    Chunk(
                        heading="Details",
                        heading_level=2,
                        content="the details",
                        start_line=5,
                    ),
                ],
            )
        )
        idx.upsert_note(make_note("alpha.md", title="Alpha"))

        rows = idx.list_chunks()

        assert len(rows) == 3
        # Ordered by path, then chunk position.
        assert [r["path"] for r in rows] == [
            "Journal/entry.md",
            "Journal/entry.md",
            "alpha.md",
        ]
        first = rows[0]
        assert set(first) == {"path", "title", "folder", "heading", "content"}
        assert first["title"] == "My Entry"
        assert first["folder"] == "Journal"
        assert first["heading"] is None
        assert first["content"] == "intro"
        assert rows[1]["heading"] == "Details"
        assert rows[1]["content"] == "the details"

    def test_list_chunks_empty_index_returns_empty_list(self) -> None:
        """list_chunks() on an empty index returns []."""
        idx = FTSIndex(":memory:")
        assert idx.list_chunks() == []


class TestInMemoryMode:
    def test_in_memory_mode_works(self) -> None:
        """FTSIndex with ':memory:' is functional end-to-end."""
        idx = FTSIndex(":memory:")
        idx.upsert_note(
            make_note(
                "mem.md",
                chunks=[
                    Chunk(
                        heading=None,
                        heading_level=0,
                        content="in-memory test passage",
                        start_line=0,
                    )
                ],
            )
        )
        results = idx.search("memory")
        assert len(results) >= 1
        assert results[0].path == "mem.md"


class TestWALMode:
    def test_file_based_index_uses_wal_journal_mode(self, tmp_path: Path) -> None:
        """File-based FTSIndex uses WAL journal mode for concurrent reads."""
        db_path = tmp_path / "test.db"
        idx = FTSIndex(str(db_path))
        mode = idx._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"

    def test_in_memory_index_uses_memory_journal_mode(self) -> None:
        """In-memory FTSIndex skips WAL and retains SQLite default 'memory' mode."""
        idx = FTSIndex(":memory:")
        mode = idx._conn.execute("PRAGMA journal_mode").fetchone()[0]
        # WAL pragma is skipped for :memory: databases; SQLite uses 'memory' mode.
        assert mode.lower() == "memory"

    def test_wal_warning_logged_when_pragma_returns_non_wal(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A warning is logged when WAL mode cannot be enabled."""
        import logging
        from unittest.mock import MagicMock

        from markdown_vault_mcp.fts_index import _open_connection

        db_path = tmp_path / "nowarn.db"

        mock_conn = MagicMock()
        # WAL pragma returns "delete" (simulating a filesystem without WAL support).
        mock_conn.execute.return_value.fetchone.return_value = ["delete"]

        with (
            patch(
                "markdown_vault_mcp.fts_index.sqlite3.connect", return_value=mock_conn
            ),
            caplog.at_level(logging.WARNING, logger="markdown_vault_mcp.fts_index"),
        ):
            _open_connection(db_path)

        assert any(
            "Could not enable WAL journal mode" in r.message for r in caplog.records
        )

    def test_wal_allows_concurrent_reader_during_write(self, tmp_path: Path) -> None:
        """A reader on a second connection succeeds while the first connection writes."""
        import sqlite3

        db_path = tmp_path / "concurrent.db"
        idx = FTSIndex(str(db_path))
        # Seed one document so there is something to read.
        idx.upsert_note(make_note(path="seed.md"))

        writer_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        reader_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        try:
            # Begin an exclusive write transaction on writer_conn.
            writer_conn.execute("BEGIN EXCLUSIVE")
            writer_conn.execute(
                "INSERT OR REPLACE INTO documents(path, title, folder, "
                "frontmatter_json, content_hash, modified_at) "
                "VALUES ('concurrent.md', 'Concurrent', '', '{}', 'abc', 0.0)"
            )
            # WAL allows the reader to see the previously committed state
            # without waiting for the writer to commit.
            rows = reader_conn.execute(
                "SELECT path FROM documents WHERE path = 'seed.md'"
            ).fetchall()
            assert len(rows) == 1, (
                "Reader should see committed data while writer holds EXCLUSIVE"
            )
            writer_conn.rollback()
        finally:
            writer_conn.close()
            reader_conn.close()


class TestGetRecent:
    """Tests for FTSIndex.get_recent()."""

    def test_returns_notes_ordered_by_mtime_desc(self) -> None:
        """get_recent returns notes most-recent first."""
        idx = FTSIndex(":memory:")
        notes = [
            ParsedNote(
                path=f"note{i}.md",
                frontmatter={},
                title=f"Note {i}",
                chunks=[
                    Chunk(heading=None, heading_level=0, content="c", start_line=0)
                ],
                content_hash=f"h{i}",
                modified_at=float(i * 100),
            )
            for i in range(5)
        ]
        idx.build_from_notes(notes)
        rows = idx.get_recent(limit=5)
        mtimes = [r["modified_at"] for r in rows]
        assert mtimes == sorted(mtimes, reverse=True)

    def test_respects_limit(self) -> None:
        """get_recent returns at most `limit` rows."""
        idx = FTSIndex(":memory:")
        notes = [
            ParsedNote(
                path=f"note{i}.md",
                frontmatter={},
                title=f"Note {i}",
                chunks=[
                    Chunk(heading=None, heading_level=0, content="c", start_line=0)
                ],
                content_hash=f"h{i}",
                modified_at=float(i),
            )
            for i in range(10)
        ]
        idx.build_from_notes(notes)
        rows = idx.get_recent(limit=3)
        assert len(rows) == 3

    def test_folder_filter(self) -> None:
        """get_recent with folder returns only matching documents."""
        idx = FTSIndex(":memory:")
        notes = [
            ParsedNote(
                path="root.md",
                frontmatter={},
                title="Root",
                chunks=[
                    Chunk(heading=None, heading_level=0, content="c", start_line=0)
                ],
                content_hash="h1",
                modified_at=100.0,
            ),
            ParsedNote(
                path="Journal/day1.md",
                frontmatter={},
                title="Day 1",
                chunks=[
                    Chunk(heading=None, heading_level=0, content="c", start_line=0)
                ],
                content_hash="h2",
                modified_at=200.0,
            ),
            ParsedNote(
                path="Journal/day2.md",
                frontmatter={},
                title="Day 2",
                chunks=[
                    Chunk(heading=None, heading_level=0, content="c", start_line=0)
                ],
                content_hash="h3",
                modified_at=300.0,
            ),
        ]
        idx.build_from_notes(notes)
        rows = idx.get_recent(folder="Journal")
        paths = {r["path"] for r in rows}
        assert paths == {"Journal/day1.md", "Journal/day2.md"}

    def test_folder_filter_nested_subfolder(self) -> None:
        """get_recent with folder includes nested sub-folder documents."""
        idx = FTSIndex(":memory:")
        notes = [
            ParsedNote(
                path="Journal/day1.md",
                frontmatter={},
                title="Day 1",
                chunks=[
                    Chunk(heading=None, heading_level=0, content="c", start_line=0)
                ],
                content_hash="h1",
                modified_at=100.0,
            ),
            ParsedNote(
                path="Journal/sub/nested.md",
                frontmatter={},
                title="Nested",
                chunks=[
                    Chunk(heading=None, heading_level=0, content="c", start_line=0)
                ],
                content_hash="h2",
                modified_at=200.0,
            ),
            ParsedNote(
                path="Other/note.md",
                frontmatter={},
                title="Other",
                chunks=[
                    Chunk(heading=None, heading_level=0, content="c", start_line=0)
                ],
                content_hash="h3",
                modified_at=300.0,
            ),
        ]
        idx.build_from_notes(notes)
        rows = idx.get_recent(folder="Journal")
        paths = {r["path"] for r in rows}
        assert paths == {"Journal/day1.md", "Journal/sub/nested.md"}

    def test_empty_index_returns_empty(self) -> None:
        """get_recent on empty index returns []."""
        idx = FTSIndex(":memory:")
        assert idx.get_recent() == []


class TestShouldOptimize:
    def test_purge_at_absolute_threshold_triggers(self) -> None:
        """Purging OPTIMIZE_MIN_PURGED_DOCS documents qualifies."""
        assert should_optimize(25, 1000) is True

    def test_purge_below_both_thresholds_does_not_trigger(self) -> None:
        """A small purge of a large corpus does not qualify."""
        assert should_optimize(24, 1000) is False

    def test_purge_at_fractional_threshold_triggers(self) -> None:
        """Purging >= 10% of a small corpus qualifies."""
        assert should_optimize(3, 20) is True  # 15% of corpus.

    def test_purge_below_fractional_threshold_does_not_trigger(self) -> None:
        """Purging < 10% of a small corpus does not qualify."""
        assert should_optimize(1, 20) is False  # 5% of corpus.

    def test_zero_purged_never_triggers(self) -> None:
        """No purge, no optimize."""
        assert should_optimize(0, 100) is False

    def test_empty_corpus_never_triggers(self) -> None:
        """Guard against division by zero on an empty corpus."""
        assert should_optimize(5, 0) is False


def _dbstat_fts_data_size(db_path: Path) -> int | None:
    """Return SUM(pgsize) of the notes_fts_data shadow table via dbstat.

    Returns ``None`` when this SQLite build lacks the dbstat virtual table,
    so callers can skip size assertions in environments without it.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT SUM(pgsize) FROM dbstat WHERE name = 'notes_fts_data'"
        ).fetchone()
        return int(row[0] or 0)
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


class _FailingConnection:
    """Stand-in connection whose every execute raises OperationalError.

    ``sqlite3.Connection.execute`` cannot be patched on an instance (the
    attribute is read-only), so lock-contention tests swap the whole
    connection for this stub instead.
    """

    def __init__(self, message: str) -> None:
        self._message = message

    def __enter__(self) -> _FailingConnection:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def execute(self, *_args: object, **_kwargs: object) -> None:
        raise sqlite3.OperationalError(self._message)

    def executescript(self, *_args: object, **_kwargs: object) -> None:
        raise sqlite3.OperationalError(self._message)

    def commit(self) -> None:
        """No-op; vacuum() commits before executing."""


class _MigrationStubConnection:
    """Stand-in connection emulating a populated legacy (auto_vacuum=NONE) file.

    ``PRAGMA auto_vacuum`` always reports ``0`` (NONE) — even after
    ``PRAGMA auto_vacuum = INCREMENTAL`` — which is exactly how a *populated*
    database behaves: the mode only flips once a ``VACUUM`` rewrites the file.
    The migration ``VACUUM`` raises a configurable :class:`OperationalError` so
    the lock-tolerance and error-propagation branches of
    :func:`_apply_auto_vacuum` can be exercised without a real on-disk database
    (``sqlite3.Connection.execute`` is immutable and cannot be patched).
    """

    def __init__(self, locked: bool) -> None:
        self._message = "database is locked" if locked else "disk I/O error"
        self.vacuum_attempted = False

    def execute(self, sql: str, *_args: object) -> object:
        normalized = sql.strip().upper()
        if normalized == "VACUUM":
            self.vacuum_attempted = True
            raise sqlite3.OperationalError(self._message)
        if normalized.startswith("PRAGMA AUTO_VACUUM"):
            # Always NONE: a populated DB never flips via the pragma alone.
            return _StubCursor(0)
        return _StubCursor(None)

    def commit(self) -> None:
        """No-op; the helper commits the pragma before attempting VACUUM."""


class _StubCursor:
    """Minimal cursor exposing ``fetchone`` for PRAGMA reads."""

    def __init__(self, value: object) -> None:
        self._value = value

    def fetchone(self) -> tuple[object]:
        return (self._value,)


class TestOptimize:
    def test_optimize_runs_and_returns_true(self) -> None:
        """optimize() executes the FTS5 optimize command and reports success."""
        idx = FTSIndex(":memory:")
        idx.build_from_notes([make_note("a.md"), make_note("b.md")])
        idx.delete_by_path("a.md")

        assert idx.optimize() is True

    def test_optimize_preserves_search_results(self) -> None:
        """Surviving documents remain searchable after optimize()."""
        idx = FTSIndex(":memory:")
        idx.build_from_notes(
            [
                make_note(
                    "keep.md",
                    chunks=[
                        Chunk(
                            heading="K",
                            heading_level=1,
                            content="zanzibar survives",
                            start_line=0,
                        )
                    ],
                ),
                make_note("drop.md"),
            ]
        )
        idx.delete_by_path("drop.md")
        idx.optimize()

        results = idx.search("zanzibar")
        assert [r.path for r in results] == ["keep.md"]

    def test_optimize_shrinks_dead_segments(self, tmp_path: Path) -> None:
        """After a bulk delete, optimize() shrinks the FTS5 segment b-trees.

        Measured via the dbstat virtual table; the size assertion is skipped
        when this SQLite build does not provide dbstat.
        """
        db_path = tmp_path / "index.db"
        idx = FTSIndex(db_path)
        # Index enough distinct-token content for measurable segments.
        notes = [
            make_note(
                f"doc{i}.md",
                chunks=[
                    Chunk(
                        heading=f"Heading {i}",
                        heading_level=1,
                        content=" ".join(f"token{i}word{j}" for j in range(200)),
                        start_line=0,
                    )
                ],
                content_hash=f"hash{i}",
            )
            for i in range(40)
        ]
        idx.build_from_notes(notes)
        idx._conn.commit()

        for i in range(40):
            idx.delete_by_path(f"doc{i}.md")
        idx._conn.commit()
        size_before = _dbstat_fts_data_size(db_path)

        assert idx.optimize() is True
        idx._conn.commit()
        size_after = _dbstat_fts_data_size(db_path)
        idx.close()

        if size_before is None or size_after is None:
            pytest.skip("dbstat virtual table not available in this SQLite build")
        assert size_after < size_before

    def test_optimize_tolerates_locked_database(self) -> None:
        """A busy/locked database skips the optimize instead of raising."""
        idx = FTSIndex(":memory:")
        idx.build_from_notes([make_note("a.md")])

        idx._conn = _FailingConnection("database is locked")  # type: ignore[assignment]
        assert idx.optimize() is False

    def test_optimize_propagates_other_operational_errors(self) -> None:
        """Non-lock OperationalErrors are not swallowed."""
        idx = FTSIndex(":memory:")

        idx._conn = _FailingConnection("disk I/O error")  # type: ignore[assignment]
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            idx.optimize()


class TestVacuum:
    def test_vacuum_runs_and_returns_true(self, tmp_path: Path) -> None:
        """vacuum() compacts a file-backed index and reports success."""
        idx = FTSIndex(tmp_path / "index.db")
        idx.build_from_notes([make_note("a.md")])
        idx.delete_by_path("a.md")

        assert idx.vacuum() is True

    def test_vacuum_tolerates_locked_database(self) -> None:
        """A busy/locked database skips the vacuum instead of raising."""
        idx = FTSIndex(":memory:")

        idx._conn = _FailingConnection("database is busy")  # type: ignore[assignment]
        assert idx.vacuum() is False

    def test_vacuum_propagates_other_operational_errors(self) -> None:
        """Non-lock OperationalErrors are not swallowed."""
        idx = FTSIndex(":memory:")

        idx._conn = _FailingConnection("disk I/O error")  # type: ignore[assignment]
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            idx.vacuum()


# auto_vacuum mode codes: 0 == NONE, 1 == FULL, 2 == INCREMENTAL.
_AUTO_VACUUM_INCREMENTAL = 2
_AUTO_VACUUM_NONE = 0


def _auto_vacuum_mode(db_path: Path) -> int:
    """Return the persisted ``auto_vacuum`` mode of a file-backed database."""
    conn = sqlite3.connect(str(db_path))
    try:
        return int(conn.execute("PRAGMA auto_vacuum").fetchone()[0])
    finally:
        conn.close()


class TestAutoVacuum:
    """Tests for incremental auto_vacuum setup and legacy-database migration."""

    def test_fresh_file_index_uses_incremental_auto_vacuum(
        self, tmp_path: Path
    ) -> None:
        """A freshly created file-backed index reports auto_vacuum=INCREMENTAL."""
        db_path = tmp_path / "fresh.db"
        idx = FTSIndex(db_path)
        mode = idx._conn.execute("PRAGMA auto_vacuum").fetchone()[0]
        assert mode == _AUTO_VACUUM_INCREMENTAL

    def test_auto_vacuum_persists_to_disk(self, tmp_path: Path) -> None:
        """The INCREMENTAL mode persists in the file, not just the connection."""
        db_path = tmp_path / "persist.db"
        idx = FTSIndex(db_path)
        idx.build_from_notes([make_note("a.md")])
        idx.close()

        assert _auto_vacuum_mode(db_path) == _AUTO_VACUUM_INCREMENTAL

    def test_in_memory_index_skips_auto_vacuum(self) -> None:
        """In-memory databases do not get auto_vacuum (unsupported)."""
        idx = FTSIndex(":memory:")
        mode = idx._conn.execute("PRAGMA auto_vacuum").fetchone()[0]
        assert mode == _AUTO_VACUUM_NONE

    def test_legacy_none_database_is_migrated_on_open(self, tmp_path: Path) -> None:
        """A populated DB created with auto_vacuum=NONE converts on next open."""
        from markdown_vault_mcp.fts_index import _SCHEMA_SQL

        db_path = tmp_path / "legacy.db"

        # Build a populated database with the real schema but the default
        # auto_vacuum (NONE), simulating an index created before this fix.
        legacy = sqlite3.connect(str(db_path))
        legacy.execute("PRAGMA journal_mode = WAL")
        legacy.executescript(_SCHEMA_SQL)
        legacy.executemany(
            "INSERT INTO documents"
            "(path, title, content_hash, modified_at) VALUES (?, ?, ?, ?)",
            [(f"doc{i}.md", f"Doc {i}", f"hash{i}", 0.0) for i in range(100)],
        )
        legacy.commit()
        legacy.close()
        assert _auto_vacuum_mode(db_path) == _AUTO_VACUUM_NONE

        # Opening through FTSIndex must convert the file in place.
        idx = FTSIndex(db_path)
        assert idx._conn.execute("PRAGMA auto_vacuum").fetchone()[0] == (
            _AUTO_VACUUM_INCREMENTAL
        )
        idx.close()
        assert _auto_vacuum_mode(db_path) == _AUTO_VACUUM_INCREMENTAL

    def test_reopen_of_incremental_database_is_a_noop(self, tmp_path: Path) -> None:
        """Reopening an already-INCREMENTAL index leaves the mode unchanged."""
        db_path = tmp_path / "reopen.db"
        first = FTSIndex(db_path)
        first.build_from_notes([make_note("a.md")])
        first.close()

        # Second open hits the "already INCREMENTAL" fast path; no migration.
        second = FTSIndex(db_path)
        assert second._conn.execute("PRAGMA auto_vacuum").fetchone()[0] == (
            _AUTO_VACUUM_INCREMENTAL
        )
        second.close()

    def test_existing_full_auto_vacuum_is_left_untouched(self, tmp_path: Path) -> None:
        """A database already using auto_vacuum=FULL is not downgraded."""
        from markdown_vault_mcp.fts_index import _SCHEMA_SQL

        db_path = tmp_path / "full.db"
        legacy = sqlite3.connect(str(db_path))
        legacy.execute("PRAGMA auto_vacuum = FULL")
        legacy.execute("PRAGMA journal_mode = WAL")
        legacy.executescript(_SCHEMA_SQL)
        legacy.commit()
        legacy.close()
        assert _auto_vacuum_mode(db_path) == 1  # 1 == FULL.

        idx = FTSIndex(db_path)
        # FULL already reclaims pages on commit; leave it as-is.
        assert idx._conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 1
        idx.close()
        assert _auto_vacuum_mode(db_path) == 1

    def test_migration_skips_when_database_locked(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A locked legacy database skips migration and logs instead of raising.

        Exercises :func:`_apply_auto_vacuum` with a connection stub that reports
        auto_vacuum=NONE (a populated legacy file) and raises "locked" on the
        migration ``VACUUM`` — the migration must be skipped and retried on a
        later boot rather than crashing the open.
        """
        import logging

        from markdown_vault_mcp.fts_index import _apply_auto_vacuum

        conn = _MigrationStubConnection(locked=True)
        with caplog.at_level(logging.WARNING, logger="markdown_vault_mcp.fts_index"):
            # Must not raise — contention is tolerated.
            _apply_auto_vacuum(conn, "legacy.db")  # type: ignore[arg-type]

        assert conn.vacuum_attempted
        assert any(
            "fts_auto_vacuum_migration_skipped" in r.message for r in caplog.records
        )

    def test_migration_propagates_non_lock_errors(self) -> None:
        """A non-lock OperationalError from the migration VACUUM is not swallowed."""
        from markdown_vault_mcp.fts_index import _apply_auto_vacuum

        conn = _MigrationStubConnection(locked=False)
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            _apply_auto_vacuum(conn, "legacy.db")  # type: ignore[arg-type]


class TestIncrementalVacuum:
    """Tests for incremental_vacuum following optimize()."""

    def test_optimize_shrinks_file_via_incremental_vacuum(self, tmp_path: Path) -> None:
        """optimize() returns freed pages to the OS, shrinking the file.

        With auto_vacuum=INCREMENTAL the post-optimize incremental vacuum
        reduces page_count and the on-disk file size after a bulk delete.
        """
        db_path = tmp_path / "shrink.db"
        idx = FTSIndex(db_path)
        notes = [
            make_note(
                f"doc{i}.md",
                chunks=[
                    Chunk(
                        heading=f"Heading {i}",
                        heading_level=1,
                        content=" ".join(f"token{i}word{j}" for j in range(200)),
                        start_line=0,
                    )
                ],
                content_hash=f"hash{i}",
            )
            for i in range(60)
        ]
        idx.build_from_notes(notes)
        idx._conn.commit()

        for i in range(60):
            idx.delete_by_path(f"doc{i}.md")
        idx._conn.commit()
        pages_before = idx._conn.execute("PRAGMA page_count").fetchone()[0]

        assert idx.optimize() is True
        pages_after = idx._conn.execute("PRAGMA page_count").fetchone()[0]
        freelist_after = idx._conn.execute("PRAGMA freelist_count").fetchone()[0]
        idx.close()

        # Incremental vacuum should have collapsed the page count and emptied
        # the freelist (a no-arg PRAGMA incremental_vacuum reclaims everything).
        assert pages_after < pages_before
        assert freelist_after == 0
        assert db_path.stat().st_size < pages_before * 4096

    def test_incremental_vacuum_tolerates_locked_database(self) -> None:
        """A busy/locked database skips the incremental vacuum, returning False."""
        idx = FTSIndex(":memory:")
        idx._conn = _FailingConnection("database is locked")  # type: ignore[assignment]
        assert idx._incremental_vacuum(page_size=4096) is False

    def test_incremental_vacuum_propagates_other_operational_errors(self) -> None:
        """Non-lock OperationalErrors from the vacuum are not swallowed."""
        idx = FTSIndex(":memory:")
        idx._conn = _FailingConnection("disk I/O error")  # type: ignore[assignment]
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            idx._incremental_vacuum(page_size=4096)
