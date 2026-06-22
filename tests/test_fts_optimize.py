"""Tests for FTS5 optimize after bulk purges (dead-segment hygiene).

Deleting rows from an FTS5 table only marks their tokens as deleted; the
dead entries stay in the on-disk segment b-trees until a merge. These tests
cover ``should_optimize`` thresholding, ``FTSIndex.optimize()`` behaviour,
and the bulk-purge call sites in ``IndexManager`` (including the issue #255
exclusion upgrade path).
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from markdown_vault_mcp.fts_index import (
    _SCHEMA_SQL,
    FTSIndex,
    _apply_auto_vacuum,
    should_optimize,
)
from markdown_vault_mcp.managers.index import IndexManager
from markdown_vault_mcp.scanner import HeadingChunker
from markdown_vault_mcp.tracker import ChangeTracker
from markdown_vault_mcp.types import Chunk, ParsedNote

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_note(
    path: str = "test.md",
    title: str = "Test",
    chunks: list[Chunk] | None = None,
    content_hash: str = "abc123",
) -> ParsedNote:
    """Create a ParsedNote for testing.

    Args:
        path: Relative document path including ``.md`` extension.
        title: Document title.
        chunks: List of chunks. Defaults to a single generic chunk.
        content_hash: Hash string stored in the note.

    Returns:
        A fully-populated :class:`ParsedNote` suitable for indexing.
    """
    if chunks is None:
        chunks = [
            Chunk(heading="Test", heading_level=1, content="Test content", start_line=0)
        ]
    return ParsedNote(
        path=path,
        frontmatter={},
        title=title,
        chunks=chunks,
        content_hash=content_hash,
        modified_at=1000.0,
    )


class _FailingConnection:
    """Stand-in connection whose every execute raises OperationalError.

    ``sqlite3.Connection.execute`` cannot be patched on an instance (the
    attribute is read-only), so contention tests swap the whole connection
    for this stub instead.
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


class _StubCursor:
    """Minimal cursor exposing ``fetchone`` for PRAGMA reads."""

    def __init__(self, value: object) -> None:
        self._value = value

    def fetchone(self) -> tuple[object]:
        return (self._value,)


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


def _make_index_mgr(
    vault: Path,
    state_dir: Path,
    **overrides: object,
) -> tuple[IndexManager, FTSIndex]:
    """Build an IndexManager with default wiring.

    Args:
        vault: Source directory containing markdown files.
        state_dir: Directory for the change-tracker state file.
        **overrides: Keyword overrides for the IndexManager constructor;
            ``fts`` may be passed to share an index between managers.

    Returns:
        Tuple of (manager, fts index).
    """
    fts = overrides.pop("fts", None) or FTSIndex(db_path=":memory:")
    vectors_holder: dict = {"vectors": None}
    defaults: dict = {
        "fts": fts,
        "tracker": ChangeTracker(state_dir / ".state" / "state.json"),
        "source_dir": vault,
        "chunk_strategy": HeadingChunker(),
        "get_vectors": lambda: vectors_holder["vectors"],
        "set_vectors": lambda v: vectors_holder.__setitem__("vectors", v),
    }
    defaults.update(overrides)
    return IndexManager(**defaults), fts


def _write_docs(directory: Path, count: int, prefix: str = "doc") -> list[Path]:
    """Write *count* small markdown files into *directory*.

    Args:
        directory: Target directory (created if missing).
        count: Number of files to create.
        prefix: Filename prefix.

    Returns:
        List of created file paths.
    """
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(count):
        f = directory / f"{prefix}{i}.md"
        f.write_text(
            f"# {prefix} {i}\n\nUnique content {prefix}{i} body text.\n",
            encoding="utf-8",
        )
        files.append(f)
    return files


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


# ---------------------------------------------------------------------------
# should_optimize thresholds
# ---------------------------------------------------------------------------


class TestShouldOptimize:
    """Unit tests for the bulk-purge optimize threshold."""

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


# ---------------------------------------------------------------------------
# FTSIndex.optimize()
# ---------------------------------------------------------------------------


class TestOptimize:
    """Unit tests for FTSIndex.optimize()."""

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

        for i in range(40):
            idx.delete_by_path(f"doc{i}.md")
        size_before = _dbstat_fts_data_size(db_path)

        assert idx.optimize() is True
        size_after = _dbstat_fts_data_size(db_path)
        idx.close()

        if size_before is None or size_after is None:
            pytest.skip("dbstat virtual table not available in this SQLite build")
        assert size_after < size_before

    def test_optimize_tolerates_busy_database(self, monkeypatch) -> None:
        """A busy database skips the optimize instead of raising."""
        idx = FTSIndex(":memory:")
        idx.build_from_notes([make_note("a.md")])

        monkeypatch.setattr(
            idx, "_conn", lambda: _FailingConnection("database is busy")
        )
        assert idx.optimize() is False

    def test_optimize_tolerates_locked_past_retry_budget(self, monkeypatch) -> None:
        """A lock held past the retry budget skips the optimize."""
        idx = FTSIndex(":memory:")
        idx.build_from_notes([make_note("a.md")])

        def _exhausted(_operation, **_kwargs):
            raise sqlite3.OperationalError("database table is locked: notes_fts")

        monkeypatch.setattr(
            "markdown_vault_mcp.fts_index._retry_on_sqlite_locked", _exhausted
        )
        assert idx.optimize() is False

    def test_optimize_propagates_other_operational_errors(self, monkeypatch) -> None:
        """Non-lock OperationalErrors are not swallowed."""
        idx = FTSIndex(":memory:")

        monkeypatch.setattr(idx, "_conn", lambda: _FailingConnection("disk I/O error"))
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            idx.optimize()


# ---------------------------------------------------------------------------
# IndexManager bulk-purge call sites
# ---------------------------------------------------------------------------


class TestBulkPurgeOptimize:
    """Integration tests for the optimize trigger in IndexManager."""

    def test_reindex_bulk_delete_triggers_optimize(self, tmp_path: Path) -> None:
        """Deleting >= threshold docs in one reindex pass runs FTS optimize."""
        vault = tmp_path / "vault"
        files = _write_docs(vault, 30)
        mgr, fts = _make_index_mgr(vault, tmp_path)
        mgr.build_index()

        # Remove 26 files (>= OPTIMIZE_MIN_PURGED_DOCS) from disk.
        for f in files[:26]:
            f.unlink()

        with patch.object(fts, "optimize", wraps=fts.optimize) as spy:
            result = mgr.reindex()

        assert result.deleted == 26
        assert spy.call_count == 1

    def test_reindex_small_delete_does_not_optimize(self, tmp_path: Path) -> None:
        """A purge below both thresholds skips the FTS optimize."""
        vault = tmp_path / "vault"
        files = _write_docs(vault, 30)
        mgr, fts = _make_index_mgr(vault, tmp_path)
        mgr.build_index()

        # Remove 1 of 30 files: below 25 docs and below 10% of the corpus.
        files[0].unlink()

        with patch.object(fts, "optimize", wraps=fts.optimize) as spy:
            result = mgr.reindex()

        assert result.deleted == 1
        assert spy.call_count == 0

    def test_exclusion_upgrade_purge_triggers_optimize(self, tmp_path: Path) -> None:
        """Newly-configured exclude patterns purging >= threshold docs
        (issue #255 upgrade path) trigger an FTS optimize on build_index."""
        vault = tmp_path / "vault"
        _write_docs(vault, 5)
        _write_docs(vault / ".claude", 26, prefix="transcript")

        # Phase 1: index WITHOUT exclude_patterns (old configuration).
        mgr1, fts = _make_index_mgr(vault, tmp_path / "s1")
        mgr1.build_index()
        assert len(fts.list_notes()) == 31

        # Phase 2: rebuild WITH exclude_patterns on the same index.
        mgr2, _ = _make_index_mgr(
            vault,
            tmp_path / "s2",
            fts=fts,
            exclude_patterns=[".claude/**"],
        )
        with patch.object(fts, "optimize", wraps=fts.optimize) as spy:
            mgr2.build_index()

        assert len(fts.list_notes()) == 5
        assert spy.call_count == 1

    def test_exclusion_upgrade_small_purge_does_not_optimize(
        self, tmp_path: Path
    ) -> None:
        """An exclusion purge below both thresholds skips the FTS optimize."""
        vault = tmp_path / "vault"
        _write_docs(vault, 30)
        _write_docs(vault / ".claude", 1, prefix="transcript")

        mgr1, fts = _make_index_mgr(vault, tmp_path / "s1")
        mgr1.build_index()

        mgr2, _ = _make_index_mgr(
            vault,
            tmp_path / "s2",
            fts=fts,
            exclude_patterns=[".claude/**"],
        )
        with patch.object(fts, "optimize", wraps=fts.optimize) as spy:
            mgr2.build_index()

        assert len(fts.list_notes()) == 30
        assert spy.call_count == 0


# ---------------------------------------------------------------------------
# auto_vacuum setup + legacy migration
# ---------------------------------------------------------------------------

# auto_vacuum mode codes: 0 == NONE, 1 == FULL, 2 == INCREMENTAL.
_AUTO_VACUUM_INCREMENTAL = 2
_AUTO_VACUUM_NONE = 0
_AUTO_VACUUM_FULL = 1


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
        mode = idx._conn().execute("PRAGMA auto_vacuum").fetchone()[0]
        idx.close()
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
        mode = idx._conn().execute("PRAGMA auto_vacuum").fetchone()[0]
        idx.close()
        assert mode == _AUTO_VACUUM_NONE

    def test_legacy_none_database_is_migrated_on_open(self, tmp_path: Path) -> None:
        """A populated DB created with auto_vacuum=NONE converts on next open."""
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
        assert idx._conn().execute("PRAGMA auto_vacuum").fetchone()[0] == (
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
        assert second._conn().execute("PRAGMA auto_vacuum").fetchone()[0] == (
            _AUTO_VACUUM_INCREMENTAL
        )
        second.close()

    def test_existing_full_auto_vacuum_is_left_untouched(self, tmp_path: Path) -> None:
        """A database already using auto_vacuum=FULL is not downgraded."""
        db_path = tmp_path / "full.db"
        legacy = sqlite3.connect(str(db_path))
        legacy.execute("PRAGMA auto_vacuum = FULL")
        legacy.execute("PRAGMA journal_mode = WAL")
        legacy.executescript(_SCHEMA_SQL)
        legacy.commit()
        legacy.close()
        assert _auto_vacuum_mode(db_path) == _AUTO_VACUUM_FULL

        idx = FTSIndex(db_path)
        # FULL already reclaims pages on commit; leave it as-is.
        assert idx._conn().execute("PRAGMA auto_vacuum").fetchone()[0] == (
            _AUTO_VACUUM_FULL
        )
        idx.close()
        assert _auto_vacuum_mode(db_path) == _AUTO_VACUUM_FULL

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
        conn = _MigrationStubConnection(locked=False)
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            _apply_auto_vacuum(conn, "legacy.db")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FTSIndex._incremental_vacuum (post-optimize)
# ---------------------------------------------------------------------------


class TestIncrementalVacuum:
    """Tests for incremental_vacuum following optimize()."""

    def test_optimize_shrinks_file_via_incremental_vacuum(self, tmp_path: Path) -> None:
        """optimize() returns freed pages to the OS, shrinking the file.

        With auto_vacuum=INCREMENTAL the post-optimize incremental vacuum
        reduces page_count and empties the freelist after a bulk delete.
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

        for i in range(60):
            idx.delete_by_path(f"doc{i}.md")
        pages_before = idx._conn().execute("PRAGMA page_count").fetchone()[0]

        assert idx.optimize() is True
        pages_after = idx._conn().execute("PRAGMA page_count").fetchone()[0]
        freelist_after = idx._conn().execute("PRAGMA freelist_count").fetchone()[0]
        idx.close()

        # Incremental vacuum should have collapsed the page count and emptied
        # the freelist (a no-arg PRAGMA incremental_vacuum reclaims everything).
        assert pages_after < pages_before
        assert freelist_after == 0

    def test_incremental_vacuum_tolerates_locked_database(self, monkeypatch) -> None:
        """A busy/locked database skips the incremental vacuum, returning False."""
        idx = FTSIndex(":memory:")
        monkeypatch.setattr(
            idx, "_conn", lambda: _FailingConnection("database is locked")
        )
        assert idx._incremental_vacuum(page_size=4096) is False

    def test_incremental_vacuum_propagates_other_operational_errors(
        self, monkeypatch
    ) -> None:
        """Non-lock OperationalErrors from the vacuum are not swallowed."""
        idx = FTSIndex(":memory:")
        monkeypatch.setattr(idx, "_conn", lambda: _FailingConnection("disk I/O error"))
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            idx._incremental_vacuum(page_size=4096)
