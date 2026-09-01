"""Tests for per-tool-call commit scoping (issue #1264).

Pins the four properties the grouping depends on: writes fired under one scope
reach the callback as a single batch named after the tool; concurrent scopes
never merge; a write with no scope keeps the previous per-write contract; and a
drain flushes still-open scopes rather than reporting success over work no
commit contains.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest

from markdown_vault_mcp._commit_scope import (
    CommitScope,
    CommitScopeMiddleware,
    bound_commit_scope,
    current_commit_scope,
)
from markdown_vault_mcp.write_callback import WriteCallbackDispatcher

if TYPE_CHECKING:
    from collections.abc import Sequence

    from markdown_vault_mcp.types import WriteBatchItem


class _BatchRecorder:
    """A write callback that opts into batching and records both routes."""

    accepts_batch = True

    def __init__(self) -> None:
        self.batches: list[tuple[str, list[Path]]] = []
        self.singles: list[tuple[Path, str]] = []

    def __call__(
        self,
        abs_path: Path,
        content: str,  # noqa: ARG002 - required by the WriteCallback signature
        operation: str,
    ) -> None:
        self.singles.append((abs_path, operation))

    def on_write_batch(self, items: Sequence[WriteBatchItem], tool_name: str) -> None:
        self.batches.append((tool_name, [item[0] for item in items]))


class _PlainRecorder:
    """A callback that has NOT opted into batching."""

    def __init__(self) -> None:
        self.singles: list[tuple[Path, str]] = []

    def __call__(
        self,
        abs_path: Path,
        content: str,  # noqa: ARG002 - required by the WriteCallback signature
        operation: str,
    ) -> None:
        self.singles.append((abs_path, operation))


class TestScopeBinding:
    def test_no_scope_outside_a_tool_call(self) -> None:
        assert current_commit_scope() is None

    def test_scope_is_reset_afterward(self) -> None:
        with bound_commit_scope("write") as scope:
            assert current_commit_scope() is scope
        assert current_commit_scope() is None

    def test_each_binding_gets_a_distinct_token(self) -> None:
        with bound_commit_scope("write") as first:
            pass
        with bound_commit_scope("write") as second:
            pass
        assert first.token != second.token


class TestGrouping:
    def test_one_scope_produces_one_batch(self) -> None:
        cb = _BatchRecorder()
        dispatcher = WriteCallbackDispatcher(cb)
        with bound_commit_scope("okf_convert_links") as scope:
            for i in range(5):
                dispatcher.fire(Path(f"{i}.md"), str(i), "write")
            dispatcher.end_scope(scope)
        dispatcher.close()

        assert cb.singles == []
        assert cb.batches == [
            ("okf_convert_links", [Path(f"{i}.md") for i in range(5)])
        ]

    def test_a_scope_that_wrote_nothing_produces_no_batch(self) -> None:
        cb = _BatchRecorder()
        dispatcher = WriteCallbackDispatcher(cb)
        with bound_commit_scope("search") as scope:
            dispatcher.end_scope(scope)
        dispatcher.close()

        assert cb.batches == []
        assert cb.singles == []

    def test_writes_without_a_scope_dispatch_individually(self) -> None:
        cb = _BatchRecorder()
        dispatcher = WriteCallbackDispatcher(cb)
        dispatcher.fire(Path("a.md"), "body", "write")
        dispatcher.fire(Path("b.md"), "body", "write")
        dispatcher.close()

        assert cb.batches == []
        assert cb.singles == [(Path("a.md"), "write"), (Path("b.md"), "write")]

    def test_concurrent_scopes_do_not_merge(self) -> None:
        """Two tool calls interleaving in the queue stay separate commits.

        Grouping is keyed by token, not by queue position — position-based
        grouping would fold these two unrelated calls into one commit.
        """
        cb = _BatchRecorder()
        dispatcher = WriteCallbackDispatcher(cb)
        started = threading.Barrier(2)

        def worker(tool: str, name: str) -> None:
            with bound_commit_scope(tool) as scope:
                started.wait(timeout=5)
                dispatcher.fire(Path(name), "body", "write")
                dispatcher.end_scope(scope)

        threads = [
            threading.Thread(target=worker, args=("write", "one.md")),
            threading.Thread(target=worker, args=("edit", "two.md")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        dispatcher.close()

        assert len(cb.batches) == 2
        assert {tool for tool, _ in cb.batches} == {"write", "edit"}
        for _tool, paths in cb.batches:
            assert len(paths) == 1


class TestBatchOptIn:
    def test_callback_without_opt_in_still_gets_every_write(self) -> None:
        """Grouping is an optimisation, never a contract change."""
        cb = _PlainRecorder()
        dispatcher = WriteCallbackDispatcher(cb)
        with bound_commit_scope("okf_convert_links") as scope:
            dispatcher.fire(Path("a.md"), "body", "write")
            dispatcher.fire(Path("b.md"), "body", "delete")
            dispatcher.end_scope(scope)
        dispatcher.close()

        assert cb.singles == [(Path("a.md"), "write"), (Path("b.md"), "delete")]


class TestDrainFlushesOpenScopes:
    def test_drain_commits_a_still_open_scope(self) -> None:
        """``drain`` is what a git pull waits on before merging.

        Returning with a group still buffered would let the merge run over
        writes that are on disk but in no commit.
        """
        cb = _BatchRecorder()
        dispatcher = WriteCallbackDispatcher(cb)
        with bound_commit_scope("okf_convert_links"):
            dispatcher.fire(Path("a.md"), "body", "write")
            # Deliberately no end_scope: the scope is still open.
            assert dispatcher.drain(timeout=10) is True
            assert cb.batches == [("okf_convert_links", [Path("a.md")])]
        dispatcher.close()

    def test_close_flushes_a_still_open_scope(self) -> None:
        cb = _BatchRecorder()
        dispatcher = WriteCallbackDispatcher(cb)
        with bound_commit_scope("move_folder"):
            dispatcher.fire(Path("a.md"), "body", "write")
        dispatcher.close()

        assert cb.batches == [("move_folder", [Path("a.md")])]

    def test_end_scope_after_a_drain_is_a_noop(self) -> None:
        cb = _BatchRecorder()
        dispatcher = WriteCallbackDispatcher(cb)
        with bound_commit_scope("write") as scope:
            dispatcher.fire(Path("a.md"), "body", "write")
            dispatcher.drain(timeout=10)
            dispatcher.end_scope(scope)
        dispatcher.close()

        assert cb.batches == [("write", [Path("a.md")])]


class TestFailureIsolation:
    def test_a_raising_batch_callback_does_not_kill_the_worker(self) -> None:
        class _Exploding(_BatchRecorder):
            def on_write_batch(
                self, items: Sequence[WriteBatchItem], tool_name: str
            ) -> None:
                if tool_name == "boom":
                    raise RuntimeError("batch failed")
                super().on_write_batch(items, tool_name)

        cb = _Exploding()
        dispatcher = WriteCallbackDispatcher(cb)
        with bound_commit_scope("boom") as first:
            dispatcher.fire(Path("a.md"), "body", "write")
            dispatcher.end_scope(first)
        with bound_commit_scope("write") as second:
            dispatcher.fire(Path("b.md"), "body", "write")
            dispatcher.end_scope(second)
        dispatcher.close()

        assert cb.batches == [("write", [Path("b.md")])]

    def test_end_scope_is_a_noop_when_no_callback_is_configured(self) -> None:
        dispatcher = WriteCallbackDispatcher(None)
        with bound_commit_scope("write") as scope:
            dispatcher.end_scope(scope)
        dispatcher.close()

    def test_end_scope_after_close_is_a_noop(self) -> None:
        cb = _BatchRecorder()
        dispatcher = WriteCallbackDispatcher(cb)
        dispatcher.fire(Path("a.md"), "body", "write")
        dispatcher.close()
        with bound_commit_scope("write") as scope:
            dispatcher.end_scope(scope)
        assert cb.batches == []


class _FakeMessage:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeMiddlewareContext:
    def __init__(self, tool: str, fastmcp_context: object | None) -> None:
        self.message = _FakeMessage(tool)
        self.fastmcp_context = fastmcp_context


class TestMiddleware:
    @staticmethod
    async def _run(
        middleware: CommitScopeMiddleware,
        context: object,
        seen: list[CommitScope | None],
        *,
        raises: bool = False,
    ) -> str:
        async def call_next(_ctx: object) -> str:
            seen.append(current_commit_scope())
            if raises:
                raise RuntimeError("tool blew up")
            return "ok"

        return await middleware.on_call_tool(context, call_next)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_scope_is_bound_during_the_call_and_reset_after(self) -> None:
        middleware = CommitScopeMiddleware()
        context = _FakeMiddlewareContext("write", None)
        seen: list[CommitScope | None] = []

        result = await self._run(middleware, context, seen)

        assert result == "ok"
        assert seen[0] is not None
        assert seen[0].tool_name == "write"
        assert current_commit_scope() is None

    @pytest.mark.asyncio
    async def test_scope_closes_even_when_the_tool_raises(self) -> None:
        """A tool that writes then fails must still commit what it wrote."""
        closed: list[CommitScope] = []

        class _Vault:
            def end_commit_scope(self, scope: CommitScope) -> None:
                closed.append(scope)

        middleware = CommitScopeMiddleware()
        context = _FakeMiddlewareContext("write", object())
        seen: list[CommitScope | None] = []

        with (
            mock.patch("markdown_vault_mcp.domain.get_vault", return_value=_Vault()),
            pytest.raises(RuntimeError, match="tool blew up"),
        ):
            await self._run(middleware, context, seen, raises=True)

        assert len(closed) == 1
        assert closed[0].tool_name == "write"
        assert current_commit_scope() is None

    @pytest.mark.asyncio
    async def test_close_is_skipped_without_a_fastmcp_context(self) -> None:
        middleware = CommitScopeMiddleware()
        context = _FakeMiddlewareContext("write", None)
        seen: list[CommitScope | None] = []

        # No vault lookup happens, so an unpatched get_vault must not be hit.
        assert await self._run(middleware, context, seen) == "ok"

    @pytest.mark.asyncio
    async def test_an_unexpected_close_failure_does_not_destroy_the_result(
        self,
    ) -> None:
        """Bookkeeping runs in a ``finally``; anything it raises replaces the
        tool's result. Only RuntimeError/AttributeError were caught at first,
        so a ValueError from get_vault destroyed a successful call."""
        middleware = CommitScopeMiddleware()
        context = _FakeMiddlewareContext("write", object())
        seen: list[CommitScope | None] = []

        with mock.patch(
            "markdown_vault_mcp.domain.get_vault",
            side_effect=ValueError("something unexpected"),
        ):
            assert await self._run(middleware, context, seen) == "ok"

    @pytest.mark.asyncio
    async def test_a_vault_that_is_not_up_does_not_fail_the_tool_call(self) -> None:
        """Tool listing before startup must not raise out of the middleware."""
        middleware = CommitScopeMiddleware()
        context = _FakeMiddlewareContext("write", object())
        seen: list[CommitScope | None] = []

        with mock.patch(
            "markdown_vault_mcp.domain.get_vault",
            side_effect=RuntimeError("Vault not initialised"),
        ):
            assert await self._run(middleware, context, seen) == "ok"
