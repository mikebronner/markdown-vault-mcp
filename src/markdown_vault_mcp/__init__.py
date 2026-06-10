"""Generic markdown collection with FTS5 + semantic search.

Public attributes are loaded lazily (PEP 562) instead of eagerly importing
every submodule here. Eager imports pulled the full dependency tree
(``python-frontmatter`` → PyYAML, including the ``yaml._yaml`` C extension)
into any import of this package — which broke
``pytest --cov=markdown_vault_mcp.<submodule>``: coverage.py resolves dotted
source packages with :func:`importlib.util.find_spec` inside a
sys.modules-restoring context, unloading PyYAML's pure-Python modules while
the single-phase-init C extension stayed cached with references to the
original classes. Every subsequent ``CSafeLoader`` parse then failed with
``ConstructorError: could not determine a constructor for the tag None``.
Keeping this module import-light avoids that interaction entirely (and
speeds up CLI startup).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from markdown_vault_mcp.collection import Collection
    from markdown_vault_mcp.config import CollectionConfig, load_config
    from markdown_vault_mcp.exceptions import (
        ConcurrentModificationError,
        ConfigurationError,
        DocumentExistsError,
        DocumentNotFoundError,
        EditConflictError,
        MarkdownMCPError,
        ReadOnlyError,
    )
    from markdown_vault_mcp.git import GitWriteStrategy, git_write_strategy
    from markdown_vault_mcp.types import (
        AttachmentContent,
        AttachmentInfo,
        ChangeSet,
        Chunk,
        CollectionStats,
        DeleteResult,
        EditResult,
        FTSResult,
        IndexStats,
        MostLinkedNote,
        NoteContent,
        NoteContext,
        NoteInfo,
        ParsedNote,
        ReindexResult,
        RenameResult,
        SearchResult,
        SimilarItem,
        WriteCallback,
        WriteResult,
    )

# Public attribute name → defining submodule. Resolved on first access by
# __getattr__ below.
_EXPORTS: dict[str, str] = {
    "Collection": "markdown_vault_mcp.collection",
    "CollectionConfig": "markdown_vault_mcp.config",
    "load_config": "markdown_vault_mcp.config",
    "ConcurrentModificationError": "markdown_vault_mcp.exceptions",
    "ConfigurationError": "markdown_vault_mcp.exceptions",
    "DocumentExistsError": "markdown_vault_mcp.exceptions",
    "DocumentNotFoundError": "markdown_vault_mcp.exceptions",
    "EditConflictError": "markdown_vault_mcp.exceptions",
    "MarkdownMCPError": "markdown_vault_mcp.exceptions",
    "ReadOnlyError": "markdown_vault_mcp.exceptions",
    "GitWriteStrategy": "markdown_vault_mcp.git",
    "git_write_strategy": "markdown_vault_mcp.git",
    "AttachmentContent": "markdown_vault_mcp.types",
    "AttachmentInfo": "markdown_vault_mcp.types",
    "ChangeSet": "markdown_vault_mcp.types",
    "Chunk": "markdown_vault_mcp.types",
    "CollectionStats": "markdown_vault_mcp.types",
    "DeleteResult": "markdown_vault_mcp.types",
    "EditResult": "markdown_vault_mcp.types",
    "FTSResult": "markdown_vault_mcp.types",
    "IndexStats": "markdown_vault_mcp.types",
    "MostLinkedNote": "markdown_vault_mcp.types",
    "NoteContent": "markdown_vault_mcp.types",
    "NoteContext": "markdown_vault_mcp.types",
    "NoteInfo": "markdown_vault_mcp.types",
    "ParsedNote": "markdown_vault_mcp.types",
    "ReindexResult": "markdown_vault_mcp.types",
    "RenameResult": "markdown_vault_mcp.types",
    "SearchResult": "markdown_vault_mcp.types",
    "SimilarItem": "markdown_vault_mcp.types",
    "WriteCallback": "markdown_vault_mcp.types",
    "WriteResult": "markdown_vault_mcp.types",
}

__all__ = [
    "AttachmentContent",
    "AttachmentInfo",
    "ChangeSet",
    "Chunk",
    "Collection",
    "CollectionConfig",
    "CollectionStats",
    "ConcurrentModificationError",
    "ConfigurationError",
    "DeleteResult",
    "DocumentExistsError",
    "DocumentNotFoundError",
    "EditConflictError",
    "EditResult",
    "FTSResult",
    "GitWriteStrategy",
    "IndexStats",
    "MarkdownMCPError",
    "MostLinkedNote",
    "NoteContent",
    "NoteContext",
    "NoteInfo",
    "ParsedNote",
    "ReadOnlyError",
    "ReindexResult",
    "RenameResult",
    "SearchResult",
    "SimilarItem",
    "WriteCallback",
    "WriteResult",
    "git_write_strategy",
    "load_config",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve a public attribute from its defining submodule.

    Args:
        name: Attribute name being looked up on the package.

    Returns:
        The resolved attribute.

    Raises:
        AttributeError: If *name* is not a public attribute of this package.
    """
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg) from None
    value = getattr(importlib.import_module(module_name), name)
    # Cache so subsequent lookups hit the module __dict__ directly instead
    # of re-entering __getattr__.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazily resolved public attributes to :func:`dir`.

    Returns:
        Sorted list of public attribute names.
    """
    return sorted(set(__all__) | set(globals()))
