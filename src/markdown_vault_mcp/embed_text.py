"""Shared builder for context-enriched embedding text.

One :class:`EmbedTextBuilder` instance is constructed by
:class:`~markdown_vault_mcp.vault.Vault` and shared by every site that
produces embedding input text (hot reindex, cold build, boot convergence,
deferred flush). Using a single builder everywhere is load-bearing: if any
site embedded plain chunk content while another embedded enriched text, the
boot convergence pass would "heal" the enriched vectors back to plain.

The builder has two formats:

* **v1** — the default: :meth:`EmbedTextBuilder.build` returns the chunk
  content byte-for-byte unchanged, so default configurations are an exact
  behavioural no-op.
* **v2** — active when ``embed_context`` is enabled or any
  ``searchable_fields`` are configured: the chunk content is prefixed with
  the document title, the chunk heading (when present), and — on the first
  chunk only — the newline-joined scalar frontmatter values of the
  configured fields.

:meth:`EmbedTextBuilder.format_token` canonicalises the active format into
a string persisted in the vector sidecar's ``index_metadata`` so a format
flip is detected at load time and routes to a full re-embed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)


def fields_text(
    frontmatter_or_json: Mapping[str, Any] | str | None,
    fields: Sequence[str],
) -> str:
    """Return the newline-joined scalar values of *fields* from frontmatter.

    Shared by the FTS ``summary`` column population and the embedding
    preamble so both are always computed identically.

    Args:
        frontmatter_or_json: A parsed frontmatter mapping, a
            ``frontmatter_json`` string (as stored in the ``documents``
            table), or ``None``.
        fields: Frontmatter keys to extract, in order.

    Returns:
        The values of the listed fields — scalars only (``str``, ``int``,
        ``float``, ``bool``); lists/dicts/``None`` are skipped — joined
        with newlines. ``""`` when no fields are configured, the
        frontmatter is empty, or the JSON does not parse.
    """
    if not fields or not frontmatter_or_json:
        return ""
    frontmatter: Mapping[str, Any]
    if isinstance(frontmatter_or_json, str):
        try:
            parsed = json.loads(frontmatter_or_json)
        except json.JSONDecodeError:
            logger.warning("fields_text: invalid frontmatter JSON — ignoring")
            return ""
        if not isinstance(parsed, dict):
            return ""
        frontmatter = parsed
    else:
        frontmatter = frontmatter_or_json
    parts: list[str] = []
    for key in fields:
        value = frontmatter.get(key)
        if isinstance(value, (str, int, float, bool)):
            text = str(value).strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


@dataclass(frozen=True)
class EmbedTextBuilder:
    """Builds the text submitted to the embedding provider for one chunk.

    Args:
        embed_context: When ``True``, chunk content is enriched with the
            document title and chunk heading (format v2).
        searchable_fields: Frontmatter keys whose scalar values form the
            first-chunk preamble (also switches to format v2 when
            non-empty).
    """

    embed_context: bool = False
    searchable_fields: tuple[str, ...] = field(default=())

    @property
    def is_v2(self) -> bool:
        """``True`` when enrichment is active (any knob set)."""
        return self.embed_context or bool(self.searchable_fields)

    def format_token(self) -> str:
        """Canonical format token persisted in the vector sidecar.

        Returns:
            ``"v1"`` for the default no-op format, else
            ``"v2;fields=<comma-joined field names>"``.
        """
        if not self.is_v2:
            return "v1"
        return "v2;fields=" + ",".join(self.searchable_fields)

    def fields_text(self, frontmatter_or_json: Mapping[str, Any] | str | None) -> str:
        """Return the first-chunk preamble for this builder's fields.

        See the module-level :func:`fields_text` for the exact contract.
        """
        return fields_text(frontmatter_or_json, self.searchable_fields)

    def build(
        self,
        *,
        title: str,
        heading: str | None,
        content: str,
        fields_text: str,
        is_first_chunk: bool,
    ) -> str:
        """Return the embedding input text for one chunk.

        Args:
            title: Resolved document title.
            heading: The chunk's heading, or ``None``/``""`` when absent.
            content: Raw chunk content.
            fields_text: Pre-computed preamble (see :meth:`fields_text`);
                only used on the first chunk.
            is_first_chunk: ``True`` for the document's chunk-0 row.

        Returns:
            v1: *content* unchanged (byte-exact no-op). v2: the title line,
            the heading line (omitted when empty), and — first chunk only —
            the non-empty *fields_text*, followed by a blank line and the
            content.
        """
        if not self.is_v2:
            return content
        lines = [title]
        if heading:
            lines.append(heading)
        if is_first_chunk and fields_text:
            lines.append(fields_text)
        return "\n".join(lines) + "\n\n" + content
