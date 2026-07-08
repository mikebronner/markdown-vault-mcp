"""Tests for the shared embedding-text builder (embed_text.py)."""

from __future__ import annotations

import json

from markdown_vault_mcp.embed_text import EmbedTextBuilder, fields_text

# ---------------------------------------------------------------------------
# fields_text (module-level, shared with the FTS summary column)
# ---------------------------------------------------------------------------


class TestFieldsText:
    def test_joins_scalar_values_in_field_order(self) -> None:
        fm = {"summary": "An overview.", "type": "decision", "rank": 3}
        assert (
            fields_text(fm, ("summary", "type", "rank")) == "An overview.\ndecision\n3"
        )

    def test_skips_missing_none_and_non_scalar_values(self) -> None:
        fm = {
            "summary": "kept",
            "tags": ["a", "b"],
            "meta": {"nested": True},
            "empty": None,
        }
        assert fields_text(fm, ("summary", "tags", "meta", "empty", "absent")) == "kept"

    def test_accepts_frontmatter_json_string(self) -> None:
        raw = json.dumps({"summary": "from json"})
        assert fields_text(raw, ("summary",)) == "from json"

    def test_invalid_json_and_non_dict_json_return_empty(self) -> None:
        assert fields_text("{not json", ("summary",)) == ""
        assert fields_text(json.dumps(["a", "b"]), ("summary",)) == ""

    def test_no_fields_or_no_frontmatter_return_empty(self) -> None:
        assert fields_text({"summary": "x"}, ()) == ""
        assert fields_text(None, ("summary",)) == ""
        assert fields_text({}, ("summary",)) == ""


# ---------------------------------------------------------------------------
# format_token canonicalisation
# ---------------------------------------------------------------------------


class TestFormatToken:
    def test_default_builder_is_v1(self) -> None:
        assert EmbedTextBuilder().format_token() == "v1"

    def test_embed_context_alone_is_v2_with_empty_fields(self) -> None:
        builder = EmbedTextBuilder(embed_context=True)
        assert builder.format_token() == "v2;fields="

    def test_fields_alone_is_v2(self) -> None:
        builder = EmbedTextBuilder(searchable_fields=("summary", "type"))
        assert builder.format_token() == "v2;fields=summary,type"

    def test_field_order_is_canonical_not_sorted(self) -> None:
        a = EmbedTextBuilder(searchable_fields=("b", "a"))
        b = EmbedTextBuilder(searchable_fields=("a", "b"))
        assert a.format_token() != b.format_token()


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


class TestBuildV1:
    def test_v1_returns_content_byte_identical(self) -> None:
        builder = EmbedTextBuilder()
        content = "raw chunk content\nwith lines\n"
        out = builder.build(
            title="Title",
            heading="Heading",
            content=content,
            fields_text="ignored",
            is_first_chunk=True,
        )
        assert out is content or out == content
        assert out == content


class TestBuildV2:
    def test_first_chunk_includes_title_heading_and_fields(self) -> None:
        builder = EmbedTextBuilder(embed_context=True, searchable_fields=("summary",))
        out = builder.build(
            title="My Note",
            heading="Intro",
            content="body text",
            fields_text="A summary.",
            is_first_chunk=True,
        )
        assert out == "My Note\nIntro\nA summary.\n\nbody text"

    def test_first_chunk_without_heading_omits_heading_line(self) -> None:
        builder = EmbedTextBuilder(embed_context=True)
        out = builder.build(
            title="My Note",
            heading=None,
            content="body text",
            fields_text="A summary.",
            is_first_chunk=True,
        )
        assert out == "My Note\nA summary.\n\nbody text"

    def test_later_chunk_omits_fields_text(self) -> None:
        builder = EmbedTextBuilder(embed_context=True, searchable_fields=("summary",))
        out = builder.build(
            title="My Note",
            heading="Details",
            content="body text",
            fields_text="A summary.",
            is_first_chunk=False,
        )
        assert out == "My Note\nDetails\n\nbody text"

    def test_later_chunk_without_heading(self) -> None:
        builder = EmbedTextBuilder(embed_context=True)
        out = builder.build(
            title="My Note",
            heading="",
            content="body text",
            fields_text="",
            is_first_chunk=False,
        )
        assert out == "My Note\n\nbody text"

    def test_empty_fields_text_omitted_on_first_chunk(self) -> None:
        builder = EmbedTextBuilder(embed_context=True)
        out = builder.build(
            title="T",
            heading=None,
            content="c",
            fields_text="",
            is_first_chunk=True,
        )
        assert out == "T\n\nc"
