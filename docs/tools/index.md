# MCP Tools

markdown-vault-mcp exposes MCP tools across several categories. Write tools are only available when `MARKDOWN_VAULT_MCP_READ_ONLY=false`.

## Quick Reference

| Tool | Category | Description |
|------|----------|-------------|
| [`search`](#search) | Read | Hybrid full-text + semantic search with optional frontmatter filters |
| [`read`](#read) | Read | Read a document or attachment by relative path |
| [`list_documents`](#list_documents) | Read | List indexed documents and optionally attachments |
| [`list_folders`](#list_folders) | Read | List all folder paths in the vault |
| [`list_tags`](#list_tags) | Read | List all unique frontmatter tag values |
| [`stats`](#stats) | Read | Get collection statistics and capabilities |
| [`embeddings_status`](#embeddings_status) | Read | Check embedding provider and vector index status |
| [`get_backlinks`](#get_backlinks) | Read | Find all documents that link to a given document |
| [`get_outlinks`](#get_outlinks) | Read | Find all links from a document, with existence check |
| [`get_broken_links`](#get_broken_links) | Read | Find all links pointing to non-existent documents |
| [`get_similar`](#get_similar) | Read | Find semantically similar notes by document path |
| [`get_recent`](#get_recent) | Read | Get the most recently modified notes |
| [`get_context`](#get_context) | Read | Get a consolidated context dossier for a note |
| [`get_orphan_notes`](#get_orphan_notes) | Read | Find notes with no inbound or outbound links |
| [`get_most_linked`](#get_most_linked) | Read | Find the most-linked-to notes ranked by backlink count |
| [`get_connection_path`](#get_connection_path) | Read | Find the shortest path between two notes via link graph |
| [`reindex`](#reindex) | Admin | Force a full reindex of the vault |
| [`build_embeddings`](#build_embeddings) | Admin | Build or rebuild vector embeddings |
| [`write`](#write) | Write | Create or overwrite a document or attachment |
| [`edit`](#edit) | Write | Replace a unique text span in a document |
| [`delete`](#delete) | Write | Delete a document or attachment |
| [`rename`](#rename) | Write | Rename/move a document or attachment |
| [`fetch`](#fetch) | Write | Download from URL and save to vault |
| [`create_download_link`](#create_download_link) | Write | Generate a one-time download URL for a vault file |
| [`browse_vault`](#browse_vault) | Apps | Open the vault explorer SPA |
| [`show_context`](#show_context) | Apps | Open the Context Card for a note |

---

## Search & Discovery

### `search`

Find documents matching a query using full-text or semantic search.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Natural language or keyword query string |
| `limit` | int | `10` | Maximum results to return |
| `mode` | string | `"keyword"` | `"keyword"` (FTS5/BM25), `"semantic"` (vector similarity), or `"hybrid"` (reciprocal rank fusion) |
| `folder` | string | `null` | Restrict to documents under this folder path |
| `filters` | object | `null` | Filter by indexed frontmatter field values (e.g. `{"tags": "pacing"}`) |

**Returns:** List of result dicts ranked by relevance. Each contains: `path`, `title`, `folder`, `content` (matched chunk), `score`, `frontmatter`.

!!! tip "Choosing a search mode"
    - Use `mode="hybrid"` when semantic search is available — it combines keyword precision with semantic understanding
    - Use `mode="keyword"` for exact term matches
    - Use `mode="semantic"` for meaning-based similarity
    - Check `stats` to see if `semantic_search_available` is true

**Example usage:**

```json
{
  "query": "character development techniques",
  "mode": "hybrid",
  "limit": 5,
  "filters": {"tags": "craft"}
}
```

### `read`

Read the full content of a document or attachment by path.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | Relative path to the document or attachment (e.g. `"Journal/note.md"` or `"assets/diagram.pdf"`) |

**Returns:**

=== "Markdown document"

    ```json
    {
      "path": "Journal/note.md",
      "title": "My Note",
      "folder": "Journal",
      "content": "The markdown body...",
      "frontmatter": {"title": "My Note", "tags": ["journal"]},
      "modified_at": 1741564800.0
    }
    ```

=== "Attachment"

    ```json
    {
      "path": "assets/diagram.pdf",
      "mime_type": "application/pdf",
      "size_bytes": 12345,
      "content_base64": "<base64 string>",
      "modified_at": 1741564800.0
    }
    ```

### `list_documents`

List documents (and optionally attachments) in the collection.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `folder` | string | `null` | Return only documents in this folder |
| `pattern` | string | `null` | Unix glob matched against relative paths (e.g. `"Journal/*.md"`) |
| `include_attachments` | bool | `false` | When true, also returns non-`.md` files that match the configured allowlist |

**Returns:** List of info dicts. Every entry has a `kind` field (`"note"` or `"attachment"`). Body content is not included — call `read` for full text.

### `list_folders`

List all folder paths that contain documents. Use this to discover valid folder names for filtering `search` or `list_documents`. The root folder (top-level documents) is represented as an empty string `""`.

**Returns:** Sorted list of folder paths, e.g. `["", "Journal", "Projects"]`.

### `list_tags`

List all distinct values for a frontmatter field across the collection.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field` | string | `"tags"` | Frontmatter field name to enumerate. Must match a field in `indexed_frontmatter_fields` (check `stats`) |

**Returns:** Sorted list of distinct string values, e.g. `["craft", "pacing", "worldbuilding"]`.

### `stats`

Get an overview of the collection's size, capabilities, and configuration. Call this at the start of a session to understand what the collection contains and what search modes are available.

**Returns:**

```json
{
  "document_count": 42,
  "chunk_count": 156,
  "folder_count": 5,
  "semantic_search_available": true,
  "indexed_frontmatter_fields": ["tags", "cluster"],
  "attachment_extensions": ["pdf", "png", "jpg"]
}
```

### `embeddings_status`

Check the embedding provider configuration and vector index status. Use this to diagnose why semantic search is unavailable.

**Returns:**

```json
{
  "available": true,
  "provider": "OllamaProvider",
  "chunk_count": 156,
  "path": "/data/state/embeddings/embeddings"
}
```

---

## Index Management

### `reindex`

Incrementally update the full-text search index to reflect file changes made outside this server. Only processes changed files — unchanged documents are skipped.

If semantic search is already active (vector index loaded), this also re-embeds changed documents automatically.

**Returns:** `{"added": 3, "modified": 1, "deleted": 0, "unchanged": 38, "skipped": 2}`

`skipped` counts files present on disk that were deliberately not indexed (missing required frontmatter, matching an exclude pattern, or unparseable). Unchanged skipped files are remembered and not re-evaluated until their content changes.

### `build_embeddings`

Build vector embeddings to enable semantic and hybrid search. This can be slow for large collections.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `force` | bool | `false` | When true, discards existing embeddings and rebuilds from scratch. Use only if the embedding model has changed |

**Returns:** `{"chunks_embedded": 156}`

!!! note "When to use"
    Call `build_embeddings` once to enable semantic search for the first time. After that, `reindex` handles incremental re-embedding automatically.

---

## Write Operations

!!! warning "Write tools require `MARKDOWN_VAULT_MCP_READ_ONLY=false`"
    These tools are hidden when the server is in read-only mode (the default).

### `write`

Create or overwrite a document or attachment.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | Relative path. Extension determines handling (`.md` = note, else attachment) |
| `content` | string | Full markdown body for `.md` files (excluding frontmatter). Ignored for attachments |
| `frontmatter` | object | Optional YAML frontmatter dict for `.md` files. Ignored for attachments |
| `content_base64` | string | Base64-encoded binary content for attachment files. Required when path is not `.md` |

**Returns:** `{"path": "Journal/note.md", "created": true}`

!!! warning
    `write` replaces the entire file — use `edit` for targeted changes to existing documents.

### `edit`

Make a targeted text replacement in an existing document. Supports three modes:

- **Exact match** (`old_text` only) — must appear exactly once in the document.
- **Line-range** (`line_start` + `line_end`, no `old_text`) — replaces the specified lines. Pass `if_match` for safety.
- **Scoped match** (`old_text` + `line_start`/`line_end`) — searches for `old_text` within the specified line range only.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | Relative path to the document |
| `old_text` | string | Conditional | Text to replace. Required unless using line-range mode |
| `new_text` | string | Yes | Replacement text |
| `if_match` | string | No | Etag from `read` for optimistic concurrency |
| `line_start` | integer | Conditional | First line to replace (1-based, inclusive). Required with `line_end` |
| `line_end` | integer | Conditional | Last line to replace (1-based, inclusive). Required with `line_start` |

**Returns:** `{"path": "Journal/note.md", "replacements": 1, "match_type": "exact"}`

`match_type` is `"exact"` when the text matched byte-for-byte, or `"normalized"` when it matched after Unicode/whitespace normalization.

!!! tip "Usage pattern"
    Always call `read` first to get the exact current text and line numbers. For small edits, use `old_text` (exact match). For large block replacements, use `line_start`/`line_end` with the line numbers shown by `read`. Frontmatter can be edited — `old_text` may span the YAML block.

!!! info "Normalized matching"
    When exact match fails, the tool automatically tries a normalized comparison: Unicode NFC, dash normalization (en-dash/em-dash → hyphen), smart quote normalization, whitespace collapsing. If a unique match is found, it proceeds and returns `match_type: "normalized"`.

!!! warning "Diagnostic errors"
    When no match is found, the error message includes diagnostic info: the closest matching line number, the character position of the first difference, and short snippets showing what was expected vs. what was found. This helps identify the exact mismatch.

### `delete`

Permanently delete a document or attachment. For `.md` documents, also removes from all search indices.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | Relative path to the document or attachment to delete |

**Returns:** `{"path": "Journal/old-note.md"}`

!!! danger
    This is irreversible unless git history exists. Confirm the path with the user before calling.

### `rename`

Rename a document or attachment, or move it to a different folder. Parent directories for the new path are created automatically.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `old_path` | string | Current relative path |
| `new_path` | string | Target relative path. Fails if `new_path` already exists |

**Returns:** `{"old_path": "drafts/idea.md", "new_path": "projects/idea.md"}`

### `fetch`

Download a file from a URL and save it to the vault as a note or attachment. Designed for MCP-to-MCP file transfer when content is too large for the LLM context window.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | string | required | Source URL to download. Only `http`/`https` schemes allowed; private/loopback IPs are blocked; redirects are not followed (SSRF protection) |
| `path` | string | required | Destination path in vault. Extension determines handling: `.md` for notes, anything else for attachments |
| `frontmatter` | object | `null` | Optional YAML frontmatter dict for `.md` files. Ignored for attachments |
| `if_match` | string | `null` | Optional etag from a previous `read` call for optimistic concurrency |
| `timeout_s` | float | `30.0` | Download timeout in seconds |

**Returns:** `{"path": "notes/report.md", "created": true, "content_length": 4096, "content_type": "text/markdown"}`

!!! note "Dependency"
    Requires `httpx`. Install with `pip install 'markdown-vault-mcp[all]'`.

### `create_download_link`

Generate a one-time download URL for a vault file. The link expires after a single use. Useful for MCP-to-MCP file transfer where the receiving server can fetch the file directly.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | string | required | Relative path to the vault file to share |
| `ttl_seconds` | int | `300` | Link lifetime in seconds |

**Returns:** `{"download_url": "https://mcp.example.com/artifacts/abc123", "expires_in_seconds": 300, "path": "notes/report.md", "content_type": "text/markdown"}`

!!! note "Requirements"
    Only available with HTTP or SSE transport. Requires `MARKDOWN_VAULT_MCP_BASE_URL` to be set.

---

## Link Graph

### `get_backlinks`

Find all documents that link to a given document.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | Relative path to the target document |

**Returns:** List of documents containing links to the given path.

### `get_outlinks`

Find all links from a document, with existence check.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | Relative path to the source document |

**Returns:** List of link targets with an `exists` field indicating whether the target document is in the vault.

### `get_broken_links`

Find all links across the vault pointing to non-existent documents.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `folder` | string | `null` | Optional folder filter; only checks links from documents in this folder |

**Returns:** List of entries with `source_path`, `source_title`, `target_path`, `link_text`, `link_type`, `fragment`, and `raw_target` fields.

### `get_similar`

Find semantically similar notes by document path. Requires embeddings to be built.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | string | required | Relative path to the document |
| `limit` | int | `10` | Maximum results to return |

**Returns:** List of similar documents ranked by cosine similarity.

### `get_recent`

Get the most recently modified notes.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | `20` | Maximum results to return |
| `folder` | string | `null` | Optional folder filter; only returns notes from this folder (e.g. `"Journal"`) |

**Returns:** List of notes with Unix timestamps (`modified_at` as float), sorted by modification time (newest first).

### `get_context`

Get a consolidated context dossier for a note. Combines backlinks, outlinks, similar notes, folder peers, tags, and modification time into a single response.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | string | required | Relative path to the document |
| `similar_limit` | int | `5` | Max similar notes to include. Pass `0` to skip the similarity lookup (e.g. when `stats` shows `semantic_search_available=false`) |
| `link_limit` | int | `10` | Max backlinks and outlinks to include each |

**Returns:** Object with `path`, `title`, `folder`, `frontmatter`, `modified_at`, `backlinks`, `outlinks`, `similar`, `folder_notes`, and `tags` fields.

### `get_orphan_notes`

Find all notes with no inbound or outbound links — isolated documents that may need cross-referencing.

**Returns:** List of `NoteInfo` objects (`path`, `title`, `folder`, `frontmatter`, `modified_at`, `kind`), ordered by path. Returns ALL orphans with no limit — check `stats.orphan_count` before calling on large vaults.

### `get_most_linked`

Find the most-linked-to notes in the vault, ranked by backlink count.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | `10` | Maximum results to return |

**Returns:** List of `{"path": "...", "backlink_count": N}` entries.

### `get_connection_path`

Find the shortest path between two notes via BFS on the undirected link graph (max 10 hops).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | string | required | Relative path to the starting document |
| `target` | string | required | Relative path to the target document |
| `max_depth` | int | `10` | Maximum hops to search (clamped to [1, 10]) |

**Returns:** Object with `found` (bool), `path` (ordered list of note paths from source to target), and `hops` (number of edges, or `-1` if not found).

---

## MCP Apps

These tools power the browser-based vault explorer views. See the [MCP Apps guide](../guides/mcp-apps.md) for details.

### `browse_vault`

Open the vault explorer SPA. Optionally focus on a specific note and view.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | string | `null` | Note path to focus on |
| `view` | string | `null` | View to open: `context`, `graph`, `browse`, or `note` |

**Returns:** For Apps-capable clients, opens the interactive SPA. For other clients, returns a text summary.

### `show_context`

Open the Context Card view for a specific note, showing backlinks, outlinks, similar notes, tags, and folder peers.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | Relative path to the document |

**Returns:** For Apps-capable clients, opens the Context Card. For other clients, returns the context dossier as text.
