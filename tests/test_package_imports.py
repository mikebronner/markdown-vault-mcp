"""Regression tests for the lazy (PEP 562) package root.

``pytest --cov=markdown_vault_mcp.<submodule>`` used to fail dozens of tests
with ``yaml.constructor.ConstructorError: could not determine a constructor
for the tag None``: coverage.py resolves dotted source packages with
``importlib.util.find_spec`` inside a sys.modules-restoring context, and the
eager package ``__init__`` dragged PyYAML (including the cached single-phase
``yaml._yaml`` C extension) into that disposable import. These tests pin the
fix: importing the package root must stay light.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import markdown_vault_mcp


class TestLazyExports:
    def test_all_public_attributes_resolve(self) -> None:
        """Every name in __all__ resolves via the lazy __getattr__."""
        for name in markdown_vault_mcp.__all__:
            assert getattr(markdown_vault_mcp, name) is not None

    def test_exports_map_matches_all(self) -> None:
        """The lazy export map and __all__ stay in sync."""
        assert set(markdown_vault_mcp._EXPORTS) == set(markdown_vault_mcp.__all__)

    def test_unknown_attribute_raises(self) -> None:
        """Unknown attributes raise AttributeError, as a normal module would."""
        with pytest.raises(AttributeError, match="no attribute 'nope'"):
            _ = markdown_vault_mcp.nope

    def test_dir_includes_public_names(self) -> None:
        """dir() advertises the lazily resolved public surface."""
        listing = dir(markdown_vault_mcp)
        assert "Collection" in listing
        assert "ReindexResult" in listing


class TestImportIsLight:
    """The package root must not import the heavy dependency tree.

    Run in a subprocess so the assertions see a clean interpreter rather
    than whatever this test session has already imported.
    """

    def _run(self, code: str) -> None:
        """Execute *code* in a fresh interpreter and assert it succeeds."""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_package_import_does_not_load_yaml(self) -> None:
        """``import markdown_vault_mcp`` must not pull in PyYAML/frontmatter."""
        self._run(
            "import sys; import markdown_vault_mcp; "
            "heavy = {'yaml', 'frontmatter'} & "
            "{m.split('.')[0] for m in sys.modules}; "
            "assert not heavy, f'package import loaded {heavy}'"
        )

    def test_find_spec_on_submodule_does_not_load_yaml(self) -> None:
        """``find_spec('markdown_vault_mcp.tracker')`` must not import PyYAML.

        This is exactly what coverage.py does (inside a sys.modules-restoring
        context) to resolve ``--cov=markdown_vault_mcp.tracker``; if PyYAML
        gets imported here it is subsequently unloaded, corrupting the cached
        ``yaml._yaml`` C extension for the rest of the process.
        """
        self._run(
            "import importlib.util, sys; "
            "importlib.util.find_spec('markdown_vault_mcp.tracker'); "
            "heavy = {'yaml', 'frontmatter'} & "
            "{m.split('.')[0] for m in sys.modules}; "
            "assert not heavy, f'find_spec loaded {heavy}'"
        )

    def test_yaml_survives_simulated_coverage_source_resolution(self) -> None:
        """PyYAML still parses after coverage-style find_spec + module purge."""
        self._run(
            "import importlib.util, sys; "
            "before = set(sys.modules); "
            "importlib.util.find_spec('markdown_vault_mcp.collection'); "
            "[sys.modules.pop(m) for m in set(sys.modules) - before]; "
            "import frontmatter; "
            "post = frontmatter.loads('---\\ntitle: Hello\\n---\\nbody\\n'); "
            "assert post.metadata == {'title': 'Hello'}, post.metadata"
        )
