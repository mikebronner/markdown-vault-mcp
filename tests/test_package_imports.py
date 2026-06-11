"""Regression tests for the lazy (PEP 562) package root (#665, Problem 3).

``pytest --cov=markdown_vault_mcp.<submodule>`` used to kill the whole test
session at conftest load with ``ImportError: cannot import name 'claw_state'
from partially initialized module 'beartype.claw._clawstate'``: coverage.py
resolves dotted source packages with ``importlib.util.find_spec`` inside a
sys.modules-restoring context (``coverage.misc.sys_modules_saved``), and the
eager package ``__init__`` dragged the full dependency tree (``config`` ->
``fastmcp_pvl_core`` -> ``beartype``, plus ``frontmatter`` -> PyYAML) into
that disposable import. The purge on context exit removed beartype's modules
but left its claw import hook in ``sys.path_hooks``, so the next import
routed through an orphaned hook into a circular re-import. These tests pin
the fix: importing (or find_spec-ing) the package root must stay light.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import markdown_vault_mcp

# Top-level distributions that must never be imported by the package root.
# beartype arrives via fastmcp_pvl_core (whose claw import hook survives a
# sys.modules purge); yaml arrives via frontmatter (whose single-phase-init
# C extension keeps first-generation class references across a purge).
_HEAVY = "{'beartype', 'fastmcp_pvl_core', 'frontmatter', 'yaml'}"


class TestLazyExports:
    def test_all_public_attributes_resolve(self):
        """Every name in __all__ resolves via the lazy __getattr__."""
        for name in markdown_vault_mcp.__all__:
            assert getattr(markdown_vault_mcp, name) is not None

    def test_exports_map_matches_all(self):
        """The lazy export map and __all__ stay in sync."""
        assert set(markdown_vault_mcp._EXPORTS) == set(markdown_vault_mcp.__all__)

    def test_unknown_attribute_raises(self):
        """Unknown attributes raise AttributeError, as a normal module would."""
        with pytest.raises(AttributeError, match="no attribute 'nope'"):
            _ = markdown_vault_mcp.nope

    def test_dir_includes_public_names(self):
        """dir() advertises the lazily resolved public surface."""
        listing = dir(markdown_vault_mcp)
        for name in markdown_vault_mcp.__all__:
            assert name in listing

    def test_resolved_attribute_is_cached_in_module_dict(self):
        """First resolution binds the name into the module __dict__ so later
        lookups bypass __getattr__ entirely."""
        name = "Vault"
        vars(markdown_vault_mcp).pop(name, None)
        assert name not in vars(markdown_vault_mcp)
        first = getattr(markdown_vault_mcp, name)
        assert name in vars(markdown_vault_mcp)
        assert vars(markdown_vault_mcp)[name] is first


class TestImportIsLight:
    """The package root must not import the heavy dependency tree.

    Run in a subprocess so the assertions see a clean interpreter rather
    than whatever this test session has already imported.
    """

    def _run(self, code: str) -> None:
        """Execute code in a fresh interpreter and assert it succeeds."""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_package_import_is_light(self):
        """``import markdown_vault_mcp`` must not pull in the heavy deps."""
        self._run(
            "import sys; import markdown_vault_mcp; "
            f"heavy = {_HEAVY} & "
            "{m.split('.')[0] for m in sys.modules}; "
            "assert not heavy, f'package import loaded {heavy}'"
        )

    def test_find_spec_on_submodule_is_light(self):
        """``find_spec('markdown_vault_mcp.tracker')`` must stay light.

        This is exactly what coverage.py does (inside a sys.modules-restoring
        context) to resolve ``--cov=markdown_vault_mcp.tracker``; anything
        imported here is subsequently unloaded, orphaning beartype's claw
        ``sys.path_hooks`` entry and PyYAML's cached C extension.
        """
        self._run(
            "import importlib.util, sys; "
            "importlib.util.find_spec('markdown_vault_mcp.tracker'); "
            f"heavy = {_HEAVY} & "
            "{m.split('.')[0] for m in sys.modules}; "
            "assert not heavy, f'find_spec loaded {heavy}'"
        )

    def test_interpreter_survives_simulated_coverage_resolution(self):
        """Imports still work after coverage-style find_spec + module purge.

        Reproduces coverage.py's source-package resolution: find_spec on a
        dotted submodule, then purge every newly imported module. The
        interpreter must survive a subsequent heavy import (pre-fix this
        died in beartype's orphaned claw hook) and PyYAML parsing must
        still work (the 1.20.0-era symptom of the same root cause).
        """
        self._run(
            "import importlib.util, sys; "
            "before = set(sys.modules); "
            "importlib.util.find_spec('markdown_vault_mcp.tracker'); "
            "[sys.modules.pop(m) for m in set(sys.modules) - before]; "
            "import markdown_vault_mcp.config; "
            "import frontmatter; "
            "post = frontmatter.loads('---\\ntitle: Hello\\n---\\nbody\\n'); "
            "assert post.metadata == {'title': 'Hello'}, post.metadata"
        )

    def test_dotted_cov_invocation_passes(self, tmp_path):
        """The previously-fatal dotted --cov pytest invocation succeeds.

        End-to-end pin for #665 Problem 3: run the exact reported command in
        a subprocess. Pre-fix it aborted at conftest load with the beartype
        ``claw_state`` circular ImportError.
        """
        repo_root = Path(__file__).resolve().parent.parent
        env = {
            k: v
            for k, v in os.environ.items()
            # Strip outer pytest-cov subprocess hooks so the inner run
            # measures (and writes) its own coverage data only.
            if not k.startswith(("COV_CORE_", "COVERAGE_"))
        }
        # Keep the inner run's data file out of the repo root.
        env["COVERAGE_FILE"] = str(tmp_path / ".coverage")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_scanner.py",
                "--cov=markdown_vault_mcp.tracker",
                "--cov-fail-under=0",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=repo_root,
            env=env,
        )
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
