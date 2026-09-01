"""Git write-strategy configuration for a markdown vault."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from markdown_vault_mcp.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

#: Accepted ``commit_mode`` values. ``"write"`` commits each file as it is
#: written, the behaviour chosen in #54. ``"tool-call"`` groups every write
#: from one MCP tool call into a single commit named after that tool (#1264) —
#: the coexistence #54 anticipated with a mode knob it never built.
COMMIT_MODES: tuple[str, ...] = ("write", "tool-call")


@dataclass(frozen=True)
class GitConfig:
    """Git auth, identity, and sync cadence (``MARKDOWN_VAULT_MCP_GIT_*``)."""

    token: str | None = None
    repo_url: str | None = None
    username: str = "x-access-token"
    push_delay_s: float = 30.0
    commit_name: str = "markdown-vault-mcp"
    commit_email: str = "noreply@markdown-vault-mcp"
    commit_name_claim: str | None = None
    commit_email_claim: str | None = None
    lfs: bool = True
    pull_interval_s: int = 600
    commit_mode: str = "write"

    def __post_init__(self) -> None:
        """Validate cadences and the commit mode on every construction path (#638).

        Raises:
            ConfigurationError: If ``push_delay_s`` or ``pull_interval_s`` is
                negative, or ``commit_mode`` is not one of
                :data:`COMMIT_MODES`.
        """
        if self.push_delay_s < 0:
            raise ConfigurationError(
                f"push_delay_s must be >= 0, got {self.push_delay_s}"
            )
        if self.pull_interval_s < 0:
            raise ConfigurationError(
                f"pull_interval_s must be >= 0, got {self.pull_interval_s}"
            )
        if self.commit_mode not in COMMIT_MODES:
            raise ConfigurationError(
                f"commit_mode must be one of {', '.join(COMMIT_MODES)}, "
                f"got {self.commit_mode!r}"
            )
