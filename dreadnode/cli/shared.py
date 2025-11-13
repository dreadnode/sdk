import typing as t
from dataclasses import dataclass
from pathlib import Path

import cyclopts

from dreadnode.logging_ import LogLevel, configure_logging


@cyclopts.Parameter(name="dn", group="Dreadnode")
@dataclass
class DreadnodeConfig:
    server: str | None = None
    """Server URL"""
    token: str | None = None
    """API token"""
    organization: str | None = None
    """Organization name"""
    workspace: str | None = None
    """Workspace name"""
    project: str | None = None
    """Project name"""
    profile: str | None = None
    """Profile name"""
    console: t.Annotated[bool, cyclopts.Parameter(negative=False)] = False
    """Show spans in the console"""
    log_level: LogLevel | None = None
    """Console log level"""
    log_file: Path | None = None
    """File to log to"""

    def apply(self) -> None:
        from dreadnode import configure

        if self.log_level:
            configure_logging(self.log_level, self.log_file)

        configure(
            server=self.server,
            token=self.token,
            profile=self.profile,
            organization=self.organization,
            workspace=self.workspace,
            project=self.project,
            console=self.console,
        )
