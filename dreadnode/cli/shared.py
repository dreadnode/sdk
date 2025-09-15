import typing as t
from dataclasses import dataclass

import cyclopts

from dreadnode.logging_ import LogLevelLiteral, configure_logging


@cyclopts.Parameter(name="dn", group="Dreadnode")
@dataclass
class DreadnodeConfig:
    server: str | None = None
    """Server URL"""
    token: str | None = None
    """API token"""
    project: str | None = None
    """Project name"""
    profile: str | None = None
    """Profile name"""
    console: t.Annotated[bool, cyclopts.Parameter(negative=False)] = False
    """Show spans in the console"""
    log_level: LogLevelLiteral | None = None
    """Console log level"""

    def apply(self) -> None:
        from dreadnode import configure

        if self.log_level:
            configure_logging(self.log_level)

        configure(server=self.server, token=self.token, project=self.project, console=self.console)
