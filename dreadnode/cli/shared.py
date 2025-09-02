import typing as t
from dataclasses import dataclass

import cyclopts


@cyclopts.Parameter(name="dn", group="Dreadnode")
@dataclass
class DreadnodeArgs:
    server: str | None = None
    """Dreadnode server URL"""
    token: str | None = None
    """Dreadnode API token"""
    project: str | None = "bbot-agent"
    """Dreadnode project name"""
    profile: str | None = None
    """Dreadnode profile name"""
    console: t.Annotated[bool, cyclopts.Parameter(negative=False)] = False
    """Show span information in the console"""
    log_level: str = "INFO"
    """Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"""
