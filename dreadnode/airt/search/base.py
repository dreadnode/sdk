import typing as t

import rigging as rg

from dreadnode.airt.attack import AttackConfig


@t.runtime_checkable
class Search(t.Protocol):
    """Protocol defining the search interface."""

    def run(self, config: AttackConfig) -> list[rg.Chat]:
        """Check if the search found a suitable example."""
