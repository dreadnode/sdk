class MaxTurnsReachedError(Exception):
    """Raise from a hook to stop the agent's run due to reaching the maximum number of turns."""

    def __init__(self, max_turns: int):
        super().__init__(f"Maximum turns reached ({max_turns}).")
        self.max_turns = max_turns


class AgentStop(Exception):  # noqa: N818
    """Raise from a hook to cleanly terminate the agent's run with a result."""

    def __init__(self, reason: str):
        super().__init__(f"Agent run was stopped: {reason}")
        self.reason = reason
