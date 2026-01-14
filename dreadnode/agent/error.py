class MaxStepsError(Exception):
    """Raise from a hook to stop the agent's run due to reaching the maximum number of steps."""

    def __init__(self, max_steps: int):
        super().__init__(f"Maximum steps reached ({max_steps}).")
        self.max_steps = max_steps


class MaxToolCallsError(Exception):
    """Raise from a hook to stop the agent's run due to reaching the maximum number of tool calls."""

    def __init__(self, max_tool_calls: int):
        super().__init__(f"Maximum tool calls reached ({max_tool_calls}).")
        self.max_tool_calls = max_tool_calls
