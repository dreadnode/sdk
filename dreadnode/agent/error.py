class MaxStepsError(Exception):
    """Raise from a hook to stop the agent's run due to reaching the maximum number of steps."""

    def __init__(self, max_steps: int):
        super().__init__(f"Maximum steps reached ({max_steps}).")
        self.max_steps = max_steps
