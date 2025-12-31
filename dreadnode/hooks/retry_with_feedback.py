from dreadnode.core.agents.events import AgentStalled
from dreadnode.core.agents.reactions import RetryWithFeedback
from dreadnode.core.hook import Hook, hook


@hook(AgentStalled)
def retry_with_feedback(
    feedback: str,
) -> Hook:
    """
    Create a hook that provides feedback when the specified event occurs.

    Args:
        event_type: The type of event to listen for, or a callable that returns True if feedback should be provided.
        feedback: The feedback message to provide when the event occurs.

    Returns:
        A hook that provides feedback when the event occurs.
    """

    return RetryWithFeedback(feedback=feedback)
