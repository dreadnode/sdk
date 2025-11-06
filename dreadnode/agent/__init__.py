from pydantic.dataclasses import rebuild_dataclass

from dreadnode.agent import error, events, hooks, reactions, result, stop, tools
from dreadnode.agent.agent import Agent, TaskAgent
from dreadnode.agent.hooks import Hook
from dreadnode.agent.reactions import Continue, Fail, Finish, Reaction, Retry, RetryWithFeedback
from dreadnode.agent.result import AgentResult
from dreadnode.agent.thread import Thread
from dreadnode.agent.tools import tool, tool_method

Agent.model_rebuild()
Thread.model_rebuild()

rebuild_dataclass(AgentResult)  # type: ignore[arg-type]

__all__ = [
    "Agent",
    "AgentResult",
    "Continue",
    "Fail",
    "Finish",
    "Hook",
    "Reaction",
    "Retry",
    "RetryWithFeedback",
    "TaskAgent",
    "Thread",
    "error",
    "events",
    "hooks",
    "reactions",
    "result",
    "stop",
    "tool",
    "tool_method",
    "tools",
]
