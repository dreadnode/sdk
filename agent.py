from pathlib import Path

from dreadnode.agent.agent import TaskAgent
from dreadnode.agent.hooks import summarize_when_long
from dreadnode.agent.tools import tool


@tool(truncate=1000, catch=True)
async def read_file(path: str) -> str:
    "Read the contents of a file."
    return (Path("../") / path).read_text()


agent = TaskAgent(
    name="basic",
    description="A basic agent that can handle simple tasks.",
    model="gpt-4o-mini",
    hooks=[summarize_when_long(max_tokens=1000)],
    tools=[read_file],
)
