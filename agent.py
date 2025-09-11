from pathlib import Path

import dreadnode as dn
from dreadnode.agent.agent import TaskAgent
from dreadnode.agent.hooks import summarize_when_long
from dreadnode.agent.tools import tool


@tool(truncate=1000, catch=True)
async def read_file(path: str, *, max_length: int = dn.Config(123)) -> str:
    "Read the contents of a file."
    return (Path("../") / path).read_text()


agent = TaskAgent(
    name="basic",
    description="A basic agent that can handle simple tasks.",
    model="gpt-4o-mini",
    hooks=[summarize_when_long()],
    tools=[dn.agent.tools.fs.Filesystem()],
)
