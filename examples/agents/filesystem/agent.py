from dreadnode.agent.agent import Agent
from dreadnode.agent.tools.filesystem.tool import FilesystemTool

agent = Agent(
    name="filesystem-agent",
    description="An agent that uses filesystem tools to perform various tasks.",
    model="gpt-4",
    tools=[FilesystemTool(path="/")],
)


async def main() -> None:
    agent.run("List the files in the root directory.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
