import asyncio

from rich.console import Console

from dreadnode.agent.agent import Agent
from dreadnode.agent.tools.bloodhound.tool import BloodhoundTool

console = Console()


async def create_agent():
    return Agent(
        name="bloodhound-agent",
        description="An agent that uses Bloodhound to perform various tasks.",
        model="gpt-4",
        tools=[await BloodhoundTool()],
    )


async def main() -> None:
    agent = await create_agent()
    await agent.run("Given the current user, what paths are available to me?")


if __name__ == "__main__":
    asyncio.run(main())
