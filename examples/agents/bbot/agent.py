import asyncio

from rich.console import Console

from dreadnode.agent.agent import Agent
from dreadnode.agent.tools.bbot.tool import BBotTool

console = Console()


agent = Agent(
    name="bbot-agent",
    description="An agent that uses BBOT to perform various tasks.",
    model="gpt-4",
)


async def main() -> None:
    agent = await BBotTool.create()
    agent.get_presets()
    agent.get_modules()
    console.print(f"BBOT Tool created with name: {agent.name}")


# Usage
if __name__ == "__main__":
    asyncio.run(main())
