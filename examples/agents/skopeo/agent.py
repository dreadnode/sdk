from dreadnode.agent.agent import Agent
from dreadnode.agent.tools.skopeo.tool import SkopeoTool

# More often than not, you'll want to run the tool to get outputs
# and have the agent use those outputs as context.
tool = SkopeoTool()


agent = Agent(
    name="skopeo-agent",
    description="An agent that uses Skopeo to inspect Microsoft Container Registry images.",
    model="gpt-4",
    tools=[SkopeoTool()],
)


async def main() -> None:
    agent.run("List the files in the latest mcr.microsoft.com/dotnet/aspnet image.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
