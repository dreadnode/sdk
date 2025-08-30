from dreadnode.agent.agent import Agent
from dreadnode.agent.tools.mythic.apollo.tool import ApolloTool


async def create_agent() -> Agent:
    agent = Agent(
        name="apollo-agent",
        description="An agent that uses the Apollo toolset to interact with a Mythic C2 server.",
        model="gpt-4",
        tools=[
            await ApolloTool.create(
                username="admin",
                password="admin",
                server_url="http://localhost:7443",
                server_ip="localhost",
                server_port=7443,
            )
        ],
    )

    return agent


async def main() -> None:
    agent = await create_agent()
    agent.run("Create a new Apollo callback and execute a command to list directory contents.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
