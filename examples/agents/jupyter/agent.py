from dreadnode.agent.agent import Agent
from dreadnode.agent.tools.jupyter.tool import PythonKernel

agent = Agent(
    name="code-agent",
    description="An agent that uses a Python kernel to perform coding tasks.",
    model="gpt-4",
    tools=[PythonKernel()],
)


async def main() -> None:
    agent.run("Write a Python function that returns the Fibonacci sequence up to n.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
