import asyncio

from rich import print

from dreadnode.agent import Agent, AgentResult
from dreadnode.agent.tools.ilspy.tool import ILSpyTool


async def main() -> AgentResult:
    agent = Agent(
        name="dotnet-reversing-agent",
        description="An agent that uses ILSpy to reverse engineer .NET binaries.",
        model="groq/moonshotai/kimi-k2-instruct",
        tools=[ILSpyTool(base_path="./bin", binaries=["addinutil.exe", "system.addin.dll"])],
        instructions="""You are an expert dotnet reverse engineer with decades of experience. Your task is to analyze the provided static binaries and identify high impact vulnerabilities. You care most about exploitable bugs from a remote perspective. It is okay to review the code multiple times,

        DO NOT write fixes or suggestions.
        DO NOT speculate, or make assumptions. Don't say could, might, maybe, or similar.
        DO NOT report encyption issues.
        DO NOT mock or pretend.
    """,
    )

    print(agent.all_tools)

    result = await agent.run(
        "You the tools available to you to. Here are the binaries: addinutil.exe, system.addin.dll.",
    )

    print(result.__dict__)


if __name__ == "__main__":
    asyncio.run(main())
