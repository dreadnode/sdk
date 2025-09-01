import asyncio
from pathlib import Path

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from dreadnode.agent.agent import Agent
from dreadnode.agent.result import AgentResult
from dreadnode.agent.tools.ilspy.tool import ILSpyTool

console = Console()


async def main() -> AgentResult:
    agent = Agent(
        name="dotnet-reversing-agent",
        description="An agent that uses ILSpy to reverse engineer .NET binaries.",
        model="groq/moonshotai/kimi-k2-instruct",
        tools=[ILSpyTool.from_path(path=str(Path(__file__).parent / "bin"))],
        instructions="""You are an expert dotnet reverse engineer with decades of experience. Your task is to analyze the provided static binaries and identify high impact vulnerabilities using the tools available to you. You care most about exploitable bugs from a remote perspective. It is okay to review the code multiple times,

        - DO NOT write fixes or suggestions.
        - DO NOT speculate, or make assumptions. Don't say could, might, maybe, or similar.
        - DO NOT report encyption issues.
        - DO NOT mock or pretend.
    """,
    )

    result = await agent.run(
        "Analyze the assemblies and report critical and high impact vulnerabilities that result in code execution. Please find an entry point and summarize the call flow to the vulnerability as markdown. Create a Mermaid diagram of the call flow.",
    )

    # Post-run
    console.print(
        Panel(
            Markdown(result.messages[-1].content),
            title="Response",
            subtitle="powered by dreadnode",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
            expand=False,
        )
    )

    return result


if __name__ == "__main__":
    asyncio.run(main())
