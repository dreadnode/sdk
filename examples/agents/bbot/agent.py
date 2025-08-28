import typing as t
from pathlib import Path

from rich.console import Console

from dreadnode.agent.agent import Agent
from dreadnode.agent.tools.bbot.tool import BBotTool
from dreadnode.agent.tools.kali.tool import KaliTool

console = Console()

from cyclopts import App

app = App()


agent = Agent(
    name="bbot-agent",
    description="An agent that uses BBOT to perform various tasks.",
    model="gpt-4",
)


@app.command
async def modules() -> None:
    tool = await BBotTool.create()
    tool.get_modules()


@app.command
async def presets() -> None:
    tool = await BBotTool.create()
    tool.get_presets()


@app.command
async def flags() -> None:
    tool = await BBotTool.create()
    tool.get_flags()


@app.command
async def events() -> None:
    tool = await BBotTool.create()
    tool.get_events()


@app.command
async def scan(
    targets: Path | None = None,
    presets: list[str] | None = None,
    modules: list[str] | None = None,
    flags: list[str] | None = None,
    config: Path | dict[str, t.Any] | None = None,
) -> None:
    if isinstance(targets, Path):
        with Path.open(targets) as f:
            targets = f.readlines()

    if not targets:
        console.print("[red]Error:[/red] No targets provided. Use --targets to specify targets.\n")
        return

    auth_agent = Agent(
        name="auth-brute-forcer",
        description="Performs credential stuffing, password sprays and brute force attacks on login pages",
        model="groq/moonshotai/kimi-k2-instruct",
        tools=[BBotTool(), KaliTool()],
        instructions="""You are an expert at credential testing and authentication bypass.

        When you find login pages and authentication services, your job is to:
        1. Identify the login form and authentication mechanism
        2. Test common default credentials using the tools and wordlists provided
        3. Suggest any additional required brute force attack strategies
        4. Report successful authentications, interesting findings or errors encountered worth noting

        IMPORTANT: Don't just suggest strategies - actually execute credential testing using your available tools.
        Be systematic and thorough in your credential testing approach.
        """,
    )

    tool = await BBotTool.create()
    events = tool.run(
        targets=targets,
        presets=presets,
        modules=modules,
        flags=flags,
        config=config,
    )

    async for event in events:
        """
        Handle each event emitted by the BBOT scan.
        Run 'uv run python examples/agents/bbot/agent.py events' to see event types.
        """
        console.print(event)
        ### Add your agent logic here to process events ###

        # Check for login pages and trigger credential testing
        if event.type == "URL" and any(
            keyword in str(event.data).lower() for keyword in ["login", "signin", "admin", "auth"]
        ):
            try:
                console.print(f"Testing credentials on: {event.data}")
                result = await auth_agent.run(
                    f"Perform credential testing on the login page at {event.data}. "
                    "Use the available tools to test common default credentials and brute force attacks. "
                    "Report any successful authentications or interesting findings."
                )
                console.print(f"[Results:\n{result}")
            except Exception as e:
                console.print(f"Error in auth agent: {e}")


# Usage
if __name__ == "__main__":
    app()
