import typing as t
from pathlib import Path

from rich.console import Console

from dreadnode.agent.agent import Agent
from dreadnode.agent.tools.bbot.tool import BBotTool

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

    tool = await BBotTool.create()
    events = tool.run(
        targets=targets,
        presets=presets,
        modules=modules,
        flags=flags,
        config=config,
    )

    async for event in events:
        console.print(event)
        # Add your agent logic here to process events
        # if event == "FINDING":
        #     await agent.run(...)


# Usage
if __name__ == "__main__":
    app()
