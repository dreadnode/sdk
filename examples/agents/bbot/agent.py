from pathlib import Path

from rich.console import Console

import dreadnode as dn
from dreadnode.agent.agent import Agent
from dreadnode.agent.tools.bbot.tool import BBotTool

console = Console()

from cyclopts import App

app = App()

agent = Agent(
    name="bbot-agent",
    description="An agent that uses BBOT to perform various tasks.",
    model="meta-llama/llama-4-scout-17b-16e-instruct",
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
    config: str | None = None,
) -> None:
    if isinstance(targets, Path):
        with Path.open(targets) as f:
            loaded_targets = f.readlines()

    if not targets:
        console.print("[red]Error:[/red] No targets provided. Use --targets to specify targets.\n")
        return

    dn.configure(server="https://platform.dreadnode.io", project="bug-bounty-rea")

    with dn.run("scan-name", tags=presets):
        dn.log_params(
            targets=loaded_targets,
            presets=presets,
            modules=modules,
            flags=flags,
            config=config,
        )

        tool = await BBotTool.create()
        events = tool.run(
            targets=loaded_targets,
            presets=presets,
            modules=modules,
            flags=flags,
            config=config,
        )

        all_events = []

        async for event in events:
            console.print(event)
            all_events.append(event)
            # Add your agent logic here to process events
            # if event.type == "FINDING":
            #     await agent.run(...)

        for event in all_events:
            with dn.task_span(event.type):
                dn.log_output("event", event.json(siem_friendly=True))
                dn.log_metric(event.type, 1, mode="count", to="task-or-run")


# Usage
if __name__ == "__main__":
    app()
