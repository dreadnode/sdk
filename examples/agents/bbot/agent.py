from pathlib import Path

from rich.console import Console

import dreadnode as dn
from dreadnode.agent.agent import Agent
from dreadnode.agent.tools.bbot.tool import BBotTool
from dreadnode.agent.tools.kali.tool import KaliTool

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

    tool = await BBotTool.create(
        targets=loaded_targets, presets=presets, modules=modules, flags=flags, config=config
    )

    dn.configure(server="https://platform.dreadnode.io", project="bount-rea")

    with dn.run(tool.scan.name, tags=presets):
        dn.log_params(
            targets=loaded_targets,
            presets=presets,
            modules=modules,
            flags=flags,
            config=config,
            scan=tool.scan.id,
        )

        events = tool.run()

        async for event in events:
            console.print(event)
            with dn.task_span(event.type):
                dn.log_output("event", event.json(siem_friendly=True))
                dn.log_metric(event.type, 1, mode="count", to="run")
                # Add your agent logic here to process events
                if event.type == "WEBSCREENSHOT":
                    image_path = f"{tool.scan.core.scans_dir}/{tool.scan.name}/{event.data['path']}"
                    console.print(event.json())
                    console.print(
                        f"[bold green]Web Screenshot saved to:[/bold green] {event.data['path']}"
                    )
                    dn.log_output(
                        "webscreenshot",
                        dn.Image(image_path),
                    )
                    dn.log_artifact(image_path)
            #     await agent.run(...)


# Usage
if __name__ == "__main__":
    app()
