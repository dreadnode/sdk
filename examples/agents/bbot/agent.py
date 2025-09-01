from pathlib import Path

import ray
from cyclopts import App
from ray import serve
from ray.actor import ActorClass
from rich.console import Console

from dreadnode.agent.tools.bbot.tool import BBotTool

console = Console()

app = App()


@app.command
async def modules() -> None:
    BBotTool.get_modules()


@app.command
def presets() -> None:
    BBotTool.get_presets()


@app.command
async def flags() -> None:
    BBotTool.get_flags()


@app.command
async def events() -> None:
    BBotTool.get_events()


@app.command
async def scan(
    targets: Path,
    presets: list[str] | None = None,
    modules: list[str] | None = None,
    flags: list[str] | None = None,
    config: str | None = None,
) -> None:
    if isinstance(targets, Path):
        with Path.open(targets) as f:
            loaded_targets = f.read().splitlines()

    if not targets:
        console.print("[red]Error:[/red] No targets provided. Use --targets to specify targets.\n")
        return

    ray.init(address="auto")

    scan_futures = []

    ActorDemoRay: ActorClass[BBotTool] = ray.remote(BBotTool)
    BBotApplication = serve.deployment(BBotTool, name="BBotService", num_replicas=1)

    app = BBotApplication.bind()
    serve.run(app, name="BBotService", route_prefix="/bbot")

    handle = serve.get_deployment("BBotService").get_handle()

    for target in loaded_targets:
        if not target:
            continue

        result_generator_ref = await handle.run.remote(target=target, presets=["web-screenshots"])

        # Create an actor instance specifically for this one target
        actor = ActorDemoRay.remote(
            targets=[target],  # Pass the target in a list
            presets=presets or [],
            modules=modules or [],
            flags=flags or [],
            config={},
        )

        scan_futures.append(actor.run.remote())

    console.print(f"[*] Starting BBOT scan on {len(loaded_targets)} targets...")

    while scan_futures:
        done, scan_futures = ray.wait(scan_futures, num_returns=1)
        actor = done[0]

        result_generator = ray.get(done[0])

        async for event in result_generator:
            console.print(await event)


console.print("[*] Scan complete.")

# tool = await BBotTool.create(
#     targets=loaded_targets, presets=presets, modules=modules, flags=flags, config=config
# )

# dn.configure(server="https://platform.dreadnode.io", project="bount-rea-2")

# with dn.run(tool.scan.name, tags=presets):
#     for i, j in enumerate(loaded_targets):
#         dn.log_param(f"target_{i}", j.strip())

#     for p in presets:
#         dn.log_param("preset", p)

#     dn.log_param(
#         # targets=loaded_targets,
#         # presets=presets,
#         # modules=modules,
#         # flags=flags,
#         # config=config,
#         "scan",
#         tool.scan.id,
#     )

#     events = tool.run()

#     async for event in events:
#         with dn.task_span(event.type):
#             df = pd.json_normalize(event.json()).set_index("type")
#             log = event.json(siem_friendly=True)
#             log2 = event.json()
#             console.print(df)
#             dn.log_output("event", log)
#             dn.log_output("event-siem", log2)
#             dn.log_outputs(log)
#             dn.log_outputs(log2)

#             dn.log_metric(event.type, 1, mode="count", to="run")
#             # Add your agents here to process events
#             if event.type == "WEBSCREENSHOT":
#                 console.print(df)

#                 image_path = f"{tool.scan.core.scans_dir}/{tool.scan.name}/{event.data['path']}"
#                 console.print(event.json())
#                 console.print(
#                     f"[bold green]Web Screenshot saved to:[/bold green] {event.data['path']}"
#                 )
#                 dn.log_output(
#                     "webscreenshot",
#                     dn.Image(image_path),
#                 )
#                 dn.log_artifact(image_path)
#         #     await agent.run(...)


# Usage
if __name__ == "__main__":
    app()
