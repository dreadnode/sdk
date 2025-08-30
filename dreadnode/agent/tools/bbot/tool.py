import typing as t

from bbot import Preset, Scanner
from pydantic import Field
from rich.console import Console

from dreadnode.agent.tools import Toolset, tool_method

from .dispatcher import Dispatcher
from .utils import events_table, flags_table, modules_table, presets_table

console = Console()


class BBotTool(Toolset):
    tool_name: str = Field(default="bbot-agent", description="Name of the BBOT Tool")
    targets: list[str] = Field(default_factory=list, description="Targets to scan with BBOT")
    modules: list[str] = Field(default_factory=list, description="Modules to run with BBOT")
    presets: list[str] = Field(default_factory=list, description="Presets to use with BBOT")
    flags: list[str] = Field(default=None, description="Flags to enable module groups")
    config: dict[str, t.Any] = Field(
        default=None, description="Custom config options in key=value format"
    )
    scan: Scanner = Field(default=None, description="The BBOT Scanner instance")
    scan_timeout: int = Field(default=300, description="Timeout for BBOT scans in seconds")

    dispatcher: Dispatcher = Field(default_factory=Dispatcher, exclude=True)

    @classmethod
    async def create(
        cls,
        targets: list[str],
        modules: list[str] | None = None,
        presets: list[str] | None = None,
        flags: list[str] | None = None,
        config: dict[str, t.Any] | None = None,
    ) -> "BBotTool":
        """Factory method to create and initialize a BBOT Tool."""
        try:
            instance = cls(
                targets=targets,
                modules=modules or [],
                presets=presets or [],
                flags=flags or [],
                config=config or {},
            )
            instance.scan = Scanner(
                *instance.targets,
                modules=instance.modules,
                presets=instance.presets,
                flags=instance.flags,
                config=instance.config,
                dispatcher=instance.dispatcher,
            )
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to create BBOT Tool with tool_name '{instance.tool_name}': {e}"
            ) from e

        return instance

    @tool_method()
    def get_presets(self) -> None:
        """Return the presets available in the BBOT Agent."""

        preset = Preset(_log=True, name="bbot_cli_main")
        console.print(presets_table(preset))

    @tool_method()
    def get_modules(self) -> None:
        """Return the modules available in the BBOT Agent."""
        preset = Preset(_log=True, name="bbot_cli_main")
        console.print(modules_table(preset.module_loader))

    @tool_method()
    def get_flags(self) -> None:
        """Return the output modules available in the BBOT Agent."""
        preset = Preset(_log=True, name="bbot_cli_main")
        console.print(flags_table(preset.module_loader))

    @tool_method()
    def get_events(self) -> None:
        """Return the flags available in the BBOT Agent."""
        preset = Preset(_log=True, name="bbot_cli_main")
        console.print(events_table(preset.module_loader))

    async def run(self) -> str:
        r"""
        Executes a BBOT scan against the specified targets.

        This is the primary action tool. It assembles and runs a `bbot` command

        Args:
            targets: REQUIRED. Targets to scan (e.g., ['example.com']).
            modules: Modules to run (e.g., ['httpx', 'nuclei']).
            presets: Presets to use (e.g., ['subdomain-enum', 'web-basic']).
            flags: Flags to enable module groups (e.g., ['passive', 'safe']).
            config: Custom config options in key=value format (e.g., ['modules.httpx.timeout=5']).
            extra_args: An array of strings for any other `bbot` CLI flags. This is the escape hatch
                        for advanced usage. For example:
                        ['--strict-scope']
                        ['-ef aggressive --allow-deadly']
                        ['--proxy http://127.0.0.1:8080']

        Returns:
            The standard output from the bbot command, summarizing the scan.
        """
        if not self.targets:
            raise ValueError("At least one target is required to run a scan.")

        async for event in self.scan.async_start():
            yield event
