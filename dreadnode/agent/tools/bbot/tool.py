import typing as t

from bbot import Preset, Scanner
from pydantic import BaseModel, Field, PrivateAttr
from rich.console import Console

from dreadnode.agent.tools.base import Toolset

from .utils import events_table, flags_table, modules_table, presets_table

console = Console()


class BBotArgs(BaseModel):
    target: str = Field(default_factory=str, description="Target to scan with BBOT")
    modules: list[str] = Field(default_factory=list, description="Modules to run with BBOT")
    presets: list[str] = Field(default_factory=list, description="Presets to use with BBOT")
    flags: list[str] = Field(default_factory=list, description="Flags to enable module groups")
    config: dict[str, t.Any] = Field(default_factory=dict, description="Custom config options")
    extra_args: list[str] = Field(
        default_factory=list,
        description=(
            "Additional command-line arguments for BBOT. "
            "This allows for advanced usage and customization."
        ),
    )


class BBotTool(Toolset):
    _scan: Scanner | None = PrivateAttr(default=None)

    @staticmethod
    def get_presets() -> None:
        """Return the presets available in the BBOT Agent."""

        preset = Preset(_log=True, name="bbot_cli_main")
        console.print(presets_table(preset))

    @staticmethod
    def get_modules() -> None:
        """Return the modules available in the BBOT Agent."""
        preset = Preset(_log=True, name="bbot_cli_main")
        console.print(modules_table(preset.module_loader))

    @staticmethod
    def get_flags() -> None:
        """Return the output modules available in the BBOT Agent."""
        preset = Preset(_log=True, name="bbot_cli_main")
        console.print(flags_table(preset.module_loader))

    @staticmethod
    def get_events() -> None:
        """Return the flags available in the BBOT Agent."""
        preset = Preset(_log=True, name="bbot_cli_main")
        console.print(events_table(preset.module_loader))

    def run(
        self,
        target: str,
        modules: list[str] | None = None,
        presets: list[str] | None = None,
        flags: list[str] | None = None,
        config: dict[str, t.Any] | None = None,
        extra_args: list[str] | None = None,
    ) -> t.AsyncGenerator[t.Any, None]:
        """
        Executes a BBOT scan against the specified targets.

        This is the primary action tool. It assembles and runs a `bbot` command.

        Args:
            targets: REQUIRED. A list of targets to scan (e.g., ['example.com']).
            modules: A list of modules to run (e.g., ['httpx', 'nuclei']).
            presets: A list of presets to use (e.g., ['subdomain-enum', 'web-basic']).
            flags: A list of flags to enable module groups (e.g., ['passive', 'safe']).
            config: A dictionary of custom config options (e.g., {"modules.httpx.timeout": 5}).
            extra_args: A list of strings for any other `bbot` CLI flags.
                        For example: ['--strict-scope', '--proxy http://127.0.0.1:8080']

        Returns:
            An async generator that yields JSON-formatted scan events.
        """
        args = BBotArgs(
            target=target,
            modules=modules or [],
            presets=presets or [],
            flags=flags or [],
            config=config or {},
            extra_args=extra_args or [],
        )
        return self._run_scan(args)

    async def _run_scan(self, args: BBotArgs) -> t.AsyncGenerator[t.Any, None]:
        """The internal scan logic that operates on a validated BBotArgs model."""
        if not args.target:
            raise ValueError("At least one target is required to run a scan.")

        self._scan = Scanner(
            *args.target,
            modules=args.modules,
            presets=args.presets,
            flags=args.flags,
            config=args.config,
        )

        async for event in self._scan.async_start():
            yield event.json(siem_friendly=True)
