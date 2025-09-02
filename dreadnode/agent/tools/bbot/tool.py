import typing as t

import rich

from dreadnode.agent.tools.base import Toolset

from .utils import events_table, flags_table, modules_table, presets_table


class BBotTool(Toolset):
    from bbot import Preset, Scanner

    @staticmethod
    def get_presets() -> None:
        """Return the presets available in the BBOT Agent."""

        preset = Preset(_log=True, name="bbot_cli_main")
        rich.print(presets_table(preset))

    @staticmethod
    def get_modules() -> None:
        """Return the modules available in the BBOT Agent."""
        preset = Preset(_log=True, name="bbot_cli_main")
        rich.print(modules_table(preset.module_loader))

    @staticmethod
    def get_flags() -> None:
        """Return the output modules available in the BBOT Agent."""
        preset = Preset(_log=True, name="bbot_cli_main")
        rich.print(flags_table(preset.module_loader))

    @staticmethod
    def get_events() -> None:
        """Return the flags available in the BBOT Agent."""
        preset = Preset(_log=True, name="bbot_cli_main")
        rich.print(events_table(preset.module_loader))

    async def run(
        self,
        target: str,
        modules: list[str] | None = None,
        presets: list[str] | None = None,
        flags: list[str] | None = None,
        config: dict[str, t.Any] | None = None,
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
        self._scan = Scanner(
            *[target],
            modules=modules,
            presets=presets,
            flags=flags,
            config=config,
        )

        async for event in self._scan.async_start():
            yield event.json(siem_friendly=True)
