from bbot import Preset, Scanner
from pydantic import Field
from rich.console import Console

from dreadnode.agent.tools import Toolset

console = Console()


class BBotTool(Toolset):
    scanner: Scanner | None = Field(default=None, exclude=True)

    @classmethod
    async def create(cls, name: str = "bbot-agent", **kwargs: dict) -> "BBotTool":
        """Factory method to create and initialize a BBOT Tool."""
        try:
            instance = cls(name=name, **kwargs)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to create BBOT Tool with name '{name}': {e}") from e

        return instance

    def get_presets(self) -> list[str] | None:
        """Return the presets available in the BBOT Agent."""
        preset = Preset(_log=True, name="bbot_cli_main")

        console.print(preset.all_presets.keys())
        return list(preset.all_presets.keys())

    def get_modules(self) -> list[str] | None:
        """Return the modules available in the BBOT Agent."""
        preset = Preset(_log=True, name="bbot_cli_main")

        console.print(preset.module_loader.all_module_choices)
        console.print(preset.module_loader.output_module_choices)

        console.print(preset.module_loader.flag_choices)
        return None
