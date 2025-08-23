from bbot import Preset, Scanner
from pydantic import Field
from rich import print

from dreadnode.agent.tools import Toolset


class BBotTool(Toolset):
    scanner: Scanner | None = Field(default=None, exclude=True)

    @classmethod
    async def create(cls, name: str = "bbot-agent", **kwargs) -> "BBotTool":
        """Factory method to create and initialize a BBOT Tool."""
        try:
            instance = cls(**kwargs)
            instance.name = name
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to create BBOT Tool with name '{name}': {e}") from e

        return instance

    def get_presets(self) -> list[str] | None:
        """Return the presets available in the BBOT Agent."""
        preset = Preset(_log=True, name="bbot_cli_main")

        print(preset.all_presets.keys())
        return list(preset.all_presets.keys())

    def get_modules(self) -> list[str] | None:
        """Return the modules available in the BBOT Agent."""
        preset = Preset(_log=True, name="bbot_cli_main")

        print(preset.module_loader.all_module_choices)
        print(preset.module_loader.output_module_choices)

        print(preset.module_loader.flag_choices)
        return None


# Usage
if __name__ == "__main__":
    import asyncio

    async def main():
        # Try creating with minimal arguments first
        agent = await BBotTool.create()
        agent.get_presets()
        agent.get_modules()
        print(f"BBOT Tool created with name: {agent.name}")

    asyncio.run(main())
