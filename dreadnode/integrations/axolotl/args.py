"""
Pydantic args for Axolotl config validation.

These fields are merged into the main Axolotl config schema when the
DreadnodePlugin is enabled.
"""

from pydantic import BaseModel, Field


class DreadnodeAxolotlArgs(BaseModel):
    """
    Configuration options for Dreadnode integration with Axolotl.

    These fields are merged into the main Axolotl config schema.
    """

    dreadnode_project: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Dreadnode project name. If not set, Dreadnode logging is disabled."
        },
    )

    dreadnode_run_name: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Name for this training run. Defaults to Axolotl's run_name if not specified."
        },
    )

    dreadnode_tags: list[str] | None = Field(
        default=None,
        json_schema_extra={"description": "Tags to associate with this run in Strikes."},
    )
