import typing as t

from dreadnode import log_metric, log_output, log_param, tag
from dreadnode.agent.tools.base import tool
from dreadnode.data_types import Markdown


@tool(
    name="Report auth material",
    description="Report authentication material such as hardcoded keys, tokens, or passwords.",
)
async def report_auth_material(
    auth_material: t.Annotated[str, "The Markdown details or code that uses the auth material"],
) -> str:
    log_output("auth_material", Markdown(f"### Auth Material\n\n{auth_material}"))
    log_metric("auth_material", 1, mode="count", to="run")
    log_param("auth_material", auth_material, to="run")
    tag("creds", to="run")
    return "Auth material reported"
