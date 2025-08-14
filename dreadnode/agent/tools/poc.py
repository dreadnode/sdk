from dreadnode import log_metric, log_output, tag
from dreadnode.agent.tools.base import tool
from dreadnode.data_types import Markdown


@tool(
    name="Create Proof of Concept",
    description="Save a Proof of Concept (PoC) for a vulnerability.",
)
async def create_proof_of_concept(
    proof_of_concept: str,
) -> str:
    """
    Save a Proof of Concept (PoC) for a vulnerability.
    """

    log_output(
        "proof_of_concept",
        Markdown(f"### Proof of Concept\n\n{proof_of_concept}"),
    )
    tag("poc", to="run")
    log_metric("num_pocs", 1, mode="count")

    return "Successfully saved PoC"
