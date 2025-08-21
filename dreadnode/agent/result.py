from __future__ import annotations
import typing as t
import json
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from rigging.generator.base import Usage
from rigging.message import Message

if t.TYPE_CHECKING:
    from dreadnode.agent.agent import Agent


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class AgentResult:
    agent: "Agent"
    messages: list[Message]
    usage: Usage
    steps: int
    failed: bool
    error: Exception | str | None

    def __repr__(self) -> str:
        return (
            f"AgentResult(agent={self.agent.name}, "
            f"messages={len(self.messages)}, "
            f"usage={self.usage}, "
            f"steps={self.steps}, "
            f"failed={self.failed}, "
            f"error={self.error})"
        )

    @property
    def dataset(self):
        if not getattr(self, "_dataset", False):
            self._dataset = AgentDataset(self)
        return self._dataset


class AgentDataset:

    _fields = [
        "uuid",
        "role",
        "content",
        "slices",
        "tool_call_id",
        "tool_calls",
        "metadata",
    ]

    def __init__(self, agent_result: "AgentResult"):
        self.agent_result = agent_result

    def __iter__(self):
        for i in self.agent_result.messages:
            yield dict(
                uuid=str(i.uuid),
                role=i.role,
                content=i.content,
                slices=i.slices,
                tool_call_id=i.tool_call_id,
                tool_calls=(
                    [self._encode_tool_call(j) for j in i.tool_calls]
                    if i.tool_calls
                    else []
                ),
                metadata=i.metadata,
            )

    @property
    def fields(self) -> list[str]:
        """ """
        return [i for i in self._fields]

    def csv(self, filepath: str) -> str:
        """ """
        import csv

        if not filepath:
            filepath = f"{self._get_dir_path()}.csv"

        with open(filepath, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.fields)
            writer.writeheader()
            for i in self:
                writer.writerow(i)

        return filepath

    def json(self, filepath: str) -> str:
        """ """
        import json

        if not filepath:
            filepath = f"{self._get_dir_path()}.json"

        json.dump(dict(messages=list(self)), open(filepath, "w"), indent=4)

        return filepath

    def dataframe(self) -> "pd.DataFrame":
        """ """
        import pandas as pd

        return pd.DataFrame(data=list(self))

    def pytorch(self):
        pass

    def _get_dir_path(self):
        return f"/tmp/dn_agent_dataset-{self.agent_result.agent.name}-{str(uuid.uuid4())[:8]}"

    def _encode_tool_call(self, tool_call: "ToolCall") -> dict:
        """ """
        return dict(
            id=tool_call.id, name=tool_call.function.name, arguments=tool_call.arguments
        )
