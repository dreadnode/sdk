from __future__ import annotations
import typing as t
import copy, uuid
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
        parts = [
            f"agent={self.agent.name}",
            f"messages={len(self.messages)}",
            f"usage={self.usage}",
            f"steps={self.steps}",
        ]

        if self.failed:
            parts.append(f"failed={self.failed}")
        if self.error:
            parts.append(f"error={self.error}")

        return f"AgentResult({', '.join(parts)})"

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
        self._agent_result = agent_result

    def __iter__(self):
        for i in self._agent_result.messages:
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

    def __repr__(self):
        return str(self.info)

    @property
    def human(self) -> None:
        """ """
        import rich

        info = self.info
        s = "[grey66]Dreadnode Agent Dataset"
        s += f"\n [turquoise2]Agent:[white] {info["agent"]}\n [turquoise2]Failed: [white]{info["failed"]}\n [turquoise2]Error: [white]{info["error"]}\n [turquoise2]Steps: [white]{info["steps"]}\n [turquoise2]Messages: [white]{info["messages"]}\n"
        s += " [turquoise2]Usage:[white]"
        for k, v in info["usage"].items():
            s += f"\n  {k} - {v}"
        s += f"\n [turquoise2]Tool Calls: [white]{len(info["tool_calls"])}"
        s += "\n [turquoise2]Tool Call Summary:[white]"
        for k, v in info["tool_calls"].items():
            s += f"\n  {k} - {v}"

        rich.print(s)

    @property
    def info(self):
        if not getattr(self, "_info", False):
            info = dict(
                agent=self._agent_result.agent.name,
                failed=self._agent_result.failed,
                error=self._agent_result.error,
                steps=self._agent_result.steps,
                messages=len(self._agent_result.messages),
                usage=dict(self._agent_result.usage),
                tool_calls=dict(),
            )
            for i in self._agent_result.messages:
                if i.tool_calls and len(i.tool_calls) > 0:
                    for j in i.tool_calls:
                        if j.name not in info["tool_calls"]:
                            info["tool_calls"][j.name] = 0
                        info["tool_calls"][j.name] += 1

            self._info = info
        return copy.deepcopy(self._info)

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
        return f"/tmp/dn_agent_dataset-{self._agent_result.agent.name}-{str(uuid.uuid4())[:8]}"

    def _encode_tool_call(self, tool_call: "ToolCall") -> dict:
        """ """
        return dict(
            id=tool_call.id, name=tool_call.function.name, arguments=tool_call.arguments
        )
