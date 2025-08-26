import csv
import json
import typing as t
from pathlib import Path

import typing_extensions as te
from pydantic import BaseModel, ConfigDict, Field

from dreadnode.metric import Metric
from dreadnode.tracing.span import TaskSpan
from dreadnode.types import AnyDict, ErrorField

InputT = te.TypeVar("InputT", default=t.Any)
OutputT = te.TypeVar("OutputT", default=t.Any)

FileFormat = t.Literal["jsonl", "csv", "json", "yaml", "yml"]


def load_from_file(path: Path, *, file_format: FileFormat | None = None) -> list[AnyDict]:
    """
    Loads a list of objects from a file path, with support for JSONL, CSV, JSON, and YAML formats.

    Args:
        path: The path to the file to load.
        file_format: Optional format of the file. If not provided, it will be inferred from the file extension.

    Returns:
        A list of dictionaries representing the objects in the file.
    """
    path = Path(path)
    dataset: list[AnyDict] = []

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return dataset

    file_format = file_format or t.cast("FileFormat", path.suffix.lstrip(".").lower())
    if file_format not in t.get_args(FileFormat):
        raise ValueError(f"Unsupported file format: {file_format}")

    if file_format == "jsonl":
        dataset = [json.loads(line) for line in content.splitlines() if line.strip()]

    elif file_format == "csv":
        reader = csv.DictReader(content.splitlines())
        dataset = list(reader)

    elif file_format == "json":
        dataset = json.loads(content)
        if not isinstance(dataset, list):
            raise ValueError("JSON file must contain a list of objects.")

    elif file_format in {"yaml", "yml"}:
        try:
            import yaml  # type: ignore[import-untyped,unused-ignore]
        except ImportError as e:
            raise ImportError(
                "YAML support requires the 'PyYAML' package. Install with: pip install pyyaml"
            ) from e

        dataset = yaml.safe_load(content)
        if not isinstance(dataset, list):
            raise ValueError("YAML file must contain a list of objects.")

    return dataset


InputDataset = list[InputT]
InputDatasetProcessor = t.Callable[[InputDataset], InputDataset]


class Sample(BaseModel, t.Generic[InputT, OutputT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input: InputT
    output: OutputT | None = None

    metrics: dict[str, list[Metric]] = Field(default_factory=dict)
    assertions: dict[str, bool] = Field(default_factory=dict)

    error: ErrorField | None = Field(None, repr=False)
    task: TaskSpan[t.Any] | None = Field(None, repr=False, exclude=True)

    @property
    def passed(self) -> bool:
        return all(self.assertions.values()) if self.assertions else True

    def get_average_metric_value(self, key: str | None = None) -> float:
        metrics = (
            self.metrics.get(key, [])
            if key is not None
            else [m for ms in self.metrics.values() for m in ms]
        )
        return sum(metric.value for metric in metrics) / len(
            metrics,
        )

    @classmethod
    def from_task(cls, task: TaskSpan[OutputT]) -> "Sample[InputT, OutputT]":
        assertion_values: dict[str, list[float]] = {}
        for metric_name, metrics in task.metrics.items():
            for metric in metrics:
                assertion_name = getattr(metric, "_scorer_name", metric_name)
                if metric.attributes.get("assertion", False):
                    assertion_values.setdefault(assertion_name, []).append(metric.value)

        assertions = {name: any(values) for name, values in assertion_values.items()}

        return cls(
            input=t.cast("InputT", task.arguments.args[0] if task.arguments else None),
            output=task.outputs.get("output"),
            metrics=task.metrics,
            assertions=assertions,
            error=task.exception,
            task=task,
        )


class EvalResult(BaseModel, t.Generic[InputT, OutputT]):
    """
    Represents the result of an evaluation, including input, output, metrics, and error.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    samples: list[Sample[InputT, OutputT]] = Field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for s in self.samples if s.passed)

    @property
    def passed_samples(self) -> list[Sample[InputT, OutputT]]:
        return [s for s in self.samples if s.passed]
