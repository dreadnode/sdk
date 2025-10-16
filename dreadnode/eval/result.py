import json
import statistics
import typing as t
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import typing_extensions as te

from dreadnode.eval.sample import Sample
from dreadnode.util import format_dict

In = te.TypeVar("In", default=t.Any)
Out = te.TypeVar("Out", default=t.Any)

EvalStopReason = t.Literal["finished", "max_errors_reached", "max_consecutive_errors_reached"]


@te.runtime_checkable
class HasSamples(te.Protocol[In, Out]):
    @property
    def samples(self) -> list["Sample[In, Out]"]: ...


class EvalResultMixin:
    """A mixin providing a common statistical interface for evaluation results."""

    @property
    def passed_count(self: "HasSamples") -> int:
        """The number of samples that passed all assertions."""
        return sum(1 for s in self.samples if s.passed)

    @property
    def failed_count(self: "HasSamples") -> int:
        """The number of samples that failed any assertions."""
        return sum(1 for s in self.samples if not s.passed)

    @property
    def passed_samples(self: "HasSamples") -> list["Sample[In, Out]"]:
        """A list of all samples that passed all assertions."""
        return [s for s in self.samples if s.passed]

    @property
    def error_samples(self: "HasSamples") -> list["Sample[In, Out]"]:
        """A list of all samples that encountered an error during processing."""
        return [s for s in self.samples if s.error is not None]

    @property
    def error_count(self: "HasSamples") -> int:
        """The number of samples that encountered an error during processing."""
        return sum(1 for s in self.samples if s.error is not None)

    @property
    def failed_samples(self: "HasSamples") -> list["Sample[In, Out]"]:
        """A list of all samples that failed at least one assertion."""
        return [s for s in self.samples if not s.passed]

    @property
    def pass_rate(self: "HasSamples") -> float:
        """The overall pass rate of the evaluation, from 0.0 to 1.0."""
        _samples = self.samples
        if not _samples:
            return 0.0
        passed_count = sum(1 for s in self.samples if s.passed)
        return passed_count / len(_samples)

    @property
    def metrics(self: "HasSamples") -> dict[str, list[float]]:
        """
        Returns a breakdown of all metric values across all samples.

        Returns:
            A dictionary where keys are metric names and values are lists of
            metric values, with each value corresponding to a sample from the
            evaluation dataset.
        """
        breakdown: defaultdict[str, list[float]] = defaultdict(list)

        for sample in self.samples:
            for name, metric_list in sample.metrics.items():
                # NOTE(nick): Originally we were including internal
                # values (metrics reported multiple times within the same sample),
                # but I think it's safer to just take the last metric here.

                # for metric in metric_list:
                #     breakdown[name].append(metric.value)

                if metric_list:
                    breakdown[name].append(metric_list[-1].value)

        return dict(breakdown)

    @property
    def metrics_summary(self) -> dict[str, dict[str, float]]:
        """
        Calculates and returns a summary of statistics for each metric across all samples.
        """
        summary: dict[str, dict[str, float]] = {}
        for name, values in self.metrics.items():  # type: ignore[misc]
            if not values:
                continue

            summary[name] = {
                "mean": statistics.mean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }

        return summary

    @property
    def metrics_aggregated(self) -> dict[str, float]:
        """
        Aggregates metrics across all samples by calculating the mean for each metric.

        Returns:
            A dictionary where keys are metric names and values are the mean
            of that metric across all samples.
        """
        return {name: stats["mean"] for name, stats in self.metrics_summary.items()}

    @property
    def assertions_summary(self: "HasSamples") -> dict[str, dict[str, float | int]]:
        """
        Calculates and returns a summary for each assertion across all samples.

        Returns:
            A dictionary where each key is an assertion name and the value is
            another dictionary containing 'passed_count', 'failed_count', and 'pass_rate'.
        """
        assertions_results: dict[str, list[bool]] = defaultdict(list)
        for sample in self.samples:
            for name, passed in sample.assertions.items():
                assertions_results[name].append(passed)

        summary: dict[str, dict[str, float | int]] = {}
        for name, results in assertions_results.items():
            if not results:
                continue

            passed_count = sum(1 for r in results if r)
            total_count = len(results)
            pass_rate = passed_count / total_count if total_count > 0 else 0.0

            summary[name] = {
                "passed_count": passed_count,
                "failed_count": total_count - passed_count,
                "pass_rate": pass_rate,
            }
        return summary

    def to_dicts(self: "HasSamples") -> list[dict[str, t.Any]]:
        """
        Flattens the results into a list of dictionaries, where each
        dictionary represents a single sample with all its context.
        """
        return [sample.to_dict() for sample in self.samples]

    def to_dataframe(self) -> "pd.DataFrame":
        """
        Converts the results into a pandas DataFrame for analysis.
        """
        return pd.DataFrame(self.to_dicts())  # type: ignore[misc]

    def to_jsonl(self, path: str | Path) -> None:
        """
        Saves the results to a JSON Lines (JSONL) file.
        """
        records = self.to_dicts()  # type: ignore[misc]
        with Path(path).open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(record) + "\n" for record in records)


@dataclass
class IterationResult(EvalResultMixin, t.Generic[In, Out]):
    """The result of a single iteration over the dataset for a given scenario."""

    iteration: int
    """The iteration number for this result."""
    samples: list[Sample[In, Out]] = field(default_factory=list)
    """A list of samples for this iteration."""

    def __repr__(self) -> str:
        parts: list[str] = [
            f"iteration={self.iteration}",
            f"samples={len(self.samples)}",
        ]

        if self.samples:
            parts.extend(
                [
                    f"passed={self.passed_count}",
                    f"failed={self.failed_count}",
                    f"pass_rate={self.pass_rate:.3f}",
                ]
            )

        return f"{self.__class__.__name__}({', '.join(parts)})"


@dataclass
class ScenarioResult(EvalResultMixin, t.Generic[In, Out]):
    """Groups all iterations for a single scenario (parameter set)."""

    params: dict[str, t.Any]
    """The parameters defining this scenario."""
    iterations: list[IterationResult[In, Out]] = field(default_factory=list)
    """A list of iteration results for this scenario."""

    @property
    def samples(self) -> list[Sample[In, Out]]:
        """Returns a single, flat list of all samples from all iterations."""
        return [sample for iteration in self.iterations for sample in iteration.samples]

    def __repr__(self) -> str:
        params_str = format_dict(self.params, max_length=50)

        parts: list[str] = [
            f"params={params_str}",
            f"iterations={len(self.iterations)}",
            f"samples={len(self.samples)}",
        ]

        if self.samples:
            parts.extend(
                [
                    f"passed={self.passed_count}",
                    f"failed={self.failed_count}",
                    f"pass_rate={self.pass_rate:.3f}",
                ]
            )

        return f"{self.__class__.__name__}({', '.join(parts)})"


@dataclass
class EvalResult(EvalResultMixin, t.Generic[In, Out]):
    """Collection of samples resulting from an evaluation, grouped by scenario."""

    scenarios: list[ScenarioResult[In, Out]] = field(default_factory=list)
    """A list of results, one for each scenario in the evaluation."""
    stop_reason: EvalStopReason | None = None
    """The reason the evaluation stopped."""

    @property
    def samples(self) -> list[Sample[In, Out]]:
        """Returns a single, flat list of all samples from all scenarios and iterations."""
        return [sample for scenario in self.scenarios for sample in scenario.samples]

    def __repr__(self) -> str:
        total_samples = len(self.samples)
        total_iterations = sum(len(scenario.iterations) for scenario in self.scenarios)

        parts: list[str] = [
            f"scenarios={len(self.scenarios)}",
            f"samples={total_samples}",
            f"iterations={total_iterations}",
        ]

        if self.samples:
            parts.extend(
                [
                    f"passed={self.passed_count}",
                    f"failed={self.failed_count}",
                    f"pass_rate={self.pass_rate:.3f}",
                ]
            )

        return f"{self.__class__.__name__}({', '.join(parts)})"
