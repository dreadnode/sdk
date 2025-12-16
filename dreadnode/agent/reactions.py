import rigging as rg
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass


@dataclass
class Reaction(Exception): ...  # noqa: N818


@dataclass
class Continue(Reaction):
    messages: list[rg.Message] = Field(repr=False)

    def log_metrics(self, step: int) -> None:
        from dreadnode import log_metric

        log_metric("continues", 1, step=step, mode="count")
        log_metric("messages", len(self.messages), step=step)


@dataclass
class Retry(Reaction):
    messages: list[rg.Message] | None = Field(default=None, repr=False)

    def log_metrics(self, step: int) -> None:
        from dreadnode import log_metric

        log_metric("retries", 1, step=step, mode="count")
        if self.messages:
            log_metric("messages", len(self.messages), step=step)


@dataclass
class RetryWithFeedback(Reaction):
    feedback: str


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Fail(Reaction):
    error: Exception | str


@dataclass
class Finish(Reaction):
    reason: str | None = None
