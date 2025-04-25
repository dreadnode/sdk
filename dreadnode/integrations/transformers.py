import importlib.util

if importlib.util.find_spec("transformers") is None:
    raise ModuleNotFoundError("Please install the `transformers` package to use this integration")

import typing as t

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

import dreadnode as dn

if t.TYPE_CHECKING:
    from dreadnode.tracing.span import RunSpan, Span

# ruff: noqa: ARG002


def _clean_keys(data: dict[str, t.Any]) -> dict[str, t.Any]:
    cleaned: dict[str, t.Any] = {}
    for key, val in data.items():
        _key = key.replace("eval_", "eval/").replace("test_", "test/").replace("train_", "train/")
        cleaned[_key] = val
    return cleaned


class DreadnodeCallback(TrainerCallback):
    """
    An implementation of the `TrainerCallback` interface for Dreadnode.

    This callback is used to log metrics and parameters to Dreadnode during training inside
    the `transformers` library or derivations (`trl`, etc.).
    """

    def __init__(
        self,
        project: str | None = None,
        run_name: str | None = None,
        tags: list[str] | None = None,
    ):
        self.project = project
        self.run_name = run_name
        self.tags = tags or []

        self._initialized = False
        self._run: RunSpan | None = None
        self._epoch_span: Span | None = None
        self._step_span: Span | None = None

    def _shutdown(self) -> None:
        if self._step_span is not None:
            self._step_span.__exit__(None, None, None)
            self._step_span = None

        if self._epoch_span is not None:
            self._epoch_span.__exit__(None, None, None)
            self._epoch_span = None

        if self._run is not None:
            self._run.__exit__(None, None, None)
            self._run = None

    def _setup(self, args: TrainingArguments, state: TrainerState, model: t.Any) -> None:
        if self._initialized:
            return

        self._initialized = True

        if not state.is_world_process_zero:
            return

        combined_dict = {**args.to_sanitized_dict()}

        if hasattr(model, "config") and model.config is not None:
            model_config = (
                model.config if isinstance(model.config, dict) else model.config.to_dict()
            )
            for key, value in model_config.items():
                combined_dict[f"model/{key}"] = value
        if hasattr(model, "peft_config") and model.peft_config is not None:
            for key, value in model.peft_config.items():
                combined_dict[f"peft/{key}"] = value

        run_name = self.run_name or args.run_name or state.trial_name

        self._run = dn.run(
            name=run_name,
            project=self.project,
            tags=self.tags,
        )
        self._run.__enter__()

        dn.log_params(**combined_dict)
        dn.push_update()

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: t.Any | None = None,
        **kwargs: t.Any,
    ) -> None:
        if not self._initialized:
            self._setup(args, state, model)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: t.Any,
    ) -> None:
        self._shutdown()

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: t.Any,
    ) -> None:
        if self._run is None or state.epoch is None:
            return

        dn.log_metric("epoch", state.epoch)

        self._epoch_span = dn.task_span(f"Epoch {state.epoch}")
        self._epoch_span.__enter__()

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: t.Any,
    ) -> None:
        if self._epoch_span is not None:
            self._epoch_span.__exit__(None, None, None)
            self._epoch_span = None

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: t.Any,
    ) -> None:
        if self._run is None:
            return

        dn.log_metric("step", state.global_step)

        self._step_span = dn.span(f"Step {state.global_step}")
        self._step_span.__enter__()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: t.Any,
    ) -> None:
        if self._step_span is not None:
            self._step_span.__exit__(None, None, None)
            self._step_span = None

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, t.Any] | None = None,
        **kwargs: t.Any,
    ) -> None:
        if self._run is None or logs is None:
            return

        for key, value in _clean_keys(logs).items():
            if isinstance(value, float | int):
                dn.log_metric(key, value, step=state.global_step)

        dn.push_update()
