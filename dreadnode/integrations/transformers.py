import typing as t

from transformers.trainer_callback import (  # type: ignore [import-untyped]
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

import dreadnode as dn

if t.TYPE_CHECKING:
    from dreadnode.tracing import RunSpan, Span


class DreadnodeCallback(TrainerCallback):  # type: ignore [misc]
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
        print("\nshutdown\n")
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
            model_config = model.config if isinstance(model.config, dict) else model.config.to_dict()
            combined_dict = {**model_config, **combined_dict}
        if hasattr(model, "peft_config") and model.peft_config is not None:
            for key, value in model.peft_config.items():
                combined_dict[f"peft_{key}"] = value

        run_name = self.run_name or args.run_name or state.trial_name

        self._run = dn.run(
            name=run_name,
            project=self.project,
            tags=self.tags,
        )
        self._run.__enter__()

        dn.log_params(**combined_dict)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: t.Any | None = None,
        **kwargs: t.Any,
    ) -> None:
        print("\non_train_begin\n")
        if not self._initialized:
            self._setup(args, state, model)

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: t.Any
    ) -> None:
        print("\non_train_end\n")
        self._shutdown()

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: t.Any,
    ) -> None:
        print("\non_epoch_begin\n")
        if self._run is None:
            return

        dn.log_metric("epoch", state.epoch)

        self._epoch_span = dn.span(f"Epoch {state.epoch}", kind="epoch")
        self._epoch_span.__enter__()

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: t.Any,
    ) -> None:
        print("\non_epoch_end\n")
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
        print("\non_step_begin\n")
        if self._run is None:
            return

        dn.log_metric("step", state.global_step)

        self._step_span = dn.span(f"Step {state.global_step}", kind="step")
        self._step_span.__enter__()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: t.Any,
    ) -> None:
        print("\non_step_end\n")
        if self._step_span is not None:
            self._step_span.__exit__(None, None, None)
            self._step_span = None

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, t.Any] | None = None,
        **kwargs: t.Any,
    ) -> None:
        print("\non_evaluate\n")
        if self._run is None or metrics is None:
            return

        for key, value in metrics.items():
            if isinstance(value, float | int):
                metric_name = key if key.startswith("eval_") else f"eval_{key}"
                dn.log_metric(metric_name, value, step=state.global_step)

    def on_predict(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, t.Any] | None = None,
        **kwargs: t.Any,
    ) -> None:
        print("\non_predict\n")
        if self._run is None or metrics is None:
            return

        for key, value in metrics.items():
            if isinstance(value, float | int):
                metric_name = key if key.startswith("test_") else f"test_{key}"
                dn.log_metric(metric_name, value)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, t.Any] | None = None,
        **kwargs: t.Any,
    ) -> None:
        print("\non_log\n")
        if self._run is None or logs is None:
            return

        for key, value in logs.items():
            if not isinstance(value, float | int):
                continue

            metric_name = key if not key.startswith(("train_", "eval_", "test_")) else f"train_{key}"
            dn.log_metric(metric_name, value, step=state.global_step)
