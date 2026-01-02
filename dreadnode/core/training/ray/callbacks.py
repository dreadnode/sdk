"""
Training callbacks for customizing training behavior.

Callbacks allow you to hook into various points of the training loop to:
- Log metrics to external services (W&B, TensorBoard, etc.)
- Implement early stopping
- Save custom checkpoints
- Modify training behavior dynamically

Usage:
    from dreadnode.core.training.ray.callbacks import (
        TrainerCallback,
        EarlyStoppingCallback,
        WandbCallback,
        CallbackHandler,
    )

    # Create custom callback
    class MyCallback(TrainerCallback):
        def on_step_end(self, state, metrics, **kwargs):
            print(f"Step {state.step}: reward={metrics.get('reward_mean', 0):.4f}")

    # Use with trainer
    trainer = RayGRPOTrainer(config, callbacks=[MyCallback(), EarlyStoppingCallback(patience=10)])
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dreadnode.core.training.ray.coordinator import TrainingState


@dataclass
class ToolCallInfo:
    """
    Information about a tool call during rollout.

    This mirrors the structure of tool calls in the agent system,
    allowing callbacks to track and log tool usage during training.

    Attributes:
        name: Name of the tool being called
        tool_call_id: Unique identifier for this tool call
        arguments: JSON string of arguments passed to the tool
        result: Result returned by the tool (only available in on_tool_end)
        error: Error that occurred (only available in on_tool_error)
        duration_seconds: Time taken for tool execution
    """

    name: str
    tool_call_id: str
    arguments: str = ""
    result: str | None = None
    error: BaseException | None = None
    duration_seconds: float | None = None


@dataclass
class TrainerControl:
    """
    Control object returned by callbacks to modify training behavior.

    Attributes:
        should_stop: Set to True to stop training
        should_save: Set to True to trigger a checkpoint save
        should_evaluate: Set to True to trigger evaluation
        should_log: Set to True to trigger logging
    """

    should_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False

    def __or__(self, other: "TrainerControl") -> "TrainerControl":
        """Combine two control objects (OR logic)."""
        return TrainerControl(
            should_stop=self.should_stop or other.should_stop,
            should_save=self.should_save or other.should_save,
            should_evaluate=self.should_evaluate or other.should_evaluate,
            should_log=self.should_log or other.should_log,
        )


class TrainerCallback(ABC):
    """
    Base class for training callbacks.

    Override any of the `on_*` methods to customize training behavior.
    Each method receives the current training state and can return a
    TrainerControl object to modify training flow.

    Example:
        class PrintCallback(TrainerCallback):
            def on_step_end(self, state, metrics, **kwargs):
                print(f"Step {state.step} complete")
                if metrics.get("reward_mean", 0) > 0.9:
                    return TrainerControl(should_stop=True)
    """

    def on_train_begin(
        self,
        state: TrainingState,
        **kwargs: Any,
    ) -> TrainerControl | None:
        """Called at the beginning of training."""
        pass

    def on_train_end(
        self,
        state: TrainingState,
        **kwargs: Any,
    ) -> TrainerControl | None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(
        self,
        state: TrainingState,
        **kwargs: Any,
    ) -> TrainerControl | None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(
        self,
        state: TrainingState,
        **kwargs: Any,
    ) -> TrainerControl | None:
        """Called at the end of each epoch."""
        pass

    def on_step_begin(
        self,
        state: TrainingState,
        **kwargs: Any,
    ) -> TrainerControl | None:
        """Called at the beginning of each training step."""
        pass

    def on_step_end(
        self,
        state: TrainingState,
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> TrainerControl | None:
        """
        Called at the end of each training step.

        Args:
            state: Current training state
            metrics: Metrics from this step (loss, reward, etc.)
        """
        pass

    def on_evaluate(
        self,
        state: TrainingState,
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> TrainerControl | None:
        """
        Called after evaluation.

        Args:
            state: Current training state
            metrics: Evaluation metrics
        """
        pass

    def on_save(
        self,
        state: TrainingState,
        checkpoint_path: str,
        **kwargs: Any,
    ) -> TrainerControl | None:
        """
        Called when a checkpoint is saved.

        Args:
            state: Current training state
            checkpoint_path: Path where checkpoint was saved
        """
        pass

    def on_log(
        self,
        state: TrainingState,
        logs: dict[str, float],
        **kwargs: Any,
    ) -> TrainerControl | None:
        """
        Called when metrics are logged.

        Args:
            state: Current training state
            logs: Metrics being logged
        """
        pass

    def on_generation_end(
        self,
        state: TrainingState,
        prompts: list[str],
        completions: list[str],
        rewards: list[float],
        **kwargs: Any,
    ) -> TrainerControl | None:
        """
        Called after generating completions.

        Args:
            state: Current training state
            prompts: Input prompts
            completions: Generated completions
            rewards: Computed rewards
        """
        pass

    # =========================================================================
    # Tool Events (for agent training with tools)
    # =========================================================================

    def on_tool_start(
        self,
        state: TrainingState,
        tool_call: ToolCallInfo,
        **kwargs: Any,
    ) -> TrainerControl | None:
        """
        Called when a tool call is about to be executed during rollout.

        Args:
            state: Current training state
            tool_call: Information about the tool being called
        """
        pass

    def on_tool_end(
        self,
        state: TrainingState,
        tool_call: ToolCallInfo,
        **kwargs: Any,
    ) -> TrainerControl | None:
        """
        Called when a tool call has completed during rollout.

        Args:
            state: Current training state
            tool_call: Information about the completed tool call (includes result)
        """
        pass

    def on_tool_error(
        self,
        state: TrainingState,
        tool_call: ToolCallInfo,
        **kwargs: Any,
    ) -> TrainerControl | None:
        """
        Called when a tool call fails during rollout.

        Args:
            state: Current training state
            tool_call: Information about the failed tool call (includes error)
        """
        pass

    def on_rollout_step(
        self,
        state: TrainingState,
        step_index: int,
        prompt: str,
        completion: str,
        **kwargs: Any,
    ) -> TrainerControl | None:
        """
        Called after each step in a multi-turn rollout.

        Args:
            state: Current training state
            step_index: Index of the step within the rollout (0-indexed)
            prompt: The prompt/input for this step
            completion: The model's output for this step
        """
        pass


class CallbackHandler:
    """
    Manages a list of callbacks and dispatches events to them.

    The handler collects TrainerControl responses from all callbacks
    and combines them using OR logic.
    """

    def __init__(self, callbacks: list[TrainerCallback] | None = None):
        self.callbacks = callbacks or []

    def add_callback(self, callback: TrainerCallback) -> None:
        """Add a callback to the handler."""
        self.callbacks.append(callback)

    def remove_callback(self, callback_type: type) -> None:
        """Remove all callbacks of a given type."""
        self.callbacks = [cb for cb in self.callbacks if not isinstance(cb, callback_type)]

    def _call_event(
        self,
        event: str,
        state: TrainingState,
        **kwargs: Any,
    ) -> TrainerControl:
        """Call an event on all callbacks and combine results."""
        control = TrainerControl()

        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method is not None:
                result = method(state=state, **kwargs)
                if result is not None:
                    control = control | result

        return control

    def on_train_begin(self, state: TrainingState, **kwargs: Any) -> TrainerControl:
        return self._call_event("on_train_begin", state, **kwargs)

    def on_train_end(self, state: TrainingState, **kwargs: Any) -> TrainerControl:
        return self._call_event("on_train_end", state, **kwargs)

    def on_epoch_begin(self, state: TrainingState, **kwargs: Any) -> TrainerControl:
        return self._call_event("on_epoch_begin", state, **kwargs)

    def on_epoch_end(self, state: TrainingState, **kwargs: Any) -> TrainerControl:
        return self._call_event("on_epoch_end", state, **kwargs)

    def on_step_begin(self, state: TrainingState, **kwargs: Any) -> TrainerControl:
        return self._call_event("on_step_begin", state, **kwargs)

    def on_step_end(
        self, state: TrainingState, metrics: dict[str, float], **kwargs: Any
    ) -> TrainerControl:
        return self._call_event("on_step_end", state, metrics=metrics, **kwargs)

    def on_evaluate(
        self, state: TrainingState, metrics: dict[str, float], **kwargs: Any
    ) -> TrainerControl:
        return self._call_event("on_evaluate", state, metrics=metrics, **kwargs)

    def on_save(
        self, state: TrainingState, checkpoint_path: str, **kwargs: Any
    ) -> TrainerControl:
        return self._call_event("on_save", state, checkpoint_path=checkpoint_path, **kwargs)

    def on_log(
        self, state: TrainingState, logs: dict[str, float], **kwargs: Any
    ) -> TrainerControl:
        return self._call_event("on_log", state, logs=logs, **kwargs)

    def on_generation_end(
        self,
        state: TrainingState,
        prompts: list[str],
        completions: list[str],
        rewards: list[float],
        **kwargs: Any,
    ) -> TrainerControl:
        return self._call_event(
            "on_generation_end",
            state,
            prompts=prompts,
            completions=completions,
            rewards=rewards,
            **kwargs,
        )

    # Tool events
    def on_tool_start(
        self, state: TrainingState, tool_call: ToolCallInfo, **kwargs: Any
    ) -> TrainerControl:
        return self._call_event("on_tool_start", state, tool_call=tool_call, **kwargs)

    def on_tool_end(
        self, state: TrainingState, tool_call: ToolCallInfo, **kwargs: Any
    ) -> TrainerControl:
        return self._call_event("on_tool_end", state, tool_call=tool_call, **kwargs)

    def on_tool_error(
        self, state: TrainingState, tool_call: ToolCallInfo, **kwargs: Any
    ) -> TrainerControl:
        return self._call_event("on_tool_error", state, tool_call=tool_call, **kwargs)

    def on_rollout_step(
        self,
        state: TrainingState,
        step_index: int,
        prompt: str,
        completion: str,
        **kwargs: Any,
    ) -> TrainerControl:
        return self._call_event(
            "on_rollout_step",
            state,
            step_index=step_index,
            prompt=prompt,
            completion=completion,
            **kwargs,
        )


# =============================================================================
# Built-in Callbacks
# =============================================================================


class PrintCallback(TrainerCallback):
    """Simple callback that prints training progress."""

    def __init__(self, print_every: int = 1):
        self.print_every = print_every

    def on_step_end(
        self,
        state: TrainingState,
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> TrainerControl | None:
        if state.step % self.print_every == 0:
            reward = metrics.get("reward_mean", 0)
            loss = metrics.get("loss", 0)
            print(f"[Step {state.step}] loss={loss:.4f} reward={reward:.4f}")
        return None


class EarlyStoppingCallback(TrainerCallback):
    """
    Stop training when a metric stops improving.

    Args:
        patience: Number of steps to wait for improvement
        metric: Metric to monitor (default: "reward_mean")
        mode: "max" to maximize metric, "min" to minimize
        min_delta: Minimum change to qualify as improvement
        verbose: Whether to print early stopping messages
    """

    def __init__(
        self,
        patience: int = 10,
        metric: str = "reward_mean",
        mode: str = "max",
        min_delta: float = 0.0,
        verbose: bool = True,
    ):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_value: float | None = None
        self.steps_without_improvement = 0

    def on_step_end(
        self,
        state: TrainingState,
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> TrainerControl | None:
        current = metrics.get(self.metric)
        if current is None:
            return None

        if self.best_value is None:
            self.best_value = current
            return None

        if self.mode == "max":
            improved = current > self.best_value + self.min_delta
        else:
            improved = current < self.best_value - self.min_delta

        if improved:
            self.best_value = current
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        if self.steps_without_improvement >= self.patience:
            if self.verbose:
                print(
                    f"Early stopping triggered at step {state.step}. "
                    f"Best {self.metric}: {self.best_value:.4f}"
                )
            return TrainerControl(should_stop=True)

        return None


class MetricHistoryCallback(TrainerCallback):
    """
    Track metric history for analysis.

    Attributes:
        history: List of (step, metrics) tuples
    """

    def __init__(self):
        self.history: list[tuple[int, dict[str, float]]] = []

    def on_step_end(
        self,
        state: TrainingState,
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> TrainerControl | None:
        self.history.append((state.step, metrics.copy()))
        return None

    def get_metric(self, name: str) -> list[tuple[int, float]]:
        """Get history for a specific metric."""
        return [(step, m[name]) for step, m in self.history if name in m]

    def to_dataframe(self):
        """Convert history to pandas DataFrame."""
        import pandas as pd

        records = []
        for step, metrics in self.history:
            records.append({"step": step, **metrics})
        return pd.DataFrame(records)


class JSONLoggingCallback(TrainerCallback):
    """
    Log metrics to a JSON file.

    Args:
        log_path: Path to the JSON log file
        log_every: Log every N steps
    """

    def __init__(self, log_path: str = "training_log.json", log_every: int = 1):
        self.log_path = log_path
        self.log_every = log_every
        self.logs: list[dict] = []

    def on_step_end(
        self,
        state: TrainingState,
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> TrainerControl | None:
        if state.step % self.log_every == 0:
            entry = {
                "step": state.step,
                "epoch": state.epoch,
                "samples_seen": state.samples_seen,
                "timestamp": time.time(),
                **metrics,
            }
            self.logs.append(entry)
            self._save()
        return None

    def _save(self) -> None:
        with open(self.log_path, "w") as f:
            json.dump(self.logs, f, indent=2)

    def on_train_end(
        self,
        state: TrainingState,
        **kwargs: Any,
    ) -> TrainerControl | None:
        self._save()
        print(f"Training logs saved to {self.log_path}")
        return None


class WandbCallback(TrainerCallback):
    """
    Log metrics to Weights & Biases.

    Args:
        project: W&B project name
        name: Run name (optional)
        config: Additional config to log
        log_every: Log every N steps
    """

    def __init__(
        self,
        project: str = "dreadnode-training",
        name: str | None = None,
        config: dict | None = None,
        log_every: int = 1,
    ):
        self.project = project
        self.name = name
        self.config = config or {}
        self.log_every = log_every
        self._run = None

    def on_train_begin(
        self,
        state: TrainingState,
        **kwargs: Any,
    ) -> TrainerControl | None:
        try:
            import wandb

            self._run = wandb.init(
                project=self.project,
                name=self.name,
                config=self.config,
                reinit=True,
            )
            print(f"W&B run started: {wandb.run.url}")
        except ImportError:
            print("wandb not installed. Install with: pip install wandb")
        return None

    def on_step_end(
        self,
        state: TrainingState,
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> TrainerControl | None:
        if self._run is None:
            return None

        if state.step % self.log_every == 0:
            import wandb

            wandb.log(
                {
                    "step": state.step,
                    "epoch": state.epoch,
                    "samples_seen": state.samples_seen,
                    **metrics,
                },
                step=state.step,
            )
        return None

    def on_train_end(
        self,
        state: TrainingState,
        **kwargs: Any,
    ) -> TrainerControl | None:
        if self._run is not None:
            import wandb

            wandb.finish()
        return None


class TensorBoardCallback(TrainerCallback):
    """
    Log metrics to TensorBoard.

    Args:
        log_dir: TensorBoard log directory
        log_every: Log every N steps
    """

    def __init__(self, log_dir: str = "./runs", log_every: int = 1):
        self.log_dir = log_dir
        self.log_every = log_every
        self._writer = None

    def on_train_begin(
        self,
        state: TrainingState,
        **kwargs: Any,
    ) -> TrainerControl | None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=self.log_dir)
            print(f"TensorBoard logging to {self.log_dir}")
        except ImportError:
            print("TensorBoard not installed. Install with: pip install tensorboard")
        return None

    def on_step_end(
        self,
        state: TrainingState,
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> TrainerControl | None:
        if self._writer is None:
            return None

        if state.step % self.log_every == 0:
            for name, value in metrics.items():
                self._writer.add_scalar(f"train/{name}", value, state.step)
        return None

    def on_train_end(
        self,
        state: TrainingState,
        **kwargs: Any,
    ) -> TrainerControl | None:
        if self._writer is not None:
            self._writer.close()
        return None


class CheckpointCallback(TrainerCallback):
    """
    Save checkpoints at regular intervals or based on metric improvement.

    Args:
        save_every: Save every N steps (0 to disable)
        save_best: Save when metric improves
        metric: Metric to monitor for best checkpoint
        mode: "max" or "min"
        checkpoint_dir: Directory for checkpoints
    """

    def __init__(
        self,
        save_every: int = 100,
        save_best: bool = True,
        metric: str = "reward_mean",
        mode: str = "max",
        checkpoint_dir: str = "./checkpoints",
    ):
        self.save_every = save_every
        self.save_best = save_best
        self.metric = metric
        self.mode = mode
        self.checkpoint_dir = checkpoint_dir
        self.best_value: float | None = None

    def on_step_end(
        self,
        state: TrainingState,
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> TrainerControl | None:
        should_save = False

        # Save at regular intervals
        if self.save_every > 0 and state.step % self.save_every == 0:
            should_save = True

        # Save if metric improved
        if self.save_best:
            current = metrics.get(self.metric)
            if current is not None:
                if self.best_value is None:
                    self.best_value = current
                    should_save = True
                elif self.mode == "max" and current > self.best_value:
                    self.best_value = current
                    should_save = True
                elif self.mode == "min" and current < self.best_value:
                    self.best_value = current
                    should_save = True

        if should_save:
            return TrainerControl(should_save=True)

        return None


class GradientClippingCallback(TrainerCallback):
    """
    Monitor and optionally adjust gradient clipping.

    Args:
        log_grad_norm: Whether to log gradient norms
        warn_threshold: Warn if grad norm exceeds this
    """

    def __init__(
        self,
        log_grad_norm: bool = True,
        warn_threshold: float = 10.0,
    ):
        self.log_grad_norm = log_grad_norm
        self.warn_threshold = warn_threshold

    def on_step_end(
        self,
        state: TrainingState,
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> TrainerControl | None:
        grad_norm = metrics.get("grad_norm")
        if grad_norm is not None and grad_norm > self.warn_threshold:
            print(f"Warning: Large gradient norm at step {state.step}: {grad_norm:.4f}")
        return None


class SampleLoggingCallback(TrainerCallback):
    """
    Log sample generations for debugging.

    Args:
        log_every: Log samples every N steps
        num_samples: Number of samples to log
        output_file: Optional file to write samples to
    """

    def __init__(
        self,
        log_every: int = 10,
        num_samples: int = 3,
        output_file: str | None = None,
    ):
        self.log_every = log_every
        self.num_samples = num_samples
        self.output_file = output_file
        self._samples: list[dict] = []

    def on_generation_end(
        self,
        state: TrainingState,
        prompts: list[str],
        completions: list[str],
        rewards: list[float],
        **kwargs: Any,
    ) -> TrainerControl | None:
        if state.step % self.log_every != 0:
            return None

        print(f"\n--- Sample Generations (Step {state.step}) ---")
        for i in range(min(self.num_samples, len(prompts))):
            print(f"\n[Sample {i+1}] Reward: {rewards[i]:.4f}")
            print(f"Prompt: {prompts[i][:100]}...")
            print(f"Completion: {completions[i][:200]}...")

            if self.output_file:
                self._samples.append({
                    "step": state.step,
                    "prompt": prompts[i],
                    "completion": completions[i],
                    "reward": rewards[i],
                })

        if self.output_file and self._samples:
            with open(self.output_file, "w") as f:
                json.dump(self._samples, f, indent=2)

        return None


class ProgressCallback(TrainerCallback):
    """
    Display a progress bar during training.

    Args:
        total_steps: Total number of training steps
    """

    def __init__(self, total_steps: int | None = None):
        self.total_steps = total_steps
        self._pbar = None

    def on_train_begin(
        self,
        state: TrainingState,
        **kwargs: Any,
    ) -> TrainerControl | None:
        try:
            from tqdm import tqdm

            self._pbar = tqdm(total=self.total_steps, desc="Training")
        except ImportError:
            pass
        return None

    def on_step_end(
        self,
        state: TrainingState,
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> TrainerControl | None:
        if self._pbar is not None:
            self._pbar.update(1)
            self._pbar.set_postfix({
                "loss": f"{metrics.get('loss', 0):.4f}",
                "reward": f"{metrics.get('reward_mean', 0):.4f}",
            })
        return None

    def on_train_end(
        self,
        state: TrainingState,
        **kwargs: Any,
    ) -> TrainerControl | None:
        if self._pbar is not None:
            self._pbar.close()
        return None


class ToolLoggingCallback(TrainerCallback):
    """
    Track and log tool usage during training.

    Provides detailed logging of tool calls, success rates, and timing
    for debugging and analyzing tool-using agents.

    Args:
        log_every: Log summary every N steps
        verbose: Print individual tool calls
        output_file: Optional file to write tool call logs
    """

    def __init__(
        self,
        log_every: int = 10,
        verbose: bool = False,
        output_file: str | None = None,
    ):
        self.log_every = log_every
        self.verbose = verbose
        self.output_file = output_file

        # Statistics
        self.total_calls: int = 0
        self.successful_calls: int = 0
        self.failed_calls: int = 0
        self.calls_by_tool: dict[str, int] = {}
        self.errors_by_tool: dict[str, int] = {}
        self.total_duration: float = 0.0
        self._call_log: list[dict] = []

    def on_tool_start(
        self,
        state: TrainingState,
        tool_call: ToolCallInfo,
        **kwargs: Any,
    ) -> TrainerControl | None:
        if self.verbose:
            print(f"[Tool Start] {tool_call.name}({tool_call.arguments[:50]}...)")
        return None

    def on_tool_end(
        self,
        state: TrainingState,
        tool_call: ToolCallInfo,
        **kwargs: Any,
    ) -> TrainerControl | None:
        self.total_calls += 1
        self.successful_calls += 1
        self.calls_by_tool[tool_call.name] = self.calls_by_tool.get(tool_call.name, 0) + 1

        if tool_call.duration_seconds:
            self.total_duration += tool_call.duration_seconds

        if self.verbose:
            result_preview = (tool_call.result or "")[:50]
            print(f"[Tool End] {tool_call.name} -> {result_preview}...")

        if self.output_file:
            self._call_log.append({
                "step": state.step,
                "tool": tool_call.name,
                "arguments": tool_call.arguments,
                "result": tool_call.result,
                "duration": tool_call.duration_seconds,
                "success": True,
            })

        return None

    def on_tool_error(
        self,
        state: TrainingState,
        tool_call: ToolCallInfo,
        **kwargs: Any,
    ) -> TrainerControl | None:
        self.total_calls += 1
        self.failed_calls += 1
        self.errors_by_tool[tool_call.name] = self.errors_by_tool.get(tool_call.name, 0) + 1

        if self.verbose:
            print(f"[Tool Error] {tool_call.name}: {tool_call.error}")

        if self.output_file:
            self._call_log.append({
                "step": state.step,
                "tool": tool_call.name,
                "arguments": tool_call.arguments,
                "error": str(tool_call.error),
                "success": False,
            })

        return None

    def on_step_end(
        self,
        state: TrainingState,
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> TrainerControl | None:
        if state.step % self.log_every == 0 and self.total_calls > 0:
            success_rate = self.successful_calls / self.total_calls if self.total_calls > 0 else 0
            avg_duration = self.total_duration / self.successful_calls if self.successful_calls > 0 else 0

            print(f"\n--- Tool Usage Summary (Step {state.step}) ---")
            print(f"  Total calls: {self.total_calls}")
            print(f"  Success rate: {success_rate:.1%}")
            print(f"  Avg duration: {avg_duration:.3f}s")
            if self.calls_by_tool:
                print(f"  Calls by tool: {self.calls_by_tool}")
            if self.errors_by_tool:
                print(f"  Errors by tool: {self.errors_by_tool}")

        return None

    def on_train_end(
        self,
        state: TrainingState,
        **kwargs: Any,
    ) -> TrainerControl | None:
        if self.output_file and self._call_log:
            with open(self.output_file, "w") as f:
                json.dump(self._call_log, f, indent=2)
            print(f"Tool call log saved to {self.output_file}")

        if self.total_calls > 0:
            success_rate = self.successful_calls / self.total_calls
            print(f"\n=== Final Tool Usage Statistics ===")
            print(f"  Total tool calls: {self.total_calls}")
            print(f"  Successful: {self.successful_calls} ({success_rate:.1%})")
            print(f"  Failed: {self.failed_calls}")
            print(f"  Total duration: {self.total_duration:.2f}s")
            if self.calls_by_tool:
                print(f"  Calls by tool: {dict(sorted(self.calls_by_tool.items(), key=lambda x: -x[1]))}")

        return None

    def get_statistics(self) -> dict:
        """Return current tool usage statistics."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / self.total_calls if self.total_calls > 0 else 0,
            "avg_duration": self.total_duration / self.successful_calls if self.successful_calls > 0 else 0,
            "calls_by_tool": self.calls_by_tool.copy(),
            "errors_by_tool": self.errors_by_tool.copy(),
        }
