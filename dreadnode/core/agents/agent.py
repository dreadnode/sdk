import inspect
import typing as t
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from copy import deepcopy
from pathlib import Path
from textwrap import dedent

import typing_extensions as te
from loguru import logger
from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
)
from ulid import ULID

from dreadnode.core.agents.events import (
    AgentEnd,
    AgentError,
    AgentEvent,
    AgentStalled,
    AgentStart,
    AgentStep,
    GenerationStep,
    ReactStep,
    ToolEnd,
    ToolError,
    ToolStart,
    ToolStep,
)
from dreadnode.core.agents.exceptions import MaxStepsError
from dreadnode.core.agents.reactions import (
    Continue,
    Fail,
    Finish,
    Reaction,
    Retry,
    RetryWithFeedback,
)
from dreadnode.core.agents.trajectory import Trajectory
from dreadnode.core.environment import Environment
from dreadnode.core.evaluations import (
    DatasetLike,
    EvalResult,
    Evaluation,
    IterationResult,
    Sample,
    ScenarioResult,
)
from dreadnode.core.exceptions import warn_at_user_stacklevel
from dreadnode.core.execution import Executor, TraceContext
from dreadnode.core.generators import caching
from dreadnode.core.generators.chat import Chat
from dreadnode.core.generators.generator import GenerateParams, Generator, get_generator
from dreadnode.core.generators.message import Message, inject_system_content
from dreadnode.core.generators.models import SystemErrorModel
from dreadnode.core.hook import Hook
from dreadnode.core.judge import Judge, Rubric
from dreadnode.core.meta import Component, Config, component
from dreadnode.core.optimization import Direction, Study, StudyResult, StudyStopCondition
from dreadnode.core.scorer import Scorer, ScorersLike
from dreadnode.core.search import Categorical, Float, Int, Search, SearchLike
from dreadnode.core.stopping import StopCondition
from dreadnode.core.task import Task, task
from dreadnode.core.tools import Tool, ToolCall, ToolMode, Toolset, discover_tools_on_obj
from dreadnode.core.tools.transforms import (
    make_tools_to_xml_transform,
    tools_to_json_in_xml_transform,
    tools_to_json_transform,
    tools_to_json_with_tag_transform,
    tools_to_pythonic_transform,
)
from dreadnode.core.transforms import PostTransform, Transform
from dreadnode.core.util import flatten_list, join_generators, safe_repr
from dreadnode.search import grid, optuna_, random


class AgentWarning(UserWarning):
    """Warning raised when an agent is used in a way that may not be safe or intended."""


class Agent(Executor[AgentEvent, Trajectory]):
    """
    Agent abstraction for applying tools, event logic, and message state to LLM generation.

    Now extends Executor for consistent streaming/tracing patterns.

    Args:

        name: The name of the agent.
        description: A brief description of the agent.
        tags: Tags associated with the agent.
        label: An optional label for the agent.
        agent_id: The unique identifier for this agent instance.
        model: Inference model (rigging generator or identifier).
        instructions: The agent's core instructions.
        max_steps: The maximum number of steps (generation + tool calls).
        cache: How to handle cache_control entries on inference messages.
        tools: Tools the agent can use.
        tool_mode: The tool calling mode to use.
        stop_conditions: The logical condition for successfully stopping a run.
        hooks: Hooks to apply during agent execution.
        trajectory: Stateful trajectory for this agent.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    tags: list[str] = Config(default_factory=lambda: ["agent"])
    agent_id: ULID = Field(default_factory=ULID)
    model: str | Generator | None = Config(default=None, expose_as=str | None)
    instructions: t.Annotated[str | None, AfterValidator(lambda x: dedent(x) if x else x)] = Config(
        default=None
    )
    max_steps: int = Config(default=10)
    cache: caching.CacheMode | None = Config(default=None, repr=False)
    tools: list[Tool | Toolset] = Config(default_factory=list, validate_default=False)
    tool_mode: ToolMode = Config(default="auto", repr=False)
    stop_conditions: list[StopCondition] = Config(default_factory=list)
    hooks: list[Hook] = Config(default_factory=list, repr=False)
    judge: Judge[Rubric] | None = Config(default=None, repr=False)
    environment: Environment | None = Config(default=None, repr=False)
    trajectory: Trajectory = Field(default_factory=Trajectory, exclude=True, repr=False)

    # Private state
    _generator: Generator | None = PrivateAttr(None, init=False)
    _current_input: str = PrivateAttr("", init=False)

    @field_validator("tools", mode="before")
    @classmethod
    def validate_tools(cls, value: t.Any) -> t.Any:
        tools: list[Tool | Toolset] = []
        for tool in flatten_list(list(value)):
            if isinstance(tool, (Toolset, Tool)):
                tools.append(tool)
            elif interior_tools := discover_tools_on_obj(tool):
                tools.extend(interior_tools)
            else:
                tools.append(
                    Tool.from_callable(tool if isinstance(tool, Component) else component(tool))
                )
        return tools

    @field_validator("hooks", mode="before")
    @classmethod
    def validate_hooks(cls, value: t.Any) -> list[Hook]:
        if value is None:
            return []
        if callable(value):
            return [value]
        return list(value)

    @property
    def model_name(self) -> str | None:
        if self.model is not None:
            return self.generator.to_identifier(short=True)
        return None

    @property
    def generator(self) -> Generator:
        if self._generator is not None:
            return self._generator

        if isinstance(self.model, str):
            self._generator = get_generator(self.model)
        elif isinstance(self.model, Generator):
            self._generator = self.model
        else:
            raise TypeError("Model must be a string or a Generator instance.")

        return self._generator

    @property
    def all_tools(self) -> list[Tool]:
        flat_tools: list[Tool] = []
        for item in self.tools:
            if isinstance(item, Toolset):
                flat_tools.extend(item.get_tools())
            elif isinstance(item, Tool):
                flat_tools.append(item)
        return flat_tools

    def _extract_result(self, event: AgentEvent) -> Trajectory | None:
        """Extract trajectory from AgentEnd event."""
        if isinstance(event, AgentEnd):
            return self.trajectory
        return None

    def _get_trace_context(self) -> TraceContext:
        """Build trace context with agent-specific information."""
        ctx = TraceContext.from_executor(self)
        ctx.span_type = "agent"

        ctx.inputs.update(
            {
                # Agent identity
                "agent_id": str(self.agent_id),
                "agent_name": self.name,
                # Execution config
                "goal": self._current_input,
                "instructions": self.instructions,
                "model": self.model_name,
                "tools": [t.name for t in self.all_tools],
                "tool_schemas": [t.api_definition for t in self.all_tools],
                "stop_conditions": [s.name for s in self.stop_conditions],
                "generator_params": self.generator.params.to_dict(),
            }
        )
        ctx.params.update(
            {
                "max_steps": self.max_steps,
                "tool_mode": self.tool_mode,
                "tool_count": len(self.all_tools),
                "instructions_length": len(self.instructions or ""),
                "stop_condition_count": len(self.stop_conditions),
            }
        )

        for tag in self.tags:
            ctx.params[f"tag:{tag}"] = True

        return ctx

    def _export_trajectory(self, instance: t.Any) -> None:
        """Export trajectory for SFT/RL training.

        Args:
            instance: The Dreadnode instance with trace config.
        """
        if self.trajectory is None:
            return

        trace_config = getattr(instance, "_trace_config", None)
        if trace_config is None:
            return

        try:
            # Convert trajectory to training-friendly format
            # Using trajectory_to_turns for conversation format suitable for SFT
            turns = self.trajectory.trajectory_to_turns()

            # Build training example with metadata
            training_example = {
                "agent_id": str(self.agent_id),
                "agent_name": self.name,
                "session_id": str(self.trajectory.session_id),
                "model": self.model_name,
                "goal": self._current_input,
                "instructions": self.instructions,
                "total_steps": len(self.trajectory.steps),
                "total_tokens": self.trajectory.usage.total_tokens,
                "input_tokens": self.trajectory.usage.input_tokens,
                "output_tokens": self.trajectory.usage.output_tokens,
                "tools": [t.name for t in self.all_tools],
                "turns": turns,
            }

            trace_config.write_trajectory(training_example)
        except Exception as e:  # noqa: BLE001
            from loguru import logger

            logger.debug(f"Failed to export trajectory: {e}")

    async def _stream(self) -> t.AsyncGenerator[AgentEvent, None]:
        """
        Core agent execution loop.

        This method contains the main agent logic, delegating to helper methods
        for generation, dispatch, and tool processing.
        """
        messages = [
            *deepcopy(self.trajectory.messages),
            Message("user", str(self._current_input)),
        ]
        messages = inject_system_content(messages, self.get_prompt())

        step_count = 0
        error: Exception | str | None = None

        # Agent start
        async for event in self._dispatch(
            AgentStart(
                agent_id=self.agent_id,
                inputs={"goal": self._current_input},
                params={},
            )
        ):
            yield event

        # Core step loop
        while step_count < self.max_steps:
            step_count += 1

            try:
                # Generation
                step_chat = await self._generate(messages)
                if step_chat.failed and step_chat.error:
                    async for event in self._dispatch(
                        AgentError(
                            agent_id=self.agent_id,
                            status="errored",
                            error=t.cast("Exception", step_chat.error),
                        )
                    ):
                        yield event
                    error = t.cast("Exception", step_chat.error)
                    break

                step_chat.generated[-1].metadata.update(step_chat.extra)
                messages.extend(step_chat.generated)

                async for event in self._dispatch(
                    GenerationStep(
                        agent_id=self.agent_id,
                        agent_name=self.name,
                        status="running",
                        generator=self._generator,
                        messages=messages,
                        step=step_count,
                        usage=step_chat.usage,
                    )
                ):
                    yield event

                # Check stop conditions
                if any(cond(self.trajectory.steps) for cond in self.stop_conditions):
                    logger.info("A stop condition was met. Ending run.")
                    break

                # Check if stalled
                if not messages[-1].tool_calls:
                    if not self.stop_conditions:
                        break

                    logger.warning(
                        f"Agent '{self.name}' stalled: No tool calls and no stop conditions met."
                    )
                    async for event in self._dispatch(
                        AgentStalled(agent_id=self.agent_id, agent_name=self.name, status="stalled")
                    ):
                        yield event
                    break

                # Process tool calls
                stopped_by_tool_call: ToolCall | None = None
                tool_call_generators = [
                    self._process_tool_call(tc, step_count) for tc in messages[-1].tool_calls
                ]

                async for event in join_generators(*tool_call_generators):
                    yield event
                    if isinstance(event, ToolStep):
                        messages.extend(event.messages)
                        if stopped_by_tool_call is None and event.stop:
                            stopped_by_tool_call = event.tool_call

                if stopped_by_tool_call:
                    raise Finish(f"Tool '{stopped_by_tool_call.name}' requested to stop the agent.")

                # Check stop conditions again
                if any(cond(self.trajectory.steps) for cond in self.stop_conditions):
                    break

            except Retry as e:
                messages = e.messages or messages
                continue
            except Fail as e:
                error = e.error
                break
            except Finish:
                break
            except Exception as e:
                error = e
                break

        # Determine stop reason
        stop_reason = "finished"
        if step_count >= self.max_steps:
            error = MaxStepsError(max_steps=self.max_steps)
            stop_reason = "max_steps_reached"
        elif error is not None:
            stop_reason = "error"
        elif self.trajectory.steps and isinstance(self.trajectory.steps[-1], AgentStalled):
            stop_reason = "stalled"

        async for event in self._dispatch(
            AgentEnd(
                agent_id=self.agent_id,
                agent_name=self.name,
                status="errored" if error else "finished",
                stop_reason=stop_reason,
                error=error,
            )
        ):
            yield event

    async def _stream_traced(self) -> t.AsyncGenerator[AgentEvent, None]:
        """Override to add trajectory logging."""
        from dreadnode import DEFAULT_INSTANCE, log_output, log_outputs

        last_event: AgentEvent | None = None

        try:
            async for event in super()._stream_traced():
                if isinstance(event, AgentStep):
                    self.trajectory.log_step(event)
                last_event = event
                yield event
        finally:
            if self.trajectory is not None:
                # Log token usage
                log_outputs(
                    input_tokens=self.trajectory.usage.input_tokens,
                    output_tokens=self.trajectory.usage.output_tokens,
                    total_tokens=self.trajectory.usage.total_tokens,
                )

                # Log execution results
                log_outputs(
                    actual_steps=len(self.trajectory.steps),
                    session_id=str(self.trajectory.session_id),
                )

                # Log end state from AgentEnd event
                if isinstance(last_event, AgentEnd):
                    log_outputs(
                        status=last_event.status,
                        stop_reason=last_event.stop_reason,
                        error=str(last_event.error) if last_event.error else None,
                    )

                # Export trajectory for SFT/RL training
                self._export_trajectory(DEFAULT_INSTANCE)

    @asynccontextmanager
    async def stream(self, goal: str) -> t.AsyncIterator[t.AsyncGenerator[AgentEvent, None]]:
        """
        Stream agent execution with tool context management.

        Args:
            goal: The goal or input for the agent.
        """
        self._current_input = goal
        self.trajectory = Trajectory(agent_id=self.agent_id)

        async with AsyncExitStack() as stack:
            for tool_container in self.tools:
                if hasattr(tool_container, "__aenter__") and hasattr(tool_container, "__aexit__"):
                    await stack.enter_async_context(tool_container)

            async with aclosing(self._stream_traced()) as stream:
                yield stream

    async def run(self, goal: str) -> Trajectory:
        """Execute the agent and return the trajectory."""
        async with self.stream(goal) as stream:
            async for event in stream:
                if isinstance(event, AgentEnd):
                    return self.trajectory

        raise RuntimeError("Agent run finished unexpectedly.")

    def with_(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        label: str | None = None,
        model: str | Generator | None = None,
        instructions: str | None = None,
        max_steps: int | None = None,
        caching: caching.CacheMode | None = None,
        tools: list[Tool | Toolset] | None = None,
        tool_mode: ToolMode | None = None,
        stop_conditions: list[StopCondition] | None = None,
        append: bool = False,
    ) -> te.Self:
        """Create a modified clone of the agent."""
        new = self.clone()

        updates = {
            "name": name,
            "description": description,
            "label": label,
            "model": model,
            "instructions": instructions,
            "max_steps": max_steps,
            "caching": caching,
            "tool_mode": tool_mode,
        }
        for field, value in updates.items():
            if value is not None:
                setattr(new, field, value)

        # Apply list fields with append support
        new._apply_updates(
            {"tags": tags, "tools": tools, "stop_conditions": stop_conditions},
            list_fields={"tags", "tools", "stop_conditions"},
            append=append,
        )

        new.model_post_init(None)
        return new

    def get_prompt(self) -> str:
        """Generate the system prompt for the agent."""
        prompt = "You are an agent that can use tools to assist with tasks."
        if self.instructions:
            prompt += f"\n\n<instructions>\n{self.instructions}\n</instructions>"
        return prompt

    def reset(self) -> Trajectory:
        """Reset the agent's internal state."""
        previous = self.trajectory
        self.trajectory = Trajectory(agent_id=self.agent_id)
        return previous

    def _get_transforms(self) -> list[Transform]:
        """Resolve message transforms for the current configuration."""
        transforms = []
        if self.tools:
            match self.tool_mode:
                case "xml":
                    transforms.append(
                        make_tools_to_xml_transform(self.all_tools, add_tool_stop_token=True)
                    )
                case "json-in-xml":
                    transforms.append(tools_to_json_in_xml_transform)
                case "json-with-tag":
                    transforms.append(tools_to_json_with_tag_transform)
                case "json":
                    transforms.append(tools_to_json_transform)
                case "pythonic":
                    transforms.append(tools_to_pythonic_transform)
        return transforms

    async def _generate(self, messages: list[Message]) -> Chat:
        """Execute a single LLM generation step."""
        messages = list(messages)
        params = GenerateParams(
            tools=[tool.api_definition for tool in self.all_tools],
        )

        if self.tool_mode == "auto" and self.tools:
            self.tool_mode = (
                "api" if await self.generator.supports_function_calling() else "json-in-xml"
            )

        transforms = self._get_transforms()
        post_transforms: list[PostTransform | None] = []
        for transform_callback in transforms:
            messages, params, post_transform = await transform_callback(messages, params)
            post_transforms.append(post_transform)

        try:
            messages = caching.apply_cache_mode_to_messages(self.cache, [messages])[0]
            logger.trace(f"Generating with model '{self.generator.model}'. Messages: {messages!r}")

            generated = (await self.generator.generate_messages([messages], [params]))[0]
            if isinstance(generated, BaseException):
                raise generated

            chat = Chat(
                messages,
                [generated.message],
                generator=self.generator,
                params=params,
                stop_reason=generated.stop_reason,
                usage=generated.usage,
                extra=generated.extra,
            )
        except Exception as error:
            logger.opt(exception=True).error("Error during generation")
            chat = Chat(
                messages,
                [],
                generator=self.generator,
                params=params,
                failed=True,
                error=error,
            )

        for post_transform in [tf for tf in post_transforms if tf]:
            chat = await post_transform(chat) or chat

        return chat

    async def _dispatch(self, event: AgentEvent) -> t.AsyncIterator[AgentEvent]:
        """Dispatch an event through hooks and handle reactions."""
        yield event

        logger.debug(
            f"Agent '{self.name}' ({self.trajectory.session_id}) dispatching '{type(event).__name__}'"
        )

        hook_reactions: dict[str, Reaction | None] = {}
        for hook in self.hooks:
            hook_name = getattr(hook, "__name__", getattr(hook, "__qualname__", safe_repr(hook)))

            reaction: Reaction | None = None
            try:
                reaction = hook(event)
                if inspect.isawaitable(reaction):
                    reaction = t.cast("Reaction", await reaction)
            except Reaction as r:
                reaction = r

            if reaction is None:
                continue

            logger.debug(f"Hook '{hook_name}' returned reaction: {reaction!r}")

            if not isinstance(reaction, Reaction):
                warn_at_user_stacklevel(
                    f"Hook '{hook_name}' returned {reaction}, expected a Reaction.",
                    AgentWarning,
                )
                continue

            if isinstance(event, (AgentEnd, ReactStep)):
                warn_at_user_stacklevel(
                    f"Hook '{hook_name}' returned {reaction} during {type(event).__name__}, ignored.",
                    AgentWarning,
                )
                continue

            hook_reactions[hook_name] = reaction

        if not hook_reactions:
            return

        # Priority: Finish > Retry > Continue
        winning_reaction = next((r for r in hook_reactions.values() if isinstance(r, Finish)), None)
        winning_reaction = winning_reaction or next(
            (r for r in hook_reactions.values() if isinstance(r, (Retry, RetryWithFeedback))), None
        )
        winning_reaction = winning_reaction or next(
            (r for r in hook_reactions.values() if isinstance(r, Continue)), None
        )
        winning_reaction = winning_reaction or next(
            (r for r in hook_reactions.values() if r is not None), None
        )

        if winning_reaction is None:
            return

        winning_hook_name = next(
            (name for name, r in hook_reactions.items() if r is winning_reaction), "unknown"
        )

        async for _event in self._dispatch(
            ReactStep(
                agent_id=self.agent_id,
                hook_name=winning_hook_name,
                reaction=winning_reaction,
            )
        ):
            yield _event

        if isinstance(winning_reaction, Continue):
            raise Continue(messages=winning_reaction.messages)
        if isinstance(winning_reaction, RetryWithFeedback):
            messages = winning_reaction.messages
            messages.extend(Message("user", winning_reaction.feedback))
            raise Retry(messages=messages)

        raise winning_reaction

    async def _process_tool_call(
        self, tool_call: "ToolCall", step_count: int
    ) -> t.AsyncGenerator[AgentEvent, None]:
        """Process a single tool call."""
        async for event in self._dispatch(
            ToolStart(
                agent_id=self.agent_id,
                agent_name=self.name,
                status="running",
                tool_call=tool_call,
            )
        ):
            yield event

        tool = next((t for t in self.all_tools if t.name == tool_call.name), None)

        if tool is None:
            error_msg = f"Tool '{tool_call.name}' not found."
            logger.warning(error_msg)
            async for event in self._dispatch(
                ToolError(
                    agent_id=self.agent_id,
                    agent_name=self.name,
                    tool_call=tool_call,
                    error=NameError(error_msg),
                )
            ):
                yield event

            message = Message.from_model(
                SystemErrorModel(content=f"Tool '{tool_call.name}' not found.")
            )
            async for event in self._dispatch(
                ToolStep(
                    agent_id=self.agent_id,
                    agent_name=self.name,
                    status="running",
                    step=step_count,
                    messages=[message],
                    error=None,
                    stop=False,
                    tool_call=tool_call,
                )
            ):
                yield event
            return

        try:
            message, stop = await tool.handle_tool_call(tool_call)

            async for event in self._dispatch(
                ToolEnd(
                    agent_id=self.agent_id,
                    agent_name=self.name,
                    tool_call=tool_call,
                    result=message.content,
                )
            ):
                yield event

            async for event in self._dispatch(
                ToolStep(
                    agent_id=self.agent_id,
                    agent_name=self.name,
                    status="running",
                    step=step_count,
                    messages=[message],
                    error=None,
                    stop=stop,
                    tool_call=tool_call,
                )
            ):
                yield event

        except Exception as e:
            logger.opt(exception=True).error(f"Error executing tool '{tool_call.name}'")
            async for event in self._dispatch(
                ToolError(
                    agent_id=self.agent_id,
                    agent_name=self.name,
                    status="errored",
                    tool_call=tool_call,
                    error=e,
                )
            ):
                yield event
            raise

    async def evaluate(
        self,
        goal: str | None = None,
        *,
        dataset: DatasetLike | None = None,
        dataset_file: Path | str | None = None,
        scorers: ScorersLike[Trajectory] | None = None,
        assert_scores: list[str] | t.Literal[True] | None = None,
        max_attempts: int = 1,
        stop_on_success: bool = False,
        concurrency: int = 1,
        iterations: int = 1,
        max_errors: int | None = None,
        max_consecutive_errors: int = 10,
        name: str | None = None,
    ) -> EvalResult[str, Trajectory]:
        """
        Evaluate the agent against one or more goals.

        For a single goal:
            result = await agent.eval(
                "Find the flag",
                scorers=[flag_found],
                max_attempts=5,
                stop_on_success=True,
            )

        For multiple goals:
            result = await agent.eval(
                dataset=[{"goal": "..."}, ...],
                scorers=[flag_found],
                concurrency=4,
            )
        """
        if goal is not None:
            if dataset is not None:
                raise ValueError("Specify either 'goal' or 'dataset', not both")

            return await self._eval_single(
                goal=goal,
                scorers=scorers,
                assert_scores=assert_scores,
                max_attempts=max_attempts,
                stop_on_success=stop_on_success,
                name=name,
            )

        if dataset is None and dataset_file is None:
            raise ValueError("Must specify 'goal', 'dataset', or 'dataset_file'")

        return await self._eval_dataset(
            dataset=dataset,
            dataset_file=dataset_file,
            scorers=scorers,
            assert_scores=assert_scores,
            concurrency=concurrency,
            iterations=iterations,
            max_errors=max_errors,
            max_consecutive_errors=max_consecutive_errors,
            name=name,
        )

    async def _eval_single(
        self,
        name: str | None,
        goal: str,
        scorers: ScorersLike[Trajectory] | None,
        assert_scores: list[str] | t.Literal[True] | None,
        max_attempts: int,
        *,
        stop_on_success: bool,
    ) -> EvalResult[str, Trajectory]:
        """Evaluate a single goal with retry semantics."""
        from dreadnode import task_and_run

        fitted_scorers = Scorer.fit_many(scorers)
        assertion_names = (
            [s.name for s in fitted_scorers] if assert_scores is True else list(assert_scores or [])
        )

        attempts: list[Sample[str, Trajectory]] = []
        best_sample: Sample | None = None

        eval_name = name or f"eval-{self.name}"

        with task_and_run(eval_name, task_type="evaluation", tags=["evaluation"]):
            for attempt in range(max_attempts):
                # Reset environment for retry
                if self.environment and attempt > 0:
                    await self.environment.reset()

                # Run agent
                trajectory = await self.run(goal)

                # Score the trajectory
                scores = {}
                for scorer in fitted_scorers:
                    score_value = scorer(trajectory)
                    if inspect.isawaitable(score_value):
                        score_value = await score_value
                    scores[scorer.name] = score_value

                # Check assertions
                passed = all(scores.get(name, 0) > 0 for name in assertion_names)

                sample = Sample(
                    input=goal,
                    output=trajectory,
                    scores=scores,
                    passed=passed,
                    index=attempt,
                )
                attempts.append(sample)

                if best_sample is None or sample.score > best_sample.score:
                    best_sample = sample

                if passed and stop_on_success:
                    break

            return EvalResult(
                scenarios=[
                    ScenarioResult(
                        params={},
                        iterations=[
                            IterationResult(
                                iteration=1,
                                samples=attempts,
                            )
                        ],
                    )
                ],
                stop_reason="finished",
            )

    async def _eval_dataset(
        self,
        dataset: DatasetLike | None,
        dataset_file: Path | str | None,
        scorers: ScorersLike[Trajectory] | None,
        assert_scores: list[str] | t.Literal[True] | None,
        concurrency: int,
        iterations: int,
        max_errors: int | None,
        max_consecutive_errors: int,
        name: str | None,
    ) -> EvalResult[str, Trajectory]:
        """Evaluate against a dataset."""

        # Wrap agent.run as a task
        @task(name=self.name)
        async def agent_task(goal: str) -> Trajectory:
            return await self.run(goal)

        # Delegate to Evaluation
        evaluation = Evaluation(
            task=agent_task,
            dataset=dataset,
            dataset_file=dataset_file,
            dataset_input_mapping=["goal"],
            scorers=scorers or [],
            assert_scores=assert_scores or [],
            concurrency=concurrency,
            iterations=iterations,
            max_errors=max_errors,
            max_consecutive_errors=max_consecutive_errors,
            name=name or f"eval-{self.name}",
        )

        return await evaluation.run()

    async def optimize(
        self,
        goal: str | None = None,
        *,
        dataset: DatasetLike | None = None,
        dataset_file: Path | str | None = None,
        search: SearchLike,
        search_strategy: t.Literal["grid", "random", "optuna", "auto"] = "auto",
        scorers: ScorersLike[Trajectory],
        directions: list[Direction] | None = None,
        # Per-trial settings
        max_attempts: int = 1,
        # Study settings
        max_trials: int = 100,
        concurrency: int = 1,
        constraints: ScorersLike | None = None,
        stop_conditions: list[StudyStopCondition] | None = None,
        name: str | None = None,
    ) -> tuple[te.Self, StudyResult]:
        """
        Optimize agent configuration.

        Single goal:
            best, result = await agent.optimize(
                "Find the flag",
                scorers=[flag_found],
                search={"model": [...], "max_steps": [...]},
            )

        Dataset:
            best, result = await agent.optimize(
                dataset=[...],
                scorers=[accuracy],
                search={"model": [...], "instructions": [...]},
            )
        """

        # Resolve search strategy
        search_instance = self._resolve_search(search, search_strategy, max_trials)

        # Factory creates agent variants
        def agent_factory(candidate: dict[str, t.Any]) -> Task:
            modified = self.with_(**candidate)

            @task(name=modified.name)
            async def run_agent(goal: str) -> Trajectory:
                # Handle retries for stochastic evaluation
                best_trajectory = None
                for attempt in range(max_attempts):
                    if modified.environment and attempt > 0:
                        await modified.environment.reset()

                    trajectory = await modified.run(goal)
                    if best_trajectory is None:
                        best_trajectory = trajectory
                    # Could add logic to keep best based on some criteria

                return best_trajectory

            return run_agent

        # Normalize goal to dataset
        eval_dataset = dataset
        if goal is not None:
            eval_dataset = [{"goal": goal}]

        study = Study(
            name=name or f"optimize-{self.name}",
            search_strategy=search_instance,
            task_factory=agent_factory,
            dataset=eval_dataset,
            dataset_file=dataset_file,
            objectives=scorers,
            directions=directions or ["maximize"] * len(Scorer.fit_many(scorers)),
            constraints=constraints,
            max_trials=max_trials,
            concurrency=concurrency,
            stop_conditions=stop_conditions or [],
        )

        result = await study.run()

        if result.best_trial is None:
            raise RuntimeError("Optimization failed - no successful trials")

        best_agent = self.with_(**result.best_trial.candidate)
        return best_agent, result

    def _resolve_search(
        self,
        search: SearchLike,
        strategy: str,
        max_trials: int,
    ) -> Search[dict[str, t.Any]]:
        """Convert search specification to a Search instance."""

        # Already a Search instance
        if isinstance(search, Search):
            return search

        # Check if it's a simple grid (all values are lists, no distributions)
        is_simple_grid = all(
            isinstance(v, list) and not isinstance(v, (Float, Int, Categorical))
            for v in search.values()
        )

        # Auto-select strategy
        if strategy == "auto":
            strategy = "grid" if is_simple_grid else "optuna"

        # Build the search
        if strategy == "grid":
            if not is_simple_grid:
                raise ValueError(
                    "Grid search requires dict[str, list[Any]]. "
                    "For distributions (Float, Int, Categorical), use 'random' or 'optuna'."
                )
            return grid(search)

        if strategy == "random":
            return random(search, max_iterations=max_trials)

        if strategy == "optuna":
            return optuna_(search)

        raise ValueError(f"Unknown search strategy: {strategy}")


def agent(
    func: t.Callable[..., t.Any] | None = None,
    /,
    *,
    name: str | None = None,
    model: str | Generator | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    label: str | None = None,
    max_steps: int = 10,
    cache: caching.CacheMode | None = None,
    tools: list[Tool | Toolset | t.Callable[..., t.Any]] | None = None,
    tool_mode: ToolMode = "auto",
    stop_conditions: list[StopCondition] | None = None,
    hooks: list[Hook] | None = None,
    judge: Judge[Rubric] | None = None,
    environment: Environment | None = None,
) -> Agent | t.Callable[[t.Callable[..., t.Any]], Agent]:
    """
    Create an Agent from a function. The function's docstring becomes the instructions.

    Can be used without arguments:
        ```python
        @dreadnode.agent
        async def my_agent():
            \"\"\"You are a helpful assistant.\"\"\"
            pass
        ```

    Or with arguments:
        ```python
        @dreadnode.agent(model="anthropic/claude-3-5-sonnet", tools=[my_tool])
        async def my_agent():
            \"\"\"You are a helpful assistant.\"\"\"
            pass

        result = await my_agent.run("Help me with something")
        ```

    Args:
        func: The function to convert to an agent.
        name: The agent name (defaults to function name).
        model: Inference model identifier.
        description: A brief description of the agent.
        tags: Tags associated with the agent.
        label: An optional label for the agent.
        max_steps: Maximum number of steps.
        cache: Cache mode for inference.
        tools: Tools the agent can use.
        tool_mode: The tool calling mode.
        stop_conditions: Conditions for stopping.
        hooks: Hooks to apply during execution.
        judge: A judge for evaluation.
        environment: The environment for the agent.

    Returns:
        An Agent instance or a decorator function.
    """
    from dreadnode.core.util import get_callable_name

    def make_agent(fn: t.Callable[..., t.Any]) -> Agent:
        """Create an agent from a function definition."""
        fn_name = get_callable_name(fn, short=True)
        instructions = inspect.getdoc(fn) or ""

        return Agent(
            name=name or fn_name,
            model=model,
            instructions=instructions,
            description=description or "",
            tags=tags or ["agent"],
            label=label,
            max_steps=max_steps,
            cache=cache,
            tools=tools or [],
            tool_mode=tool_mode,
            stop_conditions=stop_conditions or [],
            hooks=hooks or [],
            judge=judge,
            environment=environment,
        )

    # Called as @agent on a function directly
    if func is not None:
        return make_agent(func)

    # Called as @agent() or @agent(model="...", ...) - return decorator
    return make_agent
