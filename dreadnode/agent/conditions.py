import typing as t

from dreadnode.agent.events import AgentError, AgentEvent, ToolEnd, ToolStart
from dreadnode.agent.parsing import try_parse, try_parse_many
from dreadnode.scorers import Scorer

AgentConditionT = t.TypeVar("AgentConditionT", bound="AgentEvent")
Condition = t.Callable[[AgentConditionT], bool]

ModelT = t.TypeVar("ModelT")


class FieldConditionBuilder:
    """Represents a specific field of a model, ready for comparison."""

    def __init__(self, model_type: type, field_name: str):
        self.model_type = model_type
        self.field_name = field_name

    def __hash__(self):
        return hash((self.model_type, self.field_name))

    def __eq__(self, value: object) -> Condition:
        """Creates the final Condition function for an equality check."""

        def condition(event: AgentEvent) -> bool:
            model = try_parse(event, self.model_type)
            if model is None:
                return False  # If the model wasn't found, the condition is false

            field_value = getattr(model, self.field_name, None)
            return field_value == value

        return condition


class ModelConditionBuilder(t.Generic[ModelT]):
    """Represents the start of a fluent condition chain for a model."""

    def __init__(self, model_type: type[ModelT]):
        self.model_type = model_type

    def __getattr__(self, name: str) -> FieldConditionBuilder:
        # For simple checks: when(MyModel).field == value
        return FieldConditionBuilder(self.model_type, name)

    def satisfies(self, scorer: Scorer[ModelT]) -> Condition:
        """
        Creates a Condition that passes if the extracted model satisfies
        the logic of the given Scorer.

        This is the single, clear path for all complex evaluations.
        """

        def condition(event: AgentEvent) -> bool:
            model = try_parse(event, self.model_type)
            if model is None:
                return False

            metric = scorer.evaluate(model)
            return metric.value > 0.0

        return condition

    def contains(self, substring: str, *, case_sensitive: bool = False) -> Condition:
        """
        Creates a Condition that passes if the extracted model (assumed to be a string)
        contains the given substring.
        """

        def condition(model: ModelT) -> bool:
            model = try_parse(model, self.model_type)
            if model is None or not isinstance(model, str):
                return False

            if not case_sensitive:
                return substring.lower() in model.lower()
            return substring in model

        return condition


class ListConditionBuilder(t.Generic[ModelT]):
    """Builds a condition by applying a check to a list of models."""

    def __init__(self, model_type: type[ModelT], aggregator: t.Callable[[t.Iterable], bool]):
        self.model_type = model_type
        self.aggregator = aggregator

    def satisfies(self, scorer: Scorer[ModelT]) -> Condition:
        """Applies a Scorer to each model and aggregates the boolean results."""

        def condition(event: AgentEvent) -> bool:
            models = try_parse_many(event, self.model_type)
            if not models:
                return False
            results = (scorer.evaluate(model).value > 0.0 for model in models)
            return self.aggregator(results)

        return condition


class ListFieldConditionBuilder(t.Generic[ModelT]):
    """Represents a specific field on a list of models, ready for comparison."""

    def __init__(self, parent: ListConditionBuilder[ModelT], field_name: str):
        self.parent = parent
        self.field_name = field_name

    def __hash__(self):
        return hash((self.parent, self.field_name))

    def __eq__(self, value: object) -> Condition:
        """Creates the final Condition for an equality check on a list of models."""

        def condition(event: AgentEvent) -> bool:
            models = try_parse_many(event, self.parent.model_type)
            if not models:
                return False
            results = (getattr(model, self.field_name, None) == value for model in models)
            return self.parent.aggregator(results)

        return condition


def when(model_type: type[ModelT]) -> ModelConditionBuilder[ModelT]:
    """Entry point for conditions that operate on a single model instance."""
    return ModelConditionBuilder(model_type)


def when_any(model_type: type[ModelT]) -> ListConditionBuilder[ModelT]:
    """Entry point for conditions that are true if ANY model in a set matches."""
    return ListConditionBuilder(model_type, any)


def when_all(model_type: type[ModelT]) -> ListConditionBuilder[ModelT]:
    """Entry point for conditions that are true only if ALL models in a set match."""
    return ListConditionBuilder(model_type, all)


def is_file_not_found_error(event: AgentError) -> bool:
    """Returns True if the agent's error is a FileNotFoundError."""
    return isinstance(event.error, FileNotFoundError)


def tool_called(tool_name: str | t.Collection[str]) -> Condition[ToolStart | ToolEnd]:
    """
    Returns a condition function that checks if the event is for a specific tool.

    Args:
        tool_name: A single tool name or a collection of tool names to match.
    """
    names = {tool_name} if isinstance(tool_name, str) else set(tool_name)

    def condition(event: ToolStart | ToolEnd) -> bool:
        return event.tool_call.name in names

    # Give the returned function a descriptive name for easier debugging
    condition.__name__ = f"tool_is_{'_or_'.join(names)}"
    return condition


def error_contains_text(text: str, *, case_sensitive: bool = False) -> Condition[AgentError]:
    """
    Returns a condition that checks if the event's error message contains specific text.
    """

    def condition(event: AgentError) -> bool:
        error_str = str(event.error)
        if not case_sensitive:
            return text.lower() in error_str.lower()
        return text in error_str

    condition.__name__ = f"error_contains_{text.replace(' ', '_').lower()}"
    return condition
