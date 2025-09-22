import typing as t

import rigging.parsing as parse
from rigging.model import Model as RiggingModel

from dreadnode.agent.events import AgentEvent, GenerationEnd

AgentMessage = RiggingModel
ModelT = t.TypeVar("ModelT")


def try_parse(text: str, model_type: type[ModelT]) -> ModelT | None:
    """Extracts a single model instance from a GenerationEnd event, if possible."""
    model = parse.try_parse(text, model_type)
    return model if model else None


def try_parse_many(event: AgentEvent, model_type: type[ModelT]) -> list[ModelT]:
    """Generic extractor using rigging.parse.try_parse_set on GenerationEnd events."""
    if isinstance(event, GenerationEnd):
        models = parse.try_parse_many(str(event.message.content), model_type)
    return models if models else []
