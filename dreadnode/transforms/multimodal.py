from __future__ import annotations

import typing as t
from collections.abc import Awaitable, Callable

from loguru import logger

from dreadnode.data_types import Audio, Image, Text, Video
from dreadnode.data_types.message import Message as DnMessage
from dreadnode.transforms.base import Transform

ContentPart: t.TypeAlias = Text | Image | Audio | Video | str

HandlerFunc: t.TypeAlias = Callable[[ContentPart, Transform], Awaitable[ContentPart]]

_CONTENT_TYPE_HANDLERS: dict[type, HandlerFunc] = {}


def _register_handler(content_type: type) -> Callable[[HandlerFunc], HandlerFunc]:
    """Decorator to register a handler for a content type."""

    def decorator(func: HandlerFunc) -> HandlerFunc:
        _CONTENT_TYPE_HANDLERS[content_type] = func
        return func

    return decorator


@_register_handler(str)
async def _transform_str(part: ContentPart, transform: Transform) -> ContentPart:
    """Transform a plain string."""
    if not isinstance(part, str):
        return part
    result = await transform.transform(part)
    return result if isinstance(result, str) else part


@_register_handler(Text)
async def _transform_text(part: ContentPart, transform: Transform) -> ContentPart:
    """Transform a Text object using its underlying string."""
    if not isinstance(part, Text):
        return part
    result = await transform.transform(part._text)  # noqa: SLF001
    return Text(result, part._format) if isinstance(result, str) else part  # noqa: SLF001


def _make_typed_handler(expected_type: type) -> HandlerFunc:
    """Create a handler that validates transform result matches the expected type."""

    async def handler(part: ContentPart, transform: Transform) -> ContentPart:
        result = await transform.transform(part)
        return t.cast("ContentPart", result) if isinstance(result, expected_type) else part

    return handler  # type: ignore[return-value]


# Bulk register typed handlers
for content_type in (Image, Audio, Video):
    _register_handler(content_type)(_make_typed_handler(content_type))  # type: ignore[arg-type]


def _get_handler_for_part(part: ContentPart) -> HandlerFunc | None:
    """Find the registered handler for a content part type."""
    for content_type, handler in _CONTENT_TYPE_HANDLERS.items():
        if isinstance(part, content_type):
            return handler
    return None


async def apply_transform_to_part(
    part: ContentPart,
    transform: Transform,
) -> ContentPart:
    """
    Apply a single transform to a content part.

    Returns the original part if no handler exists or transform fails.
    """
    handler = _get_handler_for_part(part)
    if handler is None:
        return part

    try:
        return await handler(part, transform)
    except Exception as e:  # noqa: BLE001
        logger.trace(
            f"Transform '{transform.name}' not applicable to {type(part).__name__}: {str(e)[:100]}"
        )
        return part


async def apply_transforms_to_message(
    message: DnMessage,
    transforms: list[Transform],
) -> DnMessage:
    """Apply multiple transforms to a Message's content parts."""
    if not transforms:
        return message

    new_content: list[ContentPart] = []
    for part in message.content:
        transformed_part = part
        for transform in transforms:
            transformed_part = await apply_transform_to_part(transformed_part, transform)
        new_content.append(transformed_part)

    return DnMessage(
        role=message.role,
        content=new_content,
        metadata=message.metadata.copy(),
        uuid=message.uuid,
        tool_calls=message.tool_calls,
        tool_call_id=message.tool_call_id,
    )


async def apply_transforms_to_value(
    value: t.Any,
    transforms: list[Transform],
) -> t.Any:
    """
    Apply transforms to any value.

    Messages get per-part transformation, other values get direct transformation.
    """
    if not transforms:
        return value

    if isinstance(value, DnMessage):
        return await apply_transforms_to_message(value, transforms)

    result = value
    for transform in transforms:
        try:
            result = await transform.transform(result)
        except Exception as e:  # noqa: BLE001, PERF203
            logger.trace(f"Transform '{transform.name}' skipped: {str(e)[:100]}")
    return result


async def apply_transforms_to_kwargs(
    kwargs: dict[str, t.Any],
    transforms: list[Transform],
) -> dict[str, t.Any]:
    """Apply transforms to all kwargs values."""
    if not transforms:
        return kwargs

    return {
        key: await apply_transforms_to_value(value, transforms) for key, value in kwargs.items()
    }


# Explicitly mark handlers as used (for static analysis)
_ = (_transform_str, _transform_text)
