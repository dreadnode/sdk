import importlib
import typing as t

from dreadnode.scorers.base import (
    Scorer,
    ScorerCallable,
    ScorerLike,
    ScorerResult,
    ScorersLike,
    ScorerWarning,
    add,
    and_,
    avg,
    clip,
    equals,
    forward,
    invert,
    normalize,
    not_,
    or_,
    remap_range,
    scale,
    subtract,
    task_input,
    task_output,
    threshold,
    weighted_avg,
)
from dreadnode.scorers.classification import detect_refusal_with_zero_shot, zero_shot_classification
from dreadnode.scorers.consistency import character_consistency
from dreadnode.scorers.contains import (
    contains,
    detect_ansi_escapes,
    detect_bias,
    detect_refusal,
    detect_sensitive_keywords,
    detect_unsafe_shell_content,
)
from dreadnode.scorers.format import is_json, is_xml
from dreadnode.scorers.length import length_in_range, length_ratio, length_target
from dreadnode.scorers.lexical import type_token_ratio
from dreadnode.scorers.pii import detect_pii, detect_pii_with_presidio
from dreadnode.scorers.readability import readability
from dreadnode.scorers.sentiment import sentiment, sentiment_with_perspective

if t.TYPE_CHECKING:
    from dreadnode.scorers.crucible import contains_crucible_flag
    from dreadnode.scorers.harm import detect_harm_with_openai
    from dreadnode.scorers.image import image_distance
    from dreadnode.scorers.json import json_path
    from dreadnode.scorers.judge import llm_judge
    from dreadnode.scorers.rigging import adapt_messages, make_messages_adapter, wrap_chat
    from dreadnode.scorers.similarity import (
        bleu,
        similarity,
        similarity_with_litellm,
        similarity_with_sentence_transformers,
        similarity_with_tf_idf,
    )

__all__ = [
    "Scorer",
    "ScorerCallable",
    "ScorerLike",
    "ScorerResult",
    "ScorerWarning",
    "ScorersLike",
    "adapt_messages",
    "add",
    "and_",
    "avg",
    "bleu",
    "character_consistency",
    "clip",
    "contains",
    "contains_crucible_flag",
    "detect_ansi_escapes",
    "detect_bias",
    "detect_harm_with_openai",
    "detect_pii",
    "detect_pii_with_presidio",
    "detect_refusal",
    "detect_refusal_with_zero_shot",
    "detect_sensitive_keywords",
    "detect_unsafe_shell_content",
    "equals",
    "forward",
    "image_distance",
    "invert",
    "is_json",
    "is_xml",
    "json_path",
    "length_in_range",
    "length_ratio",
    "length_target",
    "llm_judge",
    "make_messages_adapter",
    "normalize",
    "not_",
    "or_",
    "readability",
    "remap_range",
    "scale",
    "sentiment",
    "sentiment_with_perspective",
    "similarity",
    "similarity_with_litellm",
    "similarity_with_sentence_transformers",
    "similarity_with_tf_idf",
    "subtract",
    "task_input",
    "task_output",
    "threshold",
    "type_token_ratio",
    "weighted_avg",
    "wrap_chat",
    "zero_shot_classification",
]

__lazy_submodules__: list[str] = []
__lazy_components__: dict[str, str] = {
    "llm_judge": "dreadnode.scorers.judge",
    "wrap_chat": "dreadnode.scorers.rigging",
    "adapt_messages": "dreadnode.scorers.rigging",
    "make_messages_adapter": "dreadnode.scorers.rigging",
    "detect_harm_with_openai": "dreadnode.scorers.harm",
    "contains_crucible_flag": "dreadnode.scorers.crucible",
    "similarity": "dreadnode.scorers.similarity",
    "similarity_with_sentence_transformers": "dreadnode.scorers.similarity",
    "similarity_with_tf_idf": "dreadnode.scorers.similarity",
    "similarity_with_litellm": "dreadnode.scorers.similarity",
    "bleu": "dreadnode.scorers.similarity",
    "json_path": "dreadnode.scorers.json",
    "image_distance": "dreadnode.scorers.image",
}


def __getattr__(name: str) -> t.Any:
    if name in __lazy_submodules__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    if name in __lazy_components__:
        module_name = __lazy_components__[name]
        module = importlib.import_module(module_name)
        component = getattr(module, name)
        globals()[name] = component
        return component

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
