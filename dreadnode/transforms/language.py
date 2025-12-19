import typing as t

import rigging as rg

from dreadnode.common_types import AnyDict
from dreadnode.meta import Config
from dreadnode.transforms.base import Transform


def adapt_language(
    target_language: str,
    *,
    adapter_model: str | rg.Generator,
    style: t.Literal["formal", "casual", "technical", "colloquial"] = "formal",
    preserve_meaning: bool = True,
    model_params: rg.GenerateParams | AnyDict | None = None,
    system_prompt: str | None = None,
    name: str = "adapt_language",
) -> Transform[str, str]:
    """
    Adapts text to a target language while optionally adjusting style and formality.

    This transform uses an LLM to perform intelligent language adaptation that goes
    beyond word-for-word translation. It can adjust for cultural context, idiomatic
    expressions, and linguistic style.

    Args:
        target_language: The target language (e.g., "Spanish", "Mandarin Chinese", "Arabic")
        adapter_model: The LLM to use for adaptation
        style: The linguistic style to use:
            - "formal": Professional, standardized language
            - "casual": Informal, conversational language
            - "technical": Domain-specific, precise terminology
            - "colloquial": Local dialects, slang, regional expressions
        preserve_meaning: If True, prioritize semantic accuracy over natural flow
        model_params: Optional parameters for the adapter model
        system_prompt: Custom system prompt (uses default if None)
        name: Name of the transform

    Examples:
        ```python
        # Test Spanish formal language
        spanish_formal = adapt_language("Spanish", adapter_model="groq/llama-3.1-70b")

        # Test Mandarin colloquial style
        mandarin_casual = adapt_language(
            "Mandarin Chinese",
            adapter_model="groq/llama-3.1-70b",
            style="colloquial"
        )

        # Test Arabic technical terminology
        arabic_tech = adapt_language(
            "Arabic",
            adapter_model="groq/llama-3.1-70b",
            style="technical"
        )
        ```
    """

    default_system_prompt = """You are a professional language adaptation specialist. Your task is to adapt text from one language to another while considering cultural context, linguistic nuances, and stylistic preferences.

When adapting text:
- Maintain the core meaning and intent
- Adjust for cultural appropriateness
- Use natural expressions in the target language
- Consider regional variations when specified
- Preserve any technical terms appropriately
- Maintain the requested style and formality level

Return ONLY the adapted text without explanations or meta-commentary."""

    async def transform(
        text: str,
        *,
        target_language: str = Config(target_language, help="The target language for adaptation"),
        style: t.Literal["formal", "casual", "technical", "colloquial"] = Config(
            style, help="The linguistic style to apply"
        ),
        preserve_meaning: bool = Config(
            preserve_meaning, help="Whether to prioritize semantic accuracy"
        ),
    ) -> str:
        generator: rg.Generator
        if isinstance(adapter_model, str):
            generator = rg.get_generator(
                adapter_model,
                params=model_params
                if isinstance(model_params, rg.GenerateParams)
                else rg.GenerateParams.model_validate(model_params)
                if model_params
                else None,
            )
        else:
            generator = adapter_model

        style_guidance = {
            "formal": "Use formal, professional language appropriate for official communication.",
            "casual": "Use informal, conversational language as spoken among friends.",
            "technical": "Use precise technical terminology appropriate for domain experts.",
            "colloquial": "Use local dialects, slang, and regional expressions common in everyday speech.",
        }

        meaning_guidance = (
            "Prioritize exact semantic accuracy, even if it sounds less natural."
            if preserve_meaning
            else "Prioritize natural, idiomatic expression in the target language."
        )

        user_prompt = f"""Adapt the following text to {target_language}.

Style: {style_guidance[style]}
Approach: {meaning_guidance}

Text to adapt:
===BEGIN===
{text}
===END===

Provide only the adapted text in {target_language}."""

        chat = generator.chat(
            [
                rg.Message(role="system", content=system_prompt or default_system_prompt),
                rg.Message(role="user", content=user_prompt),
            ]
        )

        response = await chat.run()
        adapted_text = response.last.content

        if not isinstance(adapted_text, str):
            adapted_text = str(adapted_text)

        adapted_text = adapted_text.strip()

        # Remove any markdown code blocks if present
        if adapted_text.startswith("```") and adapted_text.endswith("```"):
            lines = adapted_text.split("\n")
            adapted_text = "\n".join(lines[1:-1]).strip()

        return adapted_text

    return Transform(transform, name=name)


def transliterate(
    script: t.Literal["cyrillic", "arabic", "katakana", "hangul", "devanagari"] | None = None,
    *,
    custom_mapping: dict[str, str] | None = None,
    fallback_char: str | None = None,
    preserve_case: bool = True,
    name: str = "transliterate",
) -> Transform[str, str]:
    """
    Converts Latin script to other writing systems phonetically.

    Tests model handling of different scripts and character encodings.
    Useful for bypassing text-based filters that only check Latin characters.

    Args:
        script: Target script for transliteration (if None, must provide custom_mapping)
        custom_mapping: Custom character mapping dictionary. If provided, overrides script.
        fallback_char: Character to use when no mapping exists (None = keep original)
        preserve_case: If True, attempts to preserve uppercase distinction where possible
        name: Name of the transform

    Examples:
        ```python
        # Convert to Cyrillic using built-in mapping
        cyrillic = transliterate("cyrillic")
        # "Hello" -> "Хелло"

        # Convert to Arabic script
        arabic = transliterate("arabic")
        # "Hello" -> "هيللو"

        # Custom leet-speak mapping
        leet = transliterate(
            custom_mapping={
                "a": "4", "e": "3", "i": "1",
                "o": "0", "s": "5", "t": "7"
            }
        )
        # "Hello" -> "H3ll0"

        # Custom ROT13-style mapping
        rot13 = transliterate(
            custom_mapping={
                "a": "n", "b": "o", "c": "p", "d": "q",
                "e": "r", "f": "s", "g": "t", "h": "u",
                "i": "v", "j": "w", "k": "x", "l": "y",
                "m": "z", "n": "a", "o": "b", "p": "c",
                "q": "d", "r": "e", "s": "f", "t": "g",
                "u": "h", "v": "i", "w": "j", "x": "k",
                "y": "l", "z": "m"
            }
        )

        # Custom mapping with fallback
        custom = transliterate(
            custom_mapping={"a": "@", "e": "€", "i": "!", "o": "0"},
            fallback_char="*"
        )
        # "Hello" -> "H€ll0" (no fallback needed) or "H€ll0" with fallback="*"

        # Mix built-in with custom overrides
        # Use built-in Cyrillic but override specific characters
        custom_cyrillic = transliterate(
            script="cyrillic",
            custom_mapping={"x": "икс", "w": "дабл-ю"}  # Override defaults
        )
        ```

    Raises:
        ValueError: If neither script nor custom_mapping is provided
    """

    # Built-in mapping tables for phonetic transliteration
    builtin_mappings = {
        "cyrillic": {
            "a": "а",
            "b": "б",
            "c": "к",
            "d": "д",
            "e": "е",
            "f": "ф",
            "g": "г",
            "h": "х",
            "i": "и",
            "j": "й",
            "k": "к",
            "l": "л",
            "m": "м",
            "n": "н",
            "o": "о",
            "p": "п",
            "q": "к",
            "r": "р",
            "s": "с",
            "t": "т",
            "u": "у",
            "v": "в",
            "w": "в",
            "x": "кс",
            "y": "й",
            "z": "з",
        },
        "arabic": {
            "a": "ا",
            "b": "ب",
            "c": "ك",
            "d": "د",
            "e": "ي",
            "f": "ف",
            "g": "غ",
            "h": "ه",
            "i": "ي",
            "j": "ج",
            "k": "ك",
            "l": "ل",
            "m": "م",
            "n": "ن",
            "o": "و",
            "p": "ب",
            "q": "ق",
            "r": "ر",
            "s": "س",
            "t": "ت",
            "u": "و",
            "v": "ف",
            "w": "و",
            "x": "كس",
            "y": "ي",
            "z": "ز",
        },
        "katakana": {
            "a": "ア",
            "b": "ブ",
            "c": "ク",
            "d": "ド",
            "e": "エ",
            "f": "フ",
            "g": "グ",
            "h": "ハ",
            "i": "イ",
            "j": "ジ",
            "k": "ク",
            "l": "ル",
            "m": "ム",
            "n": "ン",
            "o": "オ",
            "p": "プ",
            "q": "ク",
            "r": "ル",
            "s": "ス",
            "t": "ト",
            "u": "ウ",
            "v": "ブ",
            "w": "ワ",
            "x": "クス",
            "y": "ヤ",
            "z": "ズ",
        },
        "hangul": {
            "a": "아",
            "b": "브",
            "c": "크",
            "d": "드",
            "e": "에",
            "f": "프",
            "g": "그",
            "h": "흐",
            "i": "이",
            "j": "즈",
            "k": "크",
            "l": "르",
            "m": "므",
            "n": "느",
            "o": "오",
            "p": "프",
            "q": "크",
            "r": "르",
            "s": "스",
            "t": "트",
            "u": "우",
            "v": "브",
            "w": "워",
            "x": "크스",
            "y": "야",
            "z": "즈",
        },
        "devanagari": {
            "a": "अ",
            "b": "ब",
            "c": "क",
            "d": "द",
            "e": "ए",
            "f": "फ",
            "g": "ग",
            "h": "ह",
            "i": "इ",
            "j": "ज",
            "k": "क",
            "l": "ल",
            "m": "म",
            "n": "न",
            "o": "ओ",
            "p": "प",
            "q": "क",
            "r": "र",
            "s": "स",
            "t": "त",
            "u": "उ",
            "v": "व",
            "w": "व",
            "x": "क्स",
            "y": "य",
            "z": "ज़",
        },
    }

    if script is None and custom_mapping is None:
        raise ValueError(
            "Either 'script' or 'custom_mapping' must be provided. "
            "Available scripts: cyrillic, arabic, katakana, hangul, devanagari"
        )

    if script is not None:
        if script not in builtin_mappings:
            raise ValueError(
                f"Unknown script: {script}. Available scripts: {', '.join(builtin_mappings.keys())}"
            )
        mapping = builtin_mappings[script].copy()

        if custom_mapping:
            mapping.update(custom_mapping)
    else:
        mapping = custom_mapping or {}

    # Add uppercase mappings if preserve_case is True
    if preserve_case:
        uppercase_mapping = {}
        for char, transliterated in mapping.items():
            if char.islower():
                # For scripts without case (Arabic, Katakana, etc.), keep same
                uppercase_mapping[char.upper()] = transliterated
        mapping.update(uppercase_mapping)

    def transform(
        text: str,
        *,
        fallback_char: str | None = Config(
            fallback_char, help="Character to use when no mapping exists"
        ),
    ) -> str:
        result = []
        for char in text:
            if char in mapping:
                result.append(mapping[char])
            elif fallback_char is not None:
                # Only use fallback for alphanumeric characters
                if char.isalnum():
                    result.append(fallback_char)
                else:
                    result.append(char)
            else:
                # Keep original character
                result.append(char)
        return "".join(result)

    return Transform(transform, name=name)


def code_switch(
    languages: list[str],
    *,
    adapter_model: str | rg.Generator,
    switch_ratio: float = 0.3,
    model_params: rg.GenerateParams | AnyDict | None = None,
    seed: int | None = None,
    name: str = "code_switch",
) -> Transform[str, str]:
    """
    Mixes multiple languages in a single text (code-switching).

    Tests model handling of multilingual input and context switching.
    Common in multilingual communities and social media.

    Args:
        languages: List of languages to mix (e.g., ["English", "Spanish", "Tagalog"])
        adapter_model: The LLM to use for generating code-switched text
        switch_ratio: Proportion of text to switch (0.0-1.0)
        model_params: Optional parameters for the model
        seed: Random seed for reproducibility (reserved for future use)
        name: Name of the transform

    Examples:
        ```python
        # Mix English and Spanish (Spanglish)
        spanglish = code_switch(
            ["English", "Spanish"],
            adapter_model="groq/llama-3.1-70b",
            switch_ratio=0.4
        )

        # Mix English, Hindi, and Urdu (common in South Asia)
        hinglish = code_switch(
            ["English", "Hindi", "Urdu"],
            adapter_model="groq/llama-3.1-70b"
        )
        ```
    """

    if not 0.0 <= switch_ratio <= 1.0:
        raise ValueError("switch_ratio must be between 0.0 and 1.0")

    if len(languages) < 2:
        raise ValueError("At least 2 languages required for code-switching")

    _ = seed  # Reserved for future use

    async def transform(
        text: str,
        *,
        languages: list[str] = Config(languages, help="Languages to mix in the text"),  # noqa: B008
        switch_ratio: float = Config(
            switch_ratio, ge=0.0, le=1.0, help="Proportion of text to switch"
        ),
    ) -> str:
        generator: rg.Generator
        if isinstance(adapter_model, str):
            generator = rg.get_generator(
                adapter_model,
                params=model_params
                if isinstance(model_params, rg.GenerateParams)
                else rg.GenerateParams.model_validate(model_params)
                if model_params
                else None,
            )
        else:
            generator = adapter_model

        langs_str = ", ".join(languages)

        system_prompt = f"""You are a multilingual language expert specializing in code-switching. Create natural code-switched text that mixes {langs_str} as would occur in real multilingual communities.

Rules:
- Switch languages naturally at phrase or clause boundaries
- Maintain grammatical correctness in each language
- Use authentic mixing patterns common among bilingual speakers
- Keep the core meaning intact
- Mix approximately {switch_ratio * 100:.0f}% of the content across languages

Return ONLY the code-switched text."""

        user_prompt = f"""Create a code-switched version of this text mixing {langs_str}:

{text}

Provide only the code-switched text."""

        chat = generator.chat(
            [
                rg.Message(role="system", content=system_prompt),
                rg.Message(role="user", content=user_prompt),
            ]
        )

        response = await chat.run()
        result_text = response.last.content

        if not isinstance(result_text, str):
            result_text = str(result_text)

        return result_text.strip()

    return Transform(transform, name=name)


def dialectal_variation(
    dialect: str,
    *,
    adapter_model: str | rg.Generator,
    intensity: t.Literal["light", "moderate", "heavy"] = "moderate",
    model_params: rg.GenerateParams | AnyDict | None = None,
    name: str = "dialectal_variation",
) -> Transform[str, str]:
    """
    Adapts text to specific regional dialects or variations.

    Tests model understanding of dialectal differences and regional expressions.
    Useful for evaluating bias toward standard vs. non-standard language varieties.

    Args:
        dialect: Target dialect (e.g., "AAVE", "Cockney", "Singaporean English")
        adapter_model: The LLM to use for dialect adaptation
        intensity: How heavily to apply dialectal features
        model_params: Optional parameters for the model
        name: Name of the transform

    Examples:
        ```python
        # Convert to AAVE (African American Vernacular English)
        aave = dialectal_variation(
            "African American Vernacular English",
            adapter_model="groq/llama-3.1-70b",
            intensity="moderate"
        )

        # Convert to Singaporean English (Singlish)
        singlish = dialectal_variation(
            "Singaporean English",
            adapter_model="groq/llama-3.1-70b"
        )
        ```
    """

    async def transform(
        text: str,
        *,
        dialect: str = Config(dialect, help="The target dialect or regional variation"),
        intensity: t.Literal["light", "moderate", "heavy"] = Config(
            intensity, help="How heavily to apply dialectal features"
        ),
    ) -> str:
        generator: rg.Generator
        if isinstance(adapter_model, str):
            generator = rg.get_generator(
                adapter_model,
                params=model_params
                if isinstance(model_params, rg.GenerateParams)
                else rg.GenerateParams.model_validate(model_params)
                if model_params
                else None,
            )
        else:
            generator = adapter_model

        intensity_guidance = {
            "light": "Apply subtle dialectal features while keeping most of the text standard.",
            "moderate": "Use clear dialectal features balanced with comprehensibility.",
            "heavy": "Apply strong dialectal features including vocabulary, grammar, and phonetic spelling.",
        }

        system_prompt = f"""You are a linguistics expert specializing in dialectal variations. Adapt text to authentic {dialect} while maintaining the core meaning.

Intensity: {intensity_guidance[intensity]}

Use authentic features of {dialect} including:
- Vocabulary and expressions
- Grammatical structures
- Phonetic representations where appropriate
- Cultural references and idioms

Keep the adaptation natural and respectful. Return ONLY the adapted text."""

        user_prompt = f"""Adapt this text to {dialect}:

{text}

Provide only the adapted text in {dialect}."""

        chat = generator.chat(
            [
                rg.Message(role="system", content=system_prompt),
                rg.Message(role="user", content=user_prompt),
            ]
        )

        response = await chat.run()
        result_text = response.last.content

        if not isinstance(result_text, str):
            result_text = str(result_text)

        return result_text.strip()

    return Transform(transform, name=name)
