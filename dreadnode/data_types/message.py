from __future__ import annotations

import base64
import copy
import typing as t
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import rigging as rg

from dreadnode.data_types import Audio, Image, Text, Video
from dreadnode.data_types.base import DataType
from dreadnode.serialization import serialize

Role = t.Literal["system", "user", "assistant", "tool"]


@dataclass
class Message(DataType):
    """
    Multimodal message container.
    """

    role: Role
    content: list[Text | Image | Audio | Video | str]
    metadata: dict[str, t.Any] = field(default_factory=dict)
    uuid: UUID = field(default_factory=uuid4)
    tool_calls: list[dict[str, t.Any]] | None = None
    tool_call_id: str | None = None

    @property
    def text_parts(self) -> list[Text | str]:
        return [part for part in self.content if isinstance(part, (Text, str))]

    @property
    def image_parts(self) -> list[Image]:
        return [part for part in self.content if isinstance(part, Image)]

    @property
    def audio_parts(self) -> list[Audio]:
        return [part for part in self.content if isinstance(part, Audio)]

    @property
    def video_parts(self) -> list[Video]:
        return [part for part in self.content if isinstance(part, Video)]

    @property
    def text(self) -> str:
        texts = [str(part) for part in self.text_parts]
        return "\n".join(texts)

    def to_serializable(self) -> tuple[t.Any, dict[str, t.Any]]:
        """Serialize message with explicit type field for each content part."""
        serialized_content = []

        for part in self.content:
            result = serialize(part)

            part_type = result.schema.get("x-python-datatype", "")
            if not part_type and isinstance(part, str):
                part_type = "text"
                result.schema["x-python-datatype"] = "dreadnode.Text"
            elif not part_type and isinstance(part, Text):
                part_type = "text"
                if "x-python-datatype" not in result.schema:
                    result.schema["x-python-datatype"] = "dreadnode.Text"

            serialized_content.append(
                {
                    "type": part_type,
                    "data": result.data,
                    "schema": result.schema,
                }
            )
        schema = {
            "x-python-datatype": "Message",
            "role": self.role,
            "num_parts": len(self.content),
        }

        message_data = {
            "role": self.role,
            "content": serialized_content,
            "uuid": str(self.uuid),
            "metadata": self.metadata,
            "schema": schema,
        }

        if self.tool_calls:
            message_data["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            message_data["tool_call_id"] = self.tool_call_id

        return message_data, schema

    def to_rigging(self) -> rg.Message:
        """Convert to Rigging Message for LLM API calls."""

        rg_content: list[rg.ContentText | rg.ContentImageUrl | rg.ContentAudioInput] = []

        for part in self.content:
            if isinstance(part, Image):
                base64_str = part.to_base64()
                _, meta = part.to_serializable()
                data_url = f"data:image/{meta.get('format', 'png')};base64,{base64_str}"
                rg_content.append(rg.ContentImageUrl.from_url(data_url))

            elif isinstance(part, Audio):
                audio_bytes, metadata = part.to_serializable()
                audio_base64 = base64.b64encode(audio_bytes).decode()

                audio = rg.ContentAudioInput.Audio(
                    data=audio_base64,
                    format=metadata.get("extension", "wav"),
                    transcript=metadata.get("transcript"),
                )
                rg_content.append(rg.ContentAudioInput(input_audio=audio))

            elif isinstance(part, Video):
                # Video not supported - convert to text placeholder
                rg_content.append(rg.ContentText(text="[Video content]"))

            else:
                rg_content.append(rg.ContentText(text=str(part)))

        return rg.Message(
            role=self.role,
            content=rg_content,
            tool_calls=self.tool_calls,
            tool_call_id=self.tool_call_id,
            metadata=self.metadata,
        )

    @classmethod
    def from_rigging(cls, msg: rg.Message) -> Message:
        """Parse Rigging Message back to dn.Message."""

        parts: list[Text | Image | Audio | Video | str] = []

        for part in msg.content_parts:
            if isinstance(part, rg.ContentImageUrl):
                parts.append(Image(data=part.image_url.url))

            elif isinstance(part, rg.ContentAudioInput):
                audio_bytes = base64.b64decode(part.input_audio.data)
                parts.append(Audio(data=audio_bytes, format=part.input_audio.format))

            elif isinstance(part, rg.ContentText):
                parts.append(part.text)

            else:
                parts.append(str(part))

        return cls(
            role=msg.role,
            content=parts,
            metadata=msg.metadata.copy() if msg.metadata else {},
            uuid=msg.uuid,
            tool_calls=[
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
            if msg.tool_calls
            else None,
            tool_call_id=msg.tool_call_id,
        )

    def clone(self) -> Message:
        """
        Create a deep copy of the message.

        Note: For Image/Audio/Video, creates new instances with copied data.
        For strings, they're immutable so can be reused.
        """

        cloned_content: list[Text | Image | Audio | Video | str] = []

        for part in self.content:
            if isinstance(part, Image):
                cloned_content.append(
                    Image(
                        data=part.canonical_array.copy(),
                        mode=part.mode,
                        caption=part._caption,  # noqa: SLF001
                        format=part._format,  # noqa: SLF001
                    )
                )
            elif isinstance(part, (Audio, Video, Text)):
                cloned_content.append(copy.deepcopy(part))
            else:
                cloned_content.append(part)

        return Message(
            role=self.role,
            content=cloned_content,
            metadata=self.metadata.copy(),
            uuid=self.uuid,
            tool_calls=copy.deepcopy(self.tool_calls) if self.tool_calls else None,
            tool_call_id=self.tool_call_id,
        )

    def __str__(self) -> str:
        if len(self.content) == 1 and isinstance(self.content[0], str):
            return f"[{self.role}]: {self.content[0]}"

        parts_summary = []
        for part in self.content:
            if isinstance(part, (str, Text)):
                parts_summary.append("Text")
            elif isinstance(part, Image):
                parts_summary.append("Image")
            elif isinstance(part, Audio):
                parts_summary.append("Audio")
            elif isinstance(part, Video):
                parts_summary.append("Video")
            else:
                parts_summary.append(type(part).__name__)

        return f"[{self.role}]: {len(self.content)} parts ({', '.join(parts_summary)})"

    def __repr__(self) -> str:
        parts = ", ".join(
            type(p).__name__ if not isinstance(p, str) else "str" for p in self.content
        )
        return f"Message(role='{self.role}', content=[{parts}])"
