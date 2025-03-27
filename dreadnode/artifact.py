import mimetypes
import typing as t
from dataclasses import dataclass

from dreadnode.object import ObjectRef
from dreadnode.types import AnyDict

ArtifactHint = t.Literal[
    "markdown",
    "table",
    "code",
    "image",
    "csv",
    "dataframe",
    "notebook",
    "file",
]


@dataclass
class Artifact(ObjectRef):
    hint: ArtifactHint
    description: str
    attributes: AnyDict
    mime_type: str
    extension: str


def get_mime_type_and_extension(data: t.Any) -> tuple[str, str]:
    """Get the file extension and MIME type for the given data.

    This method determines the file extension and MIME type for the given data based on its content.

    Args:
        data: The data to determine the file extension and MIME type for.

    Returns:
        The file extension and MIME type for the given data.
    """
    mime_type: str
    extension: str

    if isinstance(data, str):
        mime_type = "text/plain"
        extension = ".txt"

    elif isinstance(data, dict):
        mime_type = "application/json"
        extension = ".json"

    # Handle binary data
    elif isinstance(data, bytes):
        # Use python-magic to detect file type
        mime = magic.Magic(mime=True)
        mime_type = mime.from_buffer(data)

        # Get extension from MIME type
        extension = mimetypes.guess_extension(mime_type) or ""
        if not extension and mime_type == "image/jpeg":
            extension = ".jpg"  # Common case

    # Handle other types
    else:
        mime_type = "application/octet-stream"
        extension = ".bin"

    # get the original
    return ArtifactMetadata(mime_type=mime_type, extension=extension)
