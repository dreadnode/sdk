import base64
import io
import typing as t
from pathlib import Path

import numpy as np

from dreadnode.data_types.base import DataType

try:
    from PIL import Image as PILImage  # type: ignore[import-not-found,unused-ignore]
except ImportError:
    PILImage = None  # type: ignore[assignment,unused-ignore]

ImageDataType = t.Any | np.ndarray[t.Any, t.Any]
ImageDataOrPathType = str | Path | bytes | ImageDataType


class Image(DataType):
    """
    Image media type for Dreadnode logging.

    Supports:
    - Local file paths (str or Path)
    - PIL Image objects
    - Numpy arrays
    - Base64 encoded strings
    """

    def __init__(
        self,
        data: ImageDataOrPathType,
        mode: str | None = None,
        caption: str | None = None,
        format: str | None = None,
    ):
        """
        Initialize an Image object.

        Args:
            data: The image data, which can be:
                - A path to a local image file (str or Path)
                - A PIL Image object
                - A numpy array
                - Base64 encoded string
                - Raw bytes
            mode: Optional mode for the image (RGB, L, etc.)
            caption: Optional caption for the image
            format: Optional format to use when saving (png, jpg, etc.)
        """
        if PILImage is None:
            raise ImportError(
                "Image processing requires PIL (Pillow). Install with: pip install dreadnode[multimodal]"
            )
        self._data = data
        self._mode = mode
        self._caption = caption
        self._format = format

    def to_serializable(self) -> tuple[t.Any, dict[str, t.Any]]:
        """
        Convert the image to bytes and return with metadata.
        Returns:
            A tuple of (image_bytes, metadata_dict)
        """
        image_bytes, image_format, mode, width, height = self._process_image_data()
        metadata = self._generate_metadata(image_format, mode, width, height)
        return image_bytes, metadata

    def _process_image_data(self) -> tuple[bytes, str, str | None, int | None, int | None]:
        """
        Process the image data and return bytes, format, mode, width, and height.
        Returns:
            A tuple of (image_bytes, image_format, mode, width, height)
        """
        if isinstance(self._data, (str, Path)) and Path(self._data).exists():
            return self._process_file_path()
        if isinstance(self._data, PILImage.Image):
            return self._process_pil_image()
        if isinstance(self._data, np.ndarray):
            return self._process_numpy_array()
        if isinstance(self._data, bytes):
            return self._process_raw_bytes()
        if isinstance(self._data, str) and self._data.startswith("data:image/"):
            return self._process_base64_string()
        raise TypeError(f"Unsupported image data type: {type(self._data)}")

    def _process_file_path(self) -> tuple[bytes, str, str | None, int | None, int | None]:
        """
        Process image from file path.
        Returns:
            A tuple of (image_bytes, image_format, mode, width, height)
        """
        path_str = str(self._data)
        image_bytes = Path(path_str).read_bytes()
        image_format = self._format or Path(path_str).suffix.lstrip(".") or "png"
        mode, width, height = self._mode, None, None
        with PILImage.open(path_str) as img:
            width, height = img.size
            detected_mode = img.mode
            mode = mode or detected_mode
        return image_bytes, image_format, mode, width, height

    def _process_pil_image(self) -> tuple[bytes, str, str | None, int | None, int | None]:
        """
        Process PIL Image object.
        Returns:
            A tuple of (image_bytes, image_format, mode, width, height)
        """
        if not isinstance(self._data, PILImage.Image):
            raise TypeError(f"Expected PILImage.Image, got {type(self._data)}")

        pil_image = self._data
        mode = self._mode or pil_image.mode
        image_format = self._format or (pil_image.format.lower() if pil_image.format else "png")

        buffer = io.BytesIO()
        img_to_save = pil_image

        if mode and pil_image.mode != mode:
            if mode == "RGBA" and pil_image.mode in ("RGB", "L"):
                # For RGB to RGBA, add an alpha channel
                # Convert to RGBA first
                img_to_save = pil_image.convert("RGBA")
            else:
                # Standard conversion
                img_to_save = pil_image.convert(mode)

        # Make sure format supports alpha if using RGBA mode
        if mode == "RGBA" and image_format.lower() in ("jpg", "jpeg"):
            # JPEG doesn't support transparency, switch to PNG
            image_format = "png"

        # Save image to buffer
        img_to_save.save(buffer, format=image_format)
        image_bytes = buffer.getvalue()
        width, height = pil_image.size
        return image_bytes, image_format, mode, width, height

    def _process_numpy_array(self) -> tuple[bytes, str, str | None, int | None, int | None]:
        """
        Process numpy array to bytes.
        Returns:
            A tuple of (image_bytes, image_format, mode, width, height)
        """
        buffer = io.BytesIO()
        image_format = self._format or "png"

        mode = self._mode or (
            self._guess_mode(self._data) if isinstance(self._data, np.ndarray) else None
        )
        if not isinstance(self._data, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(self._data)}")
        valid_array = self._ensure_valid_image_array(self._data)

        # Explicitly handle float arrays with values in [0, 1]
        if valid_array.dtype.kind == "f" and valid_array.max() <= 1.0:
            valid_array = (valid_array * 255).astype(np.uint8)
        elif valid_array.dtype != np.uint8:
            valid_array = np.clip(valid_array, 0, 255).astype(np.uint8)

        img = PILImage.fromarray(valid_array, mode=mode)
        img.save(buffer, format=image_format)
        image_bytes = buffer.getvalue()
        width, height = img.size
        return image_bytes, image_format, mode, width, height

    def _process_raw_bytes(self) -> tuple[bytes, str, str | None, int | None, int | None]:
        """
        Process raw bytes.
        Returns:
            A tuple of (image_bytes, image_format, mode, width, height)
        """
        if not isinstance(self._data, bytes):
            raise TypeError(f"Expected bytes, got {type(self._data)}")
        image_bytes = self._data
        image_format = self._format or "png"
        mode, width, height = self._mode, None, None
        with PILImage.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            detected_mode = img.mode
            mode = mode or detected_mode

            if mode and img.mode != mode:
                buffer = io.BytesIO()
                img.convert(mode).save(buffer, format=image_format)
                image_bytes = buffer.getvalue()

        return image_bytes, image_format, mode, width, height

    def _process_base64_string(self) -> tuple[bytes, str, str | None, int | None, int | None]:
        """
        Process base64 encoded string.
        Returns:
            A tuple of (image_bytes, image_format, mode, width, height)
        """
        if not isinstance(self._data, str):
            raise TypeError(f"Expected str, got {type(self._data)}")

        # Handle data URL format (data:image/png;base64,...)
        if "," in self._data:
            header, encoded = self._data.split(",", 1)
            format_part = header.split("/")[1].split(";")[0] if "/" in header else "png"
        else:
            encoded = self._data
            format_part = "png"  # Default for raw base64

        image_format = self._format or format_part

        # Decode the base64 string
        # TODO(@raja): See if we could optimize this  # noqa: TD003
        image_bytes = base64.b64decode(encoded)

        # Open with PIL to get properties
        with PILImage.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            detected_mode = img.mode
            mode = self._mode or detected_mode

            # Convert mode if needed
            if mode and img.mode != mode:
                buffer = io.BytesIO()
                img.convert(mode).save(buffer, format=image_format)
                image_bytes = buffer.getvalue()

        return image_bytes, image_format, mode, width, height

    def _generate_metadata(
        self, image_format: str, mode: str | None, width: int | None, height: int | None
    ) -> dict[str, str | int | None]:
        """Generate metadata for the image."""
        metadata: dict[str, str | int | None] = {
            "extension": image_format.lower(),
            "x-python-datatype": "dreadnode.Image.bytes",
        }

        if isinstance(self._data, (str, Path)) and Path(self._data).exists():
            metadata["source-type"] = "file"
            metadata["source-path"] = str(self._data)
        elif isinstance(self._data, PILImage.Image):
            metadata["source-type"] = "PIL.Image"
        elif isinstance(self._data, np.ndarray):
            metadata["source-type"] = "numpy.ndarray"
            metadata["array-shape"] = str(self._data.shape)
            metadata["array-dtype"] = str(self._data.dtype)
        elif isinstance(self._data, bytes):
            metadata["source-type"] = "bytes"
        elif isinstance(self._data, str) and self._data.startswith("data:image/"):
            metadata["source-type"] = "base64"

        if mode:
            metadata["mode"] = mode

        if width is not None and height is not None:
            metadata["width"] = width
            metadata["height"] = height

        if self._caption:
            metadata["caption"] = self._caption

        return metadata

    def _guess_mode(self, data: np.ndarray[t.Any, np.dtype[t.Any]]) -> str:
        """Guess what type of image the np.array is representing."""
        ndims = data.ndim
        grayscale_dim = 2
        rgb_dim = 3
        if ndims == grayscale_dim:
            return "L"

        if ndims == rgb_dim:
            # Map shape to mode for channels-last (HWC) and channels-first (CHW)
            shape_to_mode = {
                (1,): "L",
                (3,): "RGB",
                (4,): "RGBA",
            }
            if data.shape[2:] in shape_to_mode:
                return shape_to_mode[data.shape[2:]]
            if data.shape[:1] in shape_to_mode:
                return shape_to_mode[data.shape[:1]]

        raise ValueError(f"Unsupported array shape for image: {data.shape}")

    def _ensure_valid_image_array(
        self, array: np.ndarray[t.Any, np.dtype[t.Any]]
    ) -> np.ndarray[t.Any, np.dtype[t.Any]]:
        """Convert numpy array to a format suitable for PIL."""
        grayscale_dim = 2
        rgb_dim = 3
        # Handle grayscale (2D arrays)
        if array.ndim == grayscale_dim:
            return array

        # Handle standard 3D arrays
        if array.ndim == rgb_dim:
            # Channels-last format (HWC) - standard for PIL
            if array.shape[2] in (1, 3, 4):
                return array

            # Channels-first format (CHW) - convert to channels-last
            if array.shape[0] in (1, 3, 4):
                return np.transpose(array, (1, 2, 0))

        raise ValueError(f"Unsupported numpy array shape: {array.shape}")
