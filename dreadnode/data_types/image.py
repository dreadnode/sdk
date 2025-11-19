import base64
import io
import typing as t
from pathlib import Path

import numpy as np

from dreadnode.data_types.base import DataType
from dreadnode.util import catch_import_error

if t.TYPE_CHECKING:
    from PIL.Image import Image as PILImage

ImageDataType: t.TypeAlias = "np.ndarray[t.Any, t.Any] | PILImage | t.Any"
ImageDataOrPathType: t.TypeAlias = "str | Path | bytes | ImageDataType"


class Image(DataType):
    """
    Image media type for Dreadnode logging.

    This class maintains a high-fidelity float32 numpy array as the canonical
    representation, ensuring no precision loss during use in transforms, scorers,
    and optimization routines.
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
                - A file path (str or Path)
                - A base64-encoded string (starting with "data:image/")
                - Raw bytes of an image file
                - A numpy array (HWC or HW format)
                - A Pillow Image object
            mode: Optional mode for the image (RGB, L, etc.)
            caption: Optional caption for the image
            format: Optional format to use when saving (png, jpg, etc.)
        """
        with catch_import_error("dreadnode[multimodal]"):
            import PIL.Image  # type: ignore[import-not-found]  # noqa: F401

        self._caption = caption

        self._source_metadata = self._extract_source_metadata(data, format)
        self._format = self._source_metadata.get("format", "png").replace("jpg", "jpeg")
        self._canonical_array, self._mode = self._load_and_convert(data, mode)

        # Caches for conversions
        self._pil_cache: PILImage | None = None
        self._base64_cache: str | None = None

    def _extract_source_metadata(
        self, data: ImageDataOrPathType, format_hint: str | None
    ) -> dict[str, t.Any]:
        """Extract metadata from source without full conversion."""
        metadata = {"source-type": "unknown"}

        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.exists():
                metadata.update(
                    {
                        "source-type": "file",
                        "source-path": str(path),
                        "format": format_hint or path.suffix.lstrip(".").lower() or "png",
                    }
                )
            elif isinstance(data, str) and data.startswith("data:image/"):
                header = data.split(",", 1)[0]
                fmt = header.split("/")[1].split(";")[0] if "/" in header else "png"
                metadata.update({"source-type": "base64", "format": format_hint or fmt})
            else:
                metadata["format"] = format_hint or "png"
        elif hasattr(data, "mode"):  # PIL Image
            pil_format = getattr(data, "format", None)
            detected_format = pil_format.lower() if pil_format else "png"
            metadata.update(
                {
                    "source-type": "PIL.Image",
                    "format": format_hint or detected_format,
                }
            )
        elif isinstance(data, np.ndarray):
            metadata.update(
                {
                    "source-type": "numpy.ndarray",
                    "array-shape": str(data.shape),
                    "array-dtype": str(data.dtype),
                    "format": format_hint or "png",
                }
            )
        elif isinstance(data, bytes):
            metadata.update({"source-type": "bytes", "format": format_hint or "png"})
        else:
            metadata["format"] = format_hint or "png"

        return metadata

    def _load_and_convert(
        self, data: ImageDataOrPathType, mode: str | None
    ) -> tuple["np.ndarray[t.Any, np.dtype[np.float32]]", str]:
        """
        Load and convert any input to canonical float32 array format.

        Returns:
            Tuple of (canonical_array, mode) where canonical_array is always:
            - dtype: float32
            - range: [0.0, 1.0]
            - format: HWC (Height, Width, Channels) or HW for grayscale
        """

        # Handle numpy arrays directly to preserve precision
        if isinstance(data, np.ndarray):
            return self._convert_numpy_direct(data, mode)

        # For other types, go through PIL (with careful handling)
        pil_img = self._to_pil_from_any(data)

        # Determine final mode
        final_mode = mode or pil_img.mode
        if final_mode != pil_img.mode:
            pil_img = pil_img.convert(final_mode)

        # Convert to canonical numpy array
        canonical_array = np.array(pil_img, dtype=np.float32)

        # Smart normalization based on data characteristics
        canonical_array = self._normalize_to_unit_range(canonical_array)

        # Ensure proper shape (remove single-channel dimension for grayscale)
        if canonical_array.ndim == 3 and canonical_array.shape[2] == 1:
            canonical_array = canonical_array[:, :, 0]

        return canonical_array, final_mode

    def _convert_numpy_direct(
        self, array: "np.ndarray[t.Any, t.Any]", mode: str | None
    ) -> tuple["np.ndarray[t.Any, np.dtype[np.float32]]", str]:
        """Convert numpy array directly to canonical format without PIL roundtrip."""
        # Make a copy and ensure valid format
        arr = self._ensure_valid_image_array(array.copy())

        # Infer or use provided mode
        final_mode = mode or self._infer_mode_from_shape(arr.shape)

        # Convert to float32 and normalize
        canonical_array = arr.astype(np.float32)
        canonical_array = self._normalize_to_unit_range(canonical_array)

        # Handle shape for grayscale
        if canonical_array.ndim == 3 and canonical_array.shape[2] == 1 and final_mode == "L":
            canonical_array = canonical_array[:, :, 0]

        return canonical_array, final_mode

    def _normalize_to_unit_range(
        self, array: "np.ndarray[t.Any, t.Any]"
    ) -> "np.ndarray[t.Any, t.Any]":
        """Smart normalization to [0,1] range based on data characteristics."""
        if array.dtype.kind == "f":  # Already float
            # Already in [0,1]
            if array.max() <= 1.0 and array.min() >= 0.0:
                return array

            # Likely [0,255] float
            if array.max() <= 255.0 and array.min() >= 0.0:
                return array / 255.0

            # Unusual range - normalize to [0,1]
            min_val, max_val = array.min(), array.max()
            if max_val > min_val:
                return (array - min_val) / (max_val - min_val)  # type: ignore[no-any-return]
            return array

        # Binary image
        if array.max() <= 1:
            return array.astype(np.float32)

        # Assume [0,255] or similar
        return array.astype(np.float32) / 255.0

    def _to_pil_from_any(self, data: ImageDataOrPathType) -> "PILImage":
        """Convert any supported input type to PIL Image."""
        import PIL.Image

        if isinstance(data, PIL.Image.Image):
            return data.copy()  # Always work with a copy

        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.exists():
                return PIL.Image.open(path)
            if isinstance(data, str) and data.startswith("data:image/"):
                _, encoded = data.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                return PIL.Image.open(io.BytesIO(image_bytes))
            raise FileNotFoundError(f"Image file not found: {path}")

        if isinstance(data, bytes):
            return PIL.Image.open(io.BytesIO(data))

        if isinstance(data, np.ndarray):
            return self._numpy_to_pil(data)

        raise TypeError(f"Unsupported image data type: {type(data)}")

    def _numpy_to_pil(self, array: "np.ndarray[t.Any, t.Any]") -> "PILImage":
        """Convert numpy array to PIL Image with proper handling."""
        import PIL.Image

        # Make a copy to avoid modifying input
        arr = array.copy()

        # Ensure valid array format
        arr = self._ensure_valid_image_array(arr)

        # Handle different data types and ranges
        if arr.dtype.kind == "f":  # floating point
            if arr.max() <= 1.0:
                # Already in [0,1], convert to uint8
                arr = (arr * 255).astype(np.uint8)
            else:
                # Assume [0,255] range, just convert type
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            # Integer type, ensure uint8 range
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Infer mode from shape
        mode = self._infer_mode_from_shape(arr.shape)

        return PIL.Image.fromarray(arr, mode=mode)

    def _ensure_valid_image_array(
        self, array: "np.ndarray[t.Any, t.Any]"
    ) -> "np.ndarray[t.Any, t.Any]":
        """Ensure numpy array is in valid image format (HWC or HW)."""
        if array.ndim == 2:  # Grayscale
            return array

        if array.ndim == 3:
            h, _, c = array.shape
            if c in (1, 3, 4):  # channels-last (HWC)
                return array
            if h in (1, 3, 4):  # channels-first (CHW)
                return np.transpose(array, (1, 2, 0))

        raise ValueError(f"Unsupported array shape: {array.shape}")

    def _infer_mode_from_shape(self, shape: tuple[int, ...]) -> str:
        """Infer PIL mode from array shape."""
        if len(shape) == 2:
            return "L"  # Grayscale
        if len(shape) == 3:
            channels = shape[2]
            if channels == 1:
                return "L"
            if channels == 3:
                return "RGB"
            if channels == 4:
                return "RGBA"

        raise ValueError(f"Cannot infer mode from shape: {shape}")

    @property
    def canonical_array(self) -> "np.ndarray[t.Any, np.dtype[np.float32]]":
        """
        Get the canonical high-fidelity representation.

        Returns:
            float32 numpy array in [0,1] range, HWC format
        """
        return t.cast("np.ndarray[t.Any, np.dtype[np.float32]]", self._canonical_array.copy())  # type: ignore[redundant-cast]

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the canonical array."""
        return self._canonical_array.shape  # type: ignore[no-any-return]

    @property
    def mode(self) -> str:
        """Get the image mode (L, RGB, RGBA, etc.)."""
        return self._mode

    def resize(self, height: int, width: int, *, resample: int | None = None) -> "Image":
        """
        Resize the image to the specified size.

        Args:
            height: The desired height of the image.
            width: The desired width of the image.
            resample: Resampling filter to use (see PIL.Image for options).

        Returns:
            New Image object with resized image
        """
        pil_img = self.to_pil()
        resized_pil = pil_img.resize((width, height), resample=resample)
        return Image(resized_pil, mode=self._mode, caption=self._caption, format=self._format)

    def to_numpy(self, dtype: t.Any = np.float32) -> "np.ndarray[t.Any, t.Any]":
        """
        Returns the image as a NumPy array with specified dtype.

        Args:
            dtype: Target dtype. Common options:
                - np.float32/np.float64: Values in [0.0, 1.0] (recommended)
                - np.uint8: Values in [0, 255]

        Returns:
            NumPy array in HWC format (or HW for grayscale)
        """
        arr = self._canonical_array.copy()

        if np.issubdtype(dtype, np.integer):  # noqa: SIM108
            # Convert to integer range [0, 255]
            arr = (arr * 255.0).astype(dtype)
        else:
            # Keep float range [0, 1]
            arr = arr.astype(dtype)

        return t.cast("np.ndarray[t.Any, t.Any]", arr)

    def to_pil(self) -> "PILImage":
        """Returns the image as a Pillow Image object."""
        if self._pil_cache is None:
            # Convert canonical array to PIL
            arr = (self._canonical_array * 255).astype(np.uint8)

            import PIL.Image

            self._pil_cache = PIL.Image.fromarray(arr, mode=self._mode)

        return self._pil_cache.copy()  # Return copy to prevent mutation

    def to_base64(self) -> str:
        """Returns the image as a base64 encoded string."""
        if self._base64_cache is None:
            buffer = io.BytesIO()
            self.to_pil().save(buffer, format=self._format.upper())
            self._base64_cache = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return self._base64_cache

    def show(self) -> None:
        """Displays the image using the default image viewer."""
        self.to_pil().show()

    def to_serializable(self) -> tuple[bytes, dict[str, t.Any]]:
        """
        Convert the image to bytes and return with metadata.

        Returns:
            Tuple of (image_bytes, metadata_dict)
        """
        buffer = io.BytesIO()
        pil_img = self.to_pil()
        pil_img.save(buffer, format=self._format.upper())
        image_bytes = buffer.getvalue()

        # Rich metadata including source information
        metadata = {
            "extension": self._format.lower(),
            "x-python-datatype": "dreadnode.Image.bytes",
            "mode": self.mode,
            "width": self.shape[1] if len(self.shape) >= 2 else self.shape[0],
            "height": self.shape[0],
        }

        # Add source metadata
        metadata.update(self._source_metadata)

        if len(self.shape) == 3:
            metadata["channels"] = self.shape[2]
        else:
            metadata["channels"] = 1

        if self._caption:
            metadata["caption"] = self._caption

        return image_bytes, metadata

    def __repr__(self) -> str:
        shape_str = "x".join(map(str, self._canonical_array.shape))
        return f"Image({shape_str}, {self._mode}, {self._canonical_array.dtype})"
