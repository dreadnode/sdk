import os
import tempfile
import typing as t
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dreadnode.data_types.base import DataType

try:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip  # type: ignore  # noqa: PGH003
    from moviepy.video.VideoClip import VideoClip  # type: ignore  # noqa: PGH003
except ImportError:
    ImageSequenceClip = None
    VideoClip = None


VideoDataType: t.TypeAlias = str | Path | NDArray[t.Any] | bytes | list[NDArray[t.Any]] | t.Any


class Video(DataType):
    """
    Video media type for Dreadnode logging.

    Supports:
    - Local file paths (str or Path)
    - Numpy array sequences with frame rate
    - Raw bytes with metadata
    - MoviePy VideoClip objects (if installed)
    """

    def __init__(
        self,
        data: VideoDataType,
        fps: float | None = None,
        caption: str | None = None,
        format: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        """
        Initialize a Video object.

        Args:
            data: The video data, which can be:
                - A path to a local video file (str or Path)
                - A numpy array of frames (requires fps)
                - A list of numpy arrays for individual frames (requires fps)
                - Raw bytes
                - A MoviePy VideoClip object (if MoviePy is installed)
            fps: Frames per second, required for numpy array input
                 (ignored if data is a file path or raw bytes)
            caption: Optional caption for the video
            format: Optional format override (mp4, avi, etc.)
            width: Optional width in pixels
            height: Optional height in pixels
        """
        self._data = data
        self._fps = fps
        self._caption = caption
        self._format = format or "mp4"
        self._width = width
        self._height = height

    def to_serializable(self) -> tuple[bytes, dict[str, t.Any]]:
        """
        Convert the video to bytes and return with metadata.

        Returns:
            A tuple of (video_bytes, metadata_dict)
        """
        if isinstance(self._data, (str, Path)) and Path(self._data).exists():
            return self._process_file_path()
        if isinstance(self._data, bytes):
            return self._process_bytes()
        if isinstance(self._data, (np.ndarray, list)):
            return self._process_numpy_array()
        if VideoClip is not None and isinstance(self._data, VideoClip):
            return self._process_moviepy_clip()
        if VideoClip is None and hasattr(self._data, "write_videofile"):
            raise ImportError(
                "MoviePy VideoClip detected but moviepy not installed. "
                "Install with: pip install dreadnode[multimodal]"
            )
        raise TypeError(f"Unsupported video data type: {type(self._data)}")

    def _process_file_path(self) -> tuple[bytes, dict[str, t.Any]]:
        """
        Process a video file from a file path.
        Returns:
            A tuple of (video_bytes, metadata_dict)
        """
        if not isinstance(self._data, (str, Path)):
            raise TypeError("Expected file path as str or Path")
        video_bytes = Path(self._data).read_bytes()
        format_name = self._format

        if not format_name or format_name == "mp4":
            ext = Path(self._data).suffix.lstrip(".")
            if ext:
                format_name = ext

        metadata = self._generate_metadata(format_name)
        return video_bytes, metadata

    def _process_bytes(self) -> tuple[bytes, dict[str, t.Any]]:
        """
        Process raw bytes of video data.
        Returns:
            A tuple of (video_bytes, metadata_dict)
        """
        if not isinstance(self._data, bytes):
            raise TypeError("Expected bytes for video data")
        metadata = self._generate_metadata(self._format)
        return self._data, metadata

    def _process_numpy_array(self) -> tuple[bytes, dict[str, t.Any]]:
        """
        Process numpy array frames using MoviePy.
        Returns:
            A tuple of (video_bytes, metadata_dict)
        """
        if ImageSequenceClip is None:
            raise ImportError(
                "Video processing from numpy arrays requires moviepy. "
                "Install with: pip install dreadnode[multimodal]"
            )
        if not self._fps:
            raise ValueError("fps is required for numpy array video frames")
        if not isinstance(self._data, (np.ndarray, list)):
            raise TypeError("data must be a numpy array or list of numpy arrays")

        # Type guard for mypy
        assert ImageSequenceClip is not None  # noqa: S101

        frames = self._extract_frames_from_data()
        if not frames:
            raise ValueError("No frames found in input data")

        return self._create_video_from_frames_data(frames)

    def _extract_frames_from_data(self) -> list[NDArray[t.Any]]:
        """Extract frames from numpy array or list data."""
        frames = []
        rgb_dim = 3
        rgba_dim = 4

        if isinstance(self._data, np.ndarray):
            if self._data.ndim == rgb_dim:  # Single frame
                frames = [self._data]
            elif self._data.ndim == rgba_dim:  # Multiple frames
                frames = [self._data[i] for i in range(self._data.shape[0])]
            else:
                raise ValueError(f"Unsupported numpy array shape: {self._data.ndim}")
        elif isinstance(self._data, list):
            frames = self._data

        return frames

    def _create_video_from_frames_data(
        self, frames: list[NDArray[t.Any]]
    ) -> tuple[bytes, dict[str, t.Any]]:
        """Create video file from frames."""
        frame_height, frame_width = frames[0].shape[:2]
        temp_fd, temp_path = tempfile.mkstemp(suffix=f".{self._format}")
        os.close(temp_fd)

        try:
            # Create clip and write to file
            clip = ImageSequenceClip(frames, fps=self._fps)
            clip.write_videofile(
                temp_path,
                fps=self._fps,
            )
            video_bytes = Path(temp_path).read_bytes()

            metadata = self._generate_metadata(self._format)
            metadata.update(
                {
                    "frame-count": len(frames),
                    "width": self._width or frame_width,
                    "height": self._height or frame_height,
                }
            )
            if isinstance(self._data, np.ndarray):
                metadata["source-type"] = "numpy.ndarray"
                metadata["array-shape"] = str(self._data.shape)
                metadata["array-dtype"] = str(self._data.dtype)
            else:
                metadata["source-type"] = "list[numpy.ndarray]"
                metadata["frames-count"] = len(frames)
            return video_bytes, metadata

        finally:
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    def _process_moviepy_clip(self) -> tuple[bytes, dict[str, t.Any]]:
        """
        Process a MoviePy VideoClip object.
        Returns:
            A tuple of (video_bytes, metadata_dict)
        """
        if not isinstance(self._data, VideoClip):
            raise TypeError("data must be a MoviePy VideoClip object")
        temp_fd, temp_path = tempfile.mkstemp(suffix=f".{self._format}")
        os.close(temp_fd)

        try:
            # Get FPS from clip or provided value
            fps = self._fps or getattr(self._data, "fps", 24)

            # Write to file with compatible parameters
            self._data.write_videofile(
                temp_path,
                fps=fps,
            )

            video_bytes = Path(temp_path).read_bytes()

            metadata = self._generate_metadata(self._format)

            metadata["source-type"] = "moviepy.VideoClip"

            # Add clip metadata if available
            for attr in ["duration", "fps", "size", "rotation", "w", "h", "aspect_ratio"]:
                if hasattr(self._data, attr):
                    value = getattr(self._data, attr)
                    if value is not None:
                        metadata[attr] = value

            return video_bytes, metadata

        finally:
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    def _generate_metadata(self, format_name: str) -> dict[str, t.Any]:
        """
        Generate metadata for the video.
        Args:
            format_name: The format of the video (mp4, avi, etc.)
        Returns:
            A dictionary of metadata
        """
        metadata: dict[str, t.Any] = {
            "extension": format_name.lower(),
            "x-python-datatype": "dreadnode.Video.bytes",
        }

        if self._fps:
            metadata["fps"] = self._fps

        if self._width:
            metadata["width"] = self._width

        if self._height:
            metadata["height"] = self._height

        if self._caption:
            metadata["caption"] = self._caption

        return metadata
