import io
import typing as t
from pathlib import Path

import numpy as np

try:
    import soundfile as sf  # type: ignore  # noqa: PGH003
except ImportError:
    sf = None

from dreadnode.data_types.base import DataType

AudioDataType: t.TypeAlias = str | Path | np.ndarray[t.Any, t.Any] | bytes


class Audio(DataType):
    """
    Audio media type for Dreadnode logging.

    Supports:
    - Local file paths (str or Path)
    - Numpy arrays with sample rate
    - Raw bytes
    """

    def __init__(
        self,
        data: AudioDataType,
        sample_rate: int | None = None,
        caption: str | None = None,
        format: str | None = None,
    ):
        """
        Initialize an Audio object.

        Args:
            data: The audio data, which can be:
                - A path to a local audio file (str or Path)
                - A numpy array (requires sample_rate)
                - Raw bytes
            sample_rate: Required when using numpy arrays
            caption: Optional caption for the audio
            format: Optional format to use (default is wav for numpy arrays)
        """
        if sf is None:
            raise ImportError(
                "Audio processing requires optional dependencies. "
                "Install with: pip install dreadnode[multimodal]"
            )
        self._data = data
        self._sample_rate = sample_rate
        self._caption = caption
        self._format = format

    def to_serializable(self) -> tuple[t.Any, dict[str, t.Any]]:
        """
        Serialize the audio data to bytes and return with metadata.
        Returns:
            A tuple of (audio_bytes, metadata_dict)
        """
        audio_bytes, format_name, sample_rate, duration = self._process_audio_data()
        metadata = self._generate_metadata(format_name, sample_rate, duration)
        return audio_bytes, metadata

    def _process_audio_data(self) -> tuple[bytes, str, int | None, float | None]:
        """
        Process the audio data and return bytes, format, sample rate, and duration.
        Returns:
            A tuple of (audio_bytes, format_name, sample_rate, duration)
        """
        if isinstance(self._data, str | Path) and Path(self._data).exists():
            return self._process_file_path()
        if isinstance(self._data, np.ndarray):
            return self._process_numpy_array()
        if isinstance(self._data, bytes):
            return self._process_raw_bytes()
        raise TypeError(f"Unsupported audio data type: {type(self._data)}")

    def _process_file_path(self) -> tuple[bytes, str, int | None, float | None]:
        """
        Process audio from file path. Obtain sample rate and duration using soundfile.
        Returns:
            A tuple of (audio_bytes, format_name, sample_rate, duration)
        """
        path_str = str(self._data)
        audio_bytes = Path(path_str).read_bytes()
        format_name = self._format or Path(path_str).suffix.lstrip(".").lower() or "wav"
        sample_rate = self._sample_rate
        duration = None
        with sf.SoundFile(path_str) as f:
            sample_rate = sample_rate or f.samplerate
            duration = f.frames / f.samplerate

        return audio_bytes, format_name, sample_rate, duration

    def _process_numpy_array(self) -> tuple[bytes, str, int | None, float | None]:
        """
        Process numpy array to WAV using soundfile.
        Returns:
            A tuple of (audio_bytes, format_name, sample_rate, duration)
        """
        if self._sample_rate is None:
            raise ValueError('Argument "sample_rate" is required when using numpy arrays.')

        buffer = io.BytesIO()
        format_name = self._format or "wav"
        sf.write(buffer, self._data, self._sample_rate, format=format_name)
        buffer.seek(0)
        audio_bytes = buffer.read()

        if isinstance(self._data, np.ndarray):
            duration = len(self._data) / float(self._sample_rate)
        else:
            raise TypeError("Invalid data type for numpy array processing.")

        return audio_bytes, format_name, self._sample_rate, duration

    def _process_raw_bytes(self) -> tuple[bytes, str, int | None, float | None]:
        """
        Process raw bytes. Format is determined by the provided format argument.
        Returns:
            A tuple of (audio_bytes, format_name, sample_rate, duration)
        """
        format_name = self._format or "wav"
        if not isinstance(self._data, bytes):
            raise TypeError("Raw bytes are expected for this processing method.")
        return self._data, format_name, self._sample_rate, None

    def _generate_metadata(
        self, format_name: str, sample_rate: int | None, duration: float | None
    ) -> dict[str, str | int | float | None]:
        """
        Generate metadata for the audio data.
        Returns:
            A dictionary of metadata
        """
        metadata: dict[str, str | int | float | None] = {
            "extension": format_name.lower(),
            "x-python-datatype": "dreadnode.Audio.bytes",
        }

        if isinstance(self._data, str | Path):
            metadata["source-type"] = "file"
            metadata["source-path"] = str(self._data)
        elif isinstance(self._data, np.ndarray):
            metadata["source-type"] = "numpy.ndarray"
        elif isinstance(self._data, bytes):
            metadata["source-type"] = "bytes"

        if sample_rate is not None:
            metadata["sample-rate"] = sample_rate

        if duration is not None:
            metadata["duration"] = duration

        if self._caption:
            metadata["caption"] = self._caption

        return metadata
