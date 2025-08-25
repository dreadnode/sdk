import tempfile
from pathlib import Path

import aiofiles
from loguru import logger


async def write_tmp_file(
    filename: str, text: str | None = None, raw_bytes: bytes | None = None
) -> str:
    """
    Creates a file, also in a temporary directory, and writes supplied contents.

    Returns: absolute filepath
    """
    if not any([raw_bytes, text]):
        raise TypeError("File contents, as bytes or text must be supplied.")

    tmp_dir = tempfile.TemporaryDirectory(delete=False)
    fullpath = Path(tmp_dir.name) / filename

    if raw_bytes:
        async with aiofiles.open(fullpath, mode="wb") as fh:
            await fh.write(raw_bytes)
    elif text:
        async with aiofiles.open(fullpath, mode="w") as fh:
            await fh.write(text)

    return str(fullpath)


async def delete_local_file(filename: Path) -> None:
    """delete a local file"""
    try:
        fp = Path.resolve(filename)
        Path.unlink(fp)
    except (FileNotFoundError, OSError) as e:
        logger.warning(f"Error trying to delete file {filename}: {e}")


async def delete_local_file_and_dir(filename: Path) -> None:
    """delete a local file and its parent directory"""
    try:
        fp = Path.resolve(filename)
        Path.unlink(fp)
        Path.rmdir(Path.parent(fp))
    except (FileNotFoundError, OSError) as e:
        logger.warning(f"Error trying to delete file and directory {filename}: {e}")
