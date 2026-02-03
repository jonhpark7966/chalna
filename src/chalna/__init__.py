"""
Chalna (찰나) - SRT subtitle generation service.

VibeVoice ASR + Qwen Forced Alignment for accurate timestamps and speaker diarization.
"""

__version__ = "0.1.0"

from chalna.exceptions import (
    AudioTooLongError,
    ChalnaError,
    CorruptedFileError,
    DiskSpaceError,
    EmptyTranscriptionError,
    ErrorCode,
    FFmpegNotFoundError,
    FilePermissionError,
    FileTooLargeError,
    ModelDownloadError,
    ModelLoadError,
    OutOfMemoryError,
    TempFileError,
    UnsupportedFormatError,
)
from chalna.models import Segment, TranscriptionResult
from chalna.pipeline import ChalnaPipeline

__all__ = [
    # Pipeline
    "ChalnaPipeline",
    # Models
    "Segment",
    "TranscriptionResult",
    # Exceptions
    "ChalnaError",
    "ErrorCode",
    "AudioTooLongError",
    "FileTooLargeError",
    "UnsupportedFormatError",
    "CorruptedFileError",
    "FilePermissionError",
    "OutOfMemoryError",
    "EmptyTranscriptionError",
    "ModelLoadError",
    "ModelDownloadError",
    "DiskSpaceError",
    "TempFileError",
    "FFmpegNotFoundError",
    # Version
    "__version__",
]
