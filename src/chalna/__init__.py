"""
Chalna (찰나) - SRT subtitle generation service.

ElevenLabs Scribe v2 transcription with optional LLM segmentation/refinement.
"""

__version__ = "0.1.0"

from chalna.exceptions import (
    AudioTooLongError,
    ChalnaError,
    CorruptedFileError,
    DiskSpaceError,
    ElevenLabsAPIError,
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
    VibevoiceAPIError,
)
from chalna.models import LlmSegmentationOptions, ScribeOptions, Segment, TranscriptionResult
from chalna.pipeline import ChalnaPipeline

__all__ = [
    # Pipeline
    "ChalnaPipeline",
    # Models
    "Segment",
    "ScribeOptions",
    "LlmSegmentationOptions",
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
    "VibevoiceAPIError",
    "ElevenLabsAPIError",
    "DiskSpaceError",
    "TempFileError",
    "FFmpegNotFoundError",
    # Version
    "__version__",
]
