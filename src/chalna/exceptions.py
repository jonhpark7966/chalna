"""
Chalna Exception Classes.

Structured error handling with error codes for API responses.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional


class ErrorCode(str, Enum):
    """Error codes for Chalna exceptions."""

    # Validation (1xxx)
    AUDIO_TOO_LONG = "E1001"
    UNSUPPORTED_FORMAT = "E1002"
    CORRUPTED_FILE = "E1003"
    FILE_TOO_LARGE = "E1004"
    PERMISSION_DENIED = "E1005"

    # Runtime (2xxx)
    OUT_OF_MEMORY = "E2001"
    EMPTY_TRANSCRIPTION = "E2002"
    MODEL_LOAD_FAILED = "E2004"

    # Network (3xxx)
    MODEL_DOWNLOAD_FAILED = "E3001"
    CODEX_API_ERROR = "E3002"
    CODEX_RATE_LIMIT = "E3003"
    VIBEVOICE_API_ERROR = "E3004"

    # System (4xxx)
    DISK_SPACE_ERROR = "E4001"
    TEMP_FILE_ERROR = "E4002"
    FFMPEG_NOT_FOUND = "E4003"


class ChalnaError(Exception):
    """Base exception class for Chalna.

    All Chalna exceptions inherit from this class and provide:
    - error_code: Unique error code for programmatic handling
    - http_status: HTTP status code for API responses
    - message: Human-readable error message
    - details: Additional context as a dictionary
    """

    error_code: ErrorCode = ErrorCode.MODEL_LOAD_FAILED  # Default
    http_status: int = 500

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        self.message = message
        self.details = details or {}
        self.cause = cause
        if cause:
            self.details["cause"] = str(cause)
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to API response dictionary."""
        return {
            "error": True,
            "error_code": self.error_code.value,
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# Validation Errors (1xxx) - HTTP 400
# =============================================================================


class AudioTooLongError(ChalnaError):
    """Raised when audio duration exceeds the maximum allowed (1 hour)."""

    error_code = ErrorCode.AUDIO_TOO_LONG
    http_status = 400

    def __init__(
        self,
        duration_seconds: float,
        max_duration_seconds: float = 3600,
        cause: Optional[Exception] = None,
    ):
        message = (
            f"Audio duration ({duration_seconds:.1f}s) exceeds maximum "
            f"allowed ({max_duration_seconds:.0f}s / 1 hour)"
        )
        details = {
            "duration_seconds": duration_seconds,
            "max_duration_seconds": max_duration_seconds,
        }
        super().__init__(message, details, cause)


class UnsupportedFormatError(ChalnaError):
    """Raised when audio/video format is not supported."""

    error_code = ErrorCode.UNSUPPORTED_FORMAT
    http_status = 400

    def __init__(
        self,
        format_name: str,
        supported_formats: Optional[list[str]] = None,
        cause: Optional[Exception] = None,
    ):
        supported = supported_formats or []
        message = f"Unsupported audio/video format: {format_name}"
        if supported:
            message += f". Supported formats: {', '.join(supported)}"
        details = {
            "format": format_name,
            "supported_formats": supported,
        }
        super().__init__(message, details, cause)


class CorruptedFileError(ChalnaError):
    """Raised when the audio/video file is corrupted or unreadable."""

    error_code = ErrorCode.CORRUPTED_FILE
    http_status = 400

    def __init__(
        self,
        file_path: str,
        reason: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        message = f"Corrupted or unreadable file: {file_path}"
        if reason:
            message += f" ({reason})"
        details = {
            "file_path": file_path,
            "reason": reason,
        }
        super().__init__(message, details, cause)


class FilePermissionError(ChalnaError):
    """Raised when file cannot be read due to permission issues."""

    error_code = ErrorCode.PERMISSION_DENIED
    http_status = 400

    def __init__(
        self,
        file_path: str,
        cause: Optional[Exception] = None,
    ):
        message = f"Permission denied: cannot read file {file_path}"
        details = {
            "file_path": file_path,
        }
        super().__init__(message, details, cause)


class FileTooLargeError(ChalnaError):
    """Raised when file size exceeds the maximum allowed (2GB)."""

    error_code = ErrorCode.FILE_TOO_LARGE
    http_status = 400

    def __init__(
        self,
        file_size_mb: float,
        max_size_mb: float = 2048,
        cause: Optional[Exception] = None,
    ):
        message = (
            f"File size ({file_size_mb:.1f}MB) exceeds maximum "
            f"allowed ({max_size_mb:.0f}MB / {max_size_mb / 1024:.1f}GB)"
        )
        details = {
            "file_size_mb": file_size_mb,
            "max_size_mb": max_size_mb,
        }
        super().__init__(message, details, cause)


# =============================================================================
# Runtime Errors (2xxx) - HTTP 500/503
# =============================================================================


class OutOfMemoryError(ChalnaError):
    """Raised when GPU or CPU memory is exhausted."""

    error_code = ErrorCode.OUT_OF_MEMORY
    http_status = 503

    def __init__(
        self,
        memory_type: str = "GPU",
        cause: Optional[Exception] = None,
    ):
        message = f"{memory_type} memory exhausted during processing"
        details = {
            "memory_type": memory_type,
        }
        super().__init__(message, details, cause)


class EmptyTranscriptionError(ChalnaError):
    """Raised when no speech is detected in the audio.

    Note: This is not necessarily an error condition. The audio may
    genuinely contain no speech. HTTP 200 is used to indicate success
    with empty content.
    """

    error_code = ErrorCode.EMPTY_TRANSCRIPTION
    http_status = 200

    def __init__(
        self,
        audio_duration: Optional[float] = None,
        cause: Optional[Exception] = None,
    ):
        message = "No speech detected in audio"
        details = {}
        if audio_duration is not None:
            details["audio_duration"] = audio_duration
        super().__init__(message, details, cause)


class ModelLoadError(ChalnaError):
    """Raised when a model fails to load."""

    error_code = ErrorCode.MODEL_LOAD_FAILED
    http_status = 500

    def __init__(
        self,
        model_name: str,
        reason: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        message = f"Failed to load model: {model_name}"
        if reason:
            message += f" ({reason})"
        details = {
            "model_name": model_name,
            "reason": reason,
        }
        super().__init__(message, details, cause)


# =============================================================================
# Network Errors (3xxx) - HTTP 503
# =============================================================================


class ModelDownloadError(ChalnaError):
    """Raised when model download fails."""

    error_code = ErrorCode.MODEL_DOWNLOAD_FAILED
    http_status = 503

    def __init__(
        self,
        model_name: str,
        cause: Optional[Exception] = None,
    ):
        message = f"Failed to download model: {model_name}"
        details = {
            "model_name": model_name,
        }
        super().__init__(message, details, cause)


class CodexAPIError(ChalnaError):
    """Raised when Codex CLI API call fails."""

    error_code = ErrorCode.CODEX_API_ERROR
    http_status = 503

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
    ):
        details = {
            "reason": message,
        }
        super().__init__(message, details, cause)


class CodexRateLimitError(ChalnaError):
    """Raised when Codex API rate limit or quota is exceeded."""

    error_code = ErrorCode.CODEX_RATE_LIMIT
    http_status = 429

    def __init__(
        self,
        message: str,
    ):
        full_message = f"Codex API 사용량 초과: {message}"
        details = {
            "reason": message,
        }
        super().__init__(full_message, details)


class VibevoiceAPIError(ChalnaError):
    """Raised when VibeVoice API call fails (connection error, bad response, etc.)."""

    error_code = ErrorCode.VIBEVOICE_API_ERROR
    http_status = 503

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
    ):
        details = {
            "reason": message,
        }
        super().__init__(message, details, cause)


# =============================================================================
# System Errors (4xxx) - HTTP 500/503
# =============================================================================


class DiskSpaceError(ChalnaError):
    """Raised when there is insufficient disk space."""

    error_code = ErrorCode.DISK_SPACE_ERROR
    http_status = 503

    def __init__(
        self,
        required_mb: float,
        available_mb: float,
        path: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        message = (
            f"Insufficient disk space: {available_mb:.1f}MB available, "
            f"{required_mb:.1f}MB required"
        )
        details = {
            "required_mb": required_mb,
            "available_mb": available_mb,
            "path": path,
        }
        super().__init__(message, details, cause)


class TempFileError(ChalnaError):
    """Raised when temporary file operations fail."""

    error_code = ErrorCode.TEMP_FILE_ERROR
    http_status = 500

    def __init__(
        self,
        operation: str,
        cause: Optional[Exception] = None,
    ):
        message = f"Temporary file operation failed: {operation}"
        details = {
            "operation": operation,
        }
        super().__init__(message, details, cause)


class FFmpegNotFoundError(ChalnaError):
    """Raised when ffmpeg/ffprobe is not installed."""

    error_code = ErrorCode.FFMPEG_NOT_FOUND
    http_status = 500

    def __init__(
        self,
        tool: str = "ffmpeg",
        cause: Optional[Exception] = None,
    ):
        message = f"{tool} is not installed or not found in PATH"
        details = {
            "tool": tool,
        }
        super().__init__(message, details, cause)
