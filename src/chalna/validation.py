"""
Chalna Audio Validation.

ffprobe-based audio file validation with duration, format, and integrity checks.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from chalna.exceptions import (
    AudioTooLongError,
    CorruptedFileError,
    DiskSpaceError,
    FFmpegNotFoundError,
    FilePermissionError,
    FileTooLargeError,
    UnsupportedFormatError,
)


# Maximum audio duration in seconds (1 hour)
MAX_DURATION_SECONDS = 3600

# Maximum file size in MB (2GB)
MAX_FILE_SIZE_MB = 2048

# Supported audio formats
SUPPORTED_AUDIO_FORMATS = {
    "mp3", "wav", "flac", "aac", "ogg", "opus", "m4a", "wma",
}

# Supported video formats (audio will be extracted)
SUPPORTED_VIDEO_FORMATS = {
    "mp4", "mov", "webm", "mkv", "avi",
}

# All supported formats
SUPPORTED_FORMATS = SUPPORTED_AUDIO_FORMATS | SUPPORTED_VIDEO_FORMATS


@dataclass
class AudioInfo:
    """Audio file information from ffprobe."""

    duration_seconds: float
    format_name: str
    codec_name: str
    sample_rate: int
    channels: int
    bit_rate: Optional[int] = None
    file_size_bytes: Optional[int] = None

    @property
    def is_audio_only(self) -> bool:
        """Check if file is audio-only (no video stream)."""
        return self.format_name.lower() in SUPPORTED_AUDIO_FORMATS


def _check_ffprobe_available() -> None:
    """Check if ffprobe is available in PATH.

    Raises:
        FFmpegNotFoundError: If ffprobe is not found.
    """
    if shutil.which("ffprobe") is None:
        raise FFmpegNotFoundError(tool="ffprobe")


def get_audio_info(file_path: Path) -> AudioInfo:
    """Extract audio information using ffprobe.

    Args:
        file_path: Path to the audio/video file.

    Returns:
        AudioInfo with duration, format, codec, etc.

    Raises:
        FFmpegNotFoundError: If ffprobe is not installed.
        FilePermissionError: If file cannot be read.
        CorruptedFileError: If file is corrupted or invalid.
    """
    _check_ffprobe_available()

    file_path = Path(file_path)

    # Check file exists and is readable
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise CorruptedFileError(str(file_path), reason="Not a regular file")

    # Check read permission
    try:
        with open(file_path, "rb") as f:
            # Just try to read first byte
            f.read(1)
    except PermissionError as e:
        raise FilePermissionError(str(file_path), cause=e)

    # Run ffprobe
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(file_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as e:
        raise CorruptedFileError(
            str(file_path),
            reason="ffprobe timed out (file may be too large or corrupted)",
            cause=e,
        )
    except Exception as e:
        raise CorruptedFileError(
            str(file_path),
            reason=f"ffprobe failed: {e}",
            cause=e,
        )

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "Unknown error"
        raise CorruptedFileError(
            str(file_path),
            reason=f"ffprobe error: {stderr}",
        )

    # Parse JSON output
    try:
        probe_data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise CorruptedFileError(
            str(file_path),
            reason="ffprobe returned invalid JSON",
            cause=e,
        )

    # Extract format info
    format_info = probe_data.get("format", {})
    if not format_info:
        raise CorruptedFileError(
            str(file_path),
            reason="No format information found",
        )

    # Find audio stream
    streams = probe_data.get("streams", [])
    audio_stream = None
    for stream in streams:
        if stream.get("codec_type") == "audio":
            audio_stream = stream
            break

    if audio_stream is None:
        raise CorruptedFileError(
            str(file_path),
            reason="No audio stream found in file",
        )

    # Extract duration
    duration = None
    if "duration" in format_info:
        try:
            duration = float(format_info["duration"])
        except (ValueError, TypeError):
            pass

    if duration is None and "duration" in audio_stream:
        try:
            duration = float(audio_stream["duration"])
        except (ValueError, TypeError):
            pass

    if duration is None:
        raise CorruptedFileError(
            str(file_path),
            reason="Could not determine audio duration",
        )

    # Get format name from extension or format_name
    format_name = format_info.get("format_name", "").split(",")[0].lower()
    if not format_name:
        format_name = file_path.suffix.lstrip(".").lower()

    # Map common format names
    format_mapping = {
        "matroska": "mkv",
        "mov,mp4,m4a,3gp,3g2,mj2": "mp4",
        "mpegts": "ts",
    }
    format_name = format_mapping.get(format_name, format_name)

    return AudioInfo(
        duration_seconds=duration,
        format_name=format_name,
        codec_name=audio_stream.get("codec_name", "unknown"),
        sample_rate=int(audio_stream.get("sample_rate", 0)),
        channels=int(audio_stream.get("channels", 0)),
        bit_rate=int(format_info.get("bit_rate", 0)) if format_info.get("bit_rate") else None,
        file_size_bytes=int(format_info.get("size", 0)) if format_info.get("size") else None,
    )


def validate_audio_file(
    file_path: Path,
    max_duration: float = MAX_DURATION_SECONDS,
    max_file_size_mb: float = MAX_FILE_SIZE_MB,
) -> AudioInfo:
    """Validate audio file for processing.

    Performs comprehensive validation:
    1. Check file exists and is readable
    2. Check file size is within limits
    3. Check format is supported
    4. Check duration is within limits
    5. Verify file is not corrupted

    Args:
        file_path: Path to the audio/video file.
        max_duration: Maximum allowed duration in seconds (default: 1 hour).
        max_file_size_mb: Maximum allowed file size in MB (default: 2GB).

    Returns:
        AudioInfo with validated file information.

    Raises:
        AudioTooLongError: If audio exceeds max_duration.
        FileTooLargeError: If file exceeds max_file_size_mb.
        UnsupportedFormatError: If format is not supported.
        CorruptedFileError: If file is corrupted.
        FilePermissionError: If file cannot be read.
        FFmpegNotFoundError: If ffprobe is not installed.
    """
    file_path = Path(file_path)

    # Check file size early (fast rejection before ffprobe)
    if file_path.exists():
        file_size_bytes = file_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        if file_size_mb > max_file_size_mb:
            raise FileTooLargeError(
                file_size_mb=file_size_mb,
                max_size_mb=max_file_size_mb,
            )

    # Get file extension
    extension = file_path.suffix.lstrip(".").lower()

    # Check if format is potentially supported before probing
    if extension and extension not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            format_name=extension,
            supported_formats=sorted(SUPPORTED_FORMATS),
        )

    # Get audio info (this also validates file integrity)
    audio_info = get_audio_info(file_path)

    # Check duration
    if audio_info.duration_seconds > max_duration:
        raise AudioTooLongError(
            duration_seconds=audio_info.duration_seconds,
            max_duration_seconds=max_duration,
        )

    return audio_info


def check_disk_space(
    required_mb: float,
    path: Optional[Path] = None,
) -> bool:
    """Check if sufficient disk space is available.

    Args:
        required_mb: Required space in megabytes.
        path: Path to check (uses temp directory if None).

    Returns:
        True if sufficient space is available.

    Raises:
        DiskSpaceError: If insufficient space.
    """
    import tempfile

    check_path = path or Path(tempfile.gettempdir())

    try:
        import shutil
        usage = shutil.disk_usage(check_path)
        available_mb = usage.free / (1024 * 1024)

        if available_mb < required_mb:
            raise DiskSpaceError(
                required_mb=required_mb,
                available_mb=available_mb,
                path=str(check_path),
            )

        return True

    except DiskSpaceError:
        raise
    except Exception:
        # If we can't check, assume it's okay
        return True


def estimate_temp_space_required(audio_info: AudioInfo) -> float:
    """Estimate temporary disk space required for processing.

    Estimates space needed for:
    - WAV conversion (worst case)
    - Segment extraction
    - Model intermediate files

    Args:
        audio_info: Audio file information.

    Returns:
        Estimated required space in megabytes.
    """
    # WAV at 16kHz mono is about 1.92 MB per minute
    # Add 50% buffer for processing overhead
    duration_minutes = audio_info.duration_seconds / 60
    wav_size_mb = duration_minutes * 1.92 * 1.5

    # Minimum 100MB for model operations
    return max(wav_size_mb, 100.0)
