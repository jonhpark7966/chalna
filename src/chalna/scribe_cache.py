"""Disk cache for raw ElevenLabs Scribe responses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from chalna.models import ScribeOptions
from chalna.validation import AudioInfo

TIMESTAMPS_GRANULARITY = "word"


def build_scribe_cache_metadata(
    *,
    audio_path: Path,
    audio_info: AudioInfo,
    model_id: str,
    language_code: Optional[str],
    options: ScribeOptions,
    timestamps_granularity: str = TIMESTAMPS_GRANULARITY,
) -> dict[str, Any]:
    """Build metadata used to identify equivalent Scribe requests."""
    file_size = audio_info.file_size_bytes
    if file_size is None and audio_path.exists():
        file_size = audio_path.stat().st_size

    return {
        "file_size_bytes": file_size,
        "duration_seconds": round(audio_info.duration_seconds, 3),
        "format_name": audio_info.format_name,
        "codec_name": audio_info.codec_name,
        "sample_rate": audio_info.sample_rate,
        "channels": audio_info.channels,
        "model_id": model_id,
        "language_code": language_code,
        "diarize": options.diarize,
        "tag_audio_events": options.tag_audio_events,
        "num_speakers": options.num_speakers,
        "timestamps_granularity": timestamps_granularity,
    }


def build_scribe_cache_key(metadata: dict[str, Any]) -> str:
    """Return a stable cache key from request metadata."""
    payload = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ScribeResponseCache:
    """Read and write raw Scribe response JSON documents."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)

    def path_for_key(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def get(self, metadata: dict[str, Any]) -> Optional[dict[str, Any]]:
        cache_key = build_scribe_cache_key(metadata)
        path = self.path_for_key(cache_key)
        if not path.exists():
            return None

        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if cached.get("metadata") != metadata:
            return None
        response = cached.get("response")
        return response if isinstance(response, dict) else None

    def put(self, metadata: dict[str, Any], response: dict[str, Any]) -> Path:
        cache_key = build_scribe_cache_key(metadata)
        path = self.path_for_key(cache_key)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_key": cache_key,
            "metadata": metadata,
            "response": response,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path
