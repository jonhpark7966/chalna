"""ElevenLabs Scribe API client."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import httpx

from chalna.exceptions import ElevenLabsAPIError
from chalna.models import ScribeOptions
from chalna.scribe_cache import TIMESTAMPS_GRANULARITY
from chalna.settings import settings


class ScribeClient:
    """Small wrapper around ElevenLabs speech-to-text."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        model_id: Optional[str] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.elevenlabs_api_key
        self.base_url = (base_url or settings.elevenlabs_base_url).rstrip("/")
        self.timeout = timeout if timeout is not None else settings.scribe_timeout
        self.model_id = model_id or settings.scribe_model_id

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language_code: Optional[str],
        options: ScribeOptions,
    ) -> dict:
        """Transcribe an audio/video file with Scribe and return the raw JSON response."""
        if not self.api_key:
            raise ElevenLabsAPIError("ELEVENLABS_API_KEY is not configured")

        audio_path = Path(audio_path)
        data = {
            "model_id": self.model_id,
            "diarize": str(options.diarize).lower(),
            "tag_audio_events": str(options.tag_audio_events).lower(),
            "timestamps_granularity": TIMESTAMPS_GRANULARITY,
        }
        if language_code:
            data["language_code"] = language_code
        if options.num_speakers is not None:
            data["num_speakers"] = str(options.num_speakers)

        try:
            with audio_path.open("rb") as f:
                files = {"file": (audio_path.name, f, "application/octet-stream")}
                response = httpx.post(
                    f"{self.base_url}/v1/speech-to-text",
                    headers={"xi-api-key": self.api_key},
                    data=data,
                    files=files,
                    timeout=self.timeout,
                )
        except httpx.HTTPError as e:
            raise ElevenLabsAPIError(f"ElevenLabs request failed: {e}", cause=e) from e
        except OSError as e:
            raise ElevenLabsAPIError(f"Failed to read input file: {audio_path}", cause=e) from e

        if response.status_code >= 400:
            body = response.text[:1000]
            raise ElevenLabsAPIError(
                f"ElevenLabs returned HTTP {response.status_code}: {body}",
                status_code=response.status_code,
            )

        try:
            payload = response.json()
        except ValueError as e:
            raise ElevenLabsAPIError("ElevenLabs returned invalid JSON", cause=e) from e

        if not isinstance(payload, dict):
            raise ElevenLabsAPIError("ElevenLabs returned an unexpected response shape")

        return payload
