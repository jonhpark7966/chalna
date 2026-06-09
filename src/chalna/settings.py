"""Chalna settings configured via environment variables or a .env file."""

from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SCRIBE_CACHE_DIR = _PROJECT_ROOT / "results" / "scribe_cache"
_DEFAULT_SEGMENT_CACHE_DIR = _PROJECT_ROOT / "results" / "segment_cache"


class Settings(BaseSettings):
    # Deprecated: kept so older .env files do not fail validation.
    vibevoice_model_path: str = "microsoft/VibeVoice-ASR"
    sentry_dsn: str = ""
    sentry_environment: str = "local"
    sentry_release: str = ""
    sentry_traces_sample_rate: float = 0.05

    elevenlabs_api_key: Optional[str] = Field(
        default=None,
        validation_alias="ELEVENLABS_API_KEY",
    )
    elevenlabs_base_url: str = Field(
        default="https://api.elevenlabs.io",
        validation_alias="ELEVENLABS_BASE_URL",
    )
    scribe_model_id: str = Field(
        default="scribe_v2",
        validation_alias="SCRIBE_MODEL_ID",
    )
    scribe_cache_dir: str = Field(
        default=str(_DEFAULT_SCRIBE_CACHE_DIR),
        validation_alias=AliasChoices("CHALNA_SCRIBE_CACHE_DIR", "SCRIBE_CACHE_DIR"),
    )
    scribe_timeout: float = Field(
        default=600.0,
        validation_alias=AliasChoices("CHALNA_SCRIBE_TIMEOUT", "SCRIBE_TIMEOUT"),
    )
    llm_segmentation_cache_dir: str = Field(
        default=str(_DEFAULT_SEGMENT_CACHE_DIR),
        validation_alias=AliasChoices(
            "CHALNA_LLM_SEGMENTATION_CACHE_DIR",
            "LLM_SEGMENTATION_CACHE_DIR",
        ),
    )
    llm_segmentation_model: str = Field(
        default="gpt-5.5",
        validation_alias=AliasChoices(
            "CHALNA_LLM_SEGMENTATION_MODEL",
            "LLM_SEGMENTATION_MODEL",
        ),
    )
    llm_segmentation_reasoning_effort: str = Field(
        default="xhigh",
        validation_alias=AliasChoices(
            "CHALNA_LLM_SEGMENTATION_REASONING_EFFORT",
            "LLM_SEGMENTATION_REASONING_EFFORT",
        ),
    )
    llm_segmentation_timeout: int = Field(
        default=180,
        validation_alias=AliasChoices(
            "CHALNA_LLM_SEGMENTATION_TIMEOUT",
            "LLM_SEGMENTATION_TIMEOUT",
        ),
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
