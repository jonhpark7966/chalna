"""Disk cache for LLM word-to-segment boundary plans."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from chalna.models import LlmSegmentationOptions, ScribeOptions

SEGMENTATION_PROMPT_VERSION = "scribe_llm_segmenter_v5_punctuation_boundary"


def build_segment_cache_metadata(
    *,
    scribe_cache_key: str,
    language_code: Optional[str],
    scribe_options: ScribeOptions,
    segmentation_options: LlmSegmentationOptions,
    prompt_version: str = SEGMENTATION_PROMPT_VERSION,
) -> dict[str, Any]:
    """Build metadata used to identify equivalent LLM segmentation requests."""
    return {
        "scribe_cache_key": scribe_cache_key,
        "language_code": language_code,
        "diarize": scribe_options.diarize,
        "tag_audio_events": scribe_options.tag_audio_events,
        "num_speakers": scribe_options.num_speakers,
        "llm_model": segmentation_options.model,
        "reasoning_effort": segmentation_options.reasoning_effort,
        "prompt_version": prompt_version,
        "max_segment_duration": segmentation_options.max_segment_duration,
        "max_words_per_call": segmentation_options.max_words_per_call,
    }


def build_segment_cache_key(metadata: dict[str, Any]) -> str:
    """Return a stable cache key from segmentation metadata."""
    payload = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SegmentPlanCache:
    """Read and write LLM segmentation plans."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)

    def path_for_key(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def get(self, metadata: dict[str, Any]) -> Optional[dict[str, Any]]:
        cache_key = build_segment_cache_key(metadata)
        path = self.path_for_key(cache_key)
        if not path.exists():
            return None

        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if cached.get("metadata") != metadata:
            return None
        plan = cached.get("plan")
        return plan if isinstance(plan, dict) else None

    def put(self, metadata: dict[str, Any], plan: dict[str, Any]) -> Path:
        cache_key = build_segment_cache_key(metadata)
        path = self.path_for_key(cache_key)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_key": cache_key,
            "metadata": metadata,
            "plan": plan,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path
