from pathlib import Path

from chalna.models import ScribeOptions
from chalna.scribe_cache import (
    ScribeResponseCache,
    build_scribe_cache_key,
    build_scribe_cache_metadata,
)
from chalna.validation import AudioInfo


def _audio_info() -> AudioInfo:
    return AudioInfo(
        duration_seconds=12.3456,
        format_name="mp4",
        codec_name="aac",
        sample_rate=48000,
        channels=2,
        file_size_bytes=12345,
    )


def test_scribe_cache_key_is_stable_for_same_metadata(tmp_path: Path):
    audio_path = tmp_path / "sample.mp4"
    audio_path.write_bytes(b"content")
    options = ScribeOptions(diarize=True, tag_audio_events=True, num_speakers=2)

    metadata_a = build_scribe_cache_metadata(
        audio_path=audio_path,
        audio_info=_audio_info(),
        model_id="scribe_v2",
        language_code="ko",
        options=options,
    )
    metadata_b = build_scribe_cache_metadata(
        audio_path=audio_path,
        audio_info=_audio_info(),
        model_id="scribe_v2",
        language_code="ko",
        options=options,
    )

    assert build_scribe_cache_key(metadata_a) == build_scribe_cache_key(metadata_b)


def test_scribe_cache_key_changes_when_options_change(tmp_path: Path):
    audio_path = tmp_path / "sample.mp4"
    audio_path.write_bytes(b"content")

    metadata_a = build_scribe_cache_metadata(
        audio_path=audio_path,
        audio_info=_audio_info(),
        model_id="scribe_v2",
        language_code="ko",
        options=ScribeOptions(diarize=True, tag_audio_events=True, num_speakers=None),
    )
    metadata_b = build_scribe_cache_metadata(
        audio_path=audio_path,
        audio_info=_audio_info(),
        model_id="scribe_v2",
        language_code="ko",
        options=ScribeOptions(diarize=False, tag_audio_events=True, num_speakers=None),
    )

    assert build_scribe_cache_key(metadata_a) != build_scribe_cache_key(metadata_b)


def test_scribe_response_cache_roundtrip(tmp_path: Path):
    cache = ScribeResponseCache(tmp_path)
    metadata = {"file_size_bytes": 1, "model_id": "scribe_v2"}
    response = {"text": "hello", "words": []}

    cache.put(metadata, response)

    assert cache.get(metadata) == response
    assert cache.get({"file_size_bytes": 2, "model_id": "scribe_v2"}) is None
