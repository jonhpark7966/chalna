from pathlib import Path

from chalna.models import Segment
from chalna.pipeline import ChalnaPipeline
from chalna.scribe_cache import ScribeResponseCache
from chalna.validation import AudioInfo

SCRIBE_RESPONSE = {
    "language_code": "ko",
    "text": "안녕하세요 반갑습니다. 다음 문장입니다.",
    "words": [
        {"type": "word", "text": "안녕하세요", "start": 0.0, "end": 0.5, "speaker_id": "A"},
        {"type": "spacing", "text": " "},
        {"type": "word", "text": "반갑습니다.", "start": 0.55, "end": 1.2, "speaker_id": "A"},
        {"type": "spacing", "text": " "},
        {"type": "word", "text": "다음", "start": 1.3, "end": 1.6, "speaker_id": "A"},
        {"type": "spacing", "text": " "},
        {"type": "word", "text": "문장입니다.", "start": 1.65, "end": 2.2, "speaker_id": "A"},
    ],
}


class FakeScribeClient:
    def __init__(self, response=None):
        self.model_id = "scribe_v2"
        self.response = response or SCRIBE_RESPONSE
        self.calls = 0

    def transcribe(self, audio_path, *, language_code, options):
        self.calls += 1
        return self.response


def _patch_audio_validation(monkeypatch):
    audio_info = AudioInfo(
        duration_seconds=3.0,
        format_name="mp4",
        codec_name="aac",
        sample_rate=48000,
        channels=2,
        file_size_bytes=100,
    )
    monkeypatch.setattr("chalna.pipeline.validate_audio_file", lambda path: audio_info)
    monkeypatch.setattr("chalna.pipeline.estimate_temp_space_required", lambda info: 0.0)
    monkeypatch.setattr("chalna.pipeline.check_disk_space", lambda required_mb: True)


def test_pipeline_uses_cache_and_skips_llm_when_disabled(tmp_path: Path, monkeypatch):
    _patch_audio_validation(monkeypatch)
    audio_path = tmp_path / "source.mp4"
    audio_path.write_bytes(b"media")
    fake_client = FakeScribeClient()
    cache = ScribeResponseCache(tmp_path / "cache")
    pipeline = ChalnaPipeline(
        use_llm_refinement=False,
        scribe_client=fake_client,
        scribe_cache=cache,
    )

    first = pipeline.transcribe(audio_path, language="ko")
    second = pipeline.transcribe(audio_path, language="ko")

    assert fake_client.calls == 1
    assert first.metadata.model_version == "scribe_v2"
    assert first.metadata.aligned is False
    assert first.metadata.refined is False
    assert first.metadata.timestamp_source == "scribe_v2"
    assert second.segments[0].text == "안녕하세요 반갑습니다."


def test_pipeline_llm_refinement_never_uses_qwen_aligner(tmp_path: Path, monkeypatch):
    _patch_audio_validation(monkeypatch)
    audio_path = tmp_path / "source.mp4"
    audio_path.write_bytes(b"media")

    from chalna.llm_refiner import RefinementOutput

    def fake_refine_segments(
        segments,
        context=None,
        chunk_size=30,
        max_workers=5,
        progress_callback=None,
    ):
        if progress_callback:
            progress_callback("refining", 1.0)
        return RefinementOutput(
            segments=[
                Segment(1, 0.0, 1.1, "안녕하세요", "A"),
                Segment(2, 1.1, 2.2, "반갑습니다.", "A"),
            ],
            log=[
                {
                    "original_index": 1,
                    "status": "split",
                    "split_texts": ["안녕하세요", "반갑습니다."],
                    "new_segment_indices": [0, 1],
                    "original_start": 0.0,
                    "original_end": 1.2,
                }
            ],
            origin_map={0: 1, 1: 1},
        )

    monkeypatch.setattr("chalna.llm_refiner.refine_segments", fake_refine_segments)
    pipeline = ChalnaPipeline(
        use_llm_refinement=True,
        scribe_client=FakeScribeClient(),
        scribe_cache=ScribeResponseCache(tmp_path / "cache"),
    )

    result = pipeline.transcribe(audio_path, language="ko")

    assert not hasattr(pipeline, "_aligner")
    assert result.metadata.aligned is False
    assert result.metadata.refined is True
    assert result.segments[0].start_time == 0.0
    assert result.segments[0].end_time == 0.5
    assert result.segments[1].start_time == 0.55
