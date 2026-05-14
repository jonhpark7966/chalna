"""
Tests for VibeVoice chunked transcription merging.
"""

from pathlib import Path

import pytest

from chalna.exceptions import VibevoiceAPIError
from chalna.models import Segment
from chalna.pipeline import ChalnaPipeline


def seg(start: float, end: float, text: str) -> Segment:
    return Segment(index=0, start_time=start, end_time=end, text=text)


class FakeChunkedPipeline(ChalnaPipeline):
    def __init__(self, tmp_path: Path, chunk_outputs: list[list[Segment] | Exception]):
        self.tmp_path = tmp_path
        self.chunk_outputs = chunk_outputs
        self.extract_calls: list[tuple[float, float]] = []
        self.deleted_paths: list[Path] = []
        self.call_index = 0

    def _load_vibevoice(self) -> None:
        return None

    def _extract_audio_chunk(self, audio_path: Path, start: float, duration: float) -> Path:
        chunk_path = self.tmp_path / f"chunk_{len(self.extract_calls)}.wav"
        chunk_path.write_bytes(b"fake audio")
        self.extract_calls.append((start, duration))
        return chunk_path

    def _call_vibevoice(
        self,
        audio_path: Path,
        duration: float,
        context: str | None = None,
        max_new_tokens: int = 32768,
        max_continuations: int = 10,
    ) -> list[Segment]:
        output = self.chunk_outputs[self.call_index]
        self.call_index += 1
        if isinstance(output, Exception):
            raise output

        # Return copies because the production code mutates timestamps in place.
        return [
            Segment(
                index=s.index,
                start_time=s.start_time,
                end_time=s.end_time,
                text=s.text,
                speaker_id=s.speaker_id,
                confidence=s.confidence,
            )
            for s in output
        ]


def test_chunked_transcription_uses_7_minute_chunks_with_10_second_overlap(tmp_path):
    pipeline = FakeChunkedPipeline(
        tmp_path,
        [
            [seg(0, 100, "first chunk")],
            [seg(0, 80, "second chunk")],
        ],
    )

    segments, per_chunk = pipeline._call_vibevoice_chunked(
        tmp_path / "source.wav",
        total_duration=900,
        context=None,
        max_new_tokens=1024,
        progress_callback=None,
    )

    assert pipeline.extract_calls == [(0.0, 420.0), (410.0, 490.0)]
    assert [s.text for s in segments] == ["first chunk", "second chunk"]
    assert segments[1].start_time == 410.0
    assert segments[1].end_time == 490.0
    assert len(per_chunk) == 2


def test_overlap_segments_are_deduplicated_by_midpoint_ownership(tmp_path):
    pipeline = FakeChunkedPipeline(
        tmp_path,
        [
            [
                seg(0, 100, "early"),
                seg(414, 418, "boundary duplicate from previous chunk"),
            ],
            [
                seg(4, 8, "boundary duplicate from next chunk"),
                seg(20, 30, "after boundary"),
            ],
        ],
    )

    segments, per_chunk = pipeline._call_vibevoice_chunked(
        tmp_path / "source.wav",
        total_duration=900,
        context=None,
        max_new_tokens=1024,
        progress_callback=None,
    )

    assert [s.text for s in segments] == [
        "early",
        "boundary duplicate from next chunk",
        "after boundary",
    ]
    assert [s.text for s in per_chunk[0]] == ["early"]
    assert [s.text for s in per_chunk[1]] == [
        "boundary duplicate from next chunk",
        "after boundary",
    ]
    assert segments[1].start_time == 414.0
    assert segments[1].end_time == 418.0


def test_empty_chunk_does_not_stall_and_progress_reaches_all_chunks(tmp_path):
    progress_updates = []
    pipeline = FakeChunkedPipeline(
        tmp_path,
        [
            [],
            [seg(10, 20, "recovered in next chunk")],
        ],
    )

    segments, per_chunk = pipeline._call_vibevoice_chunked(
        tmp_path / "source.wav",
        total_duration=900,
        context=None,
        max_new_tokens=1024,
        progress_callback=lambda stage, progress, **extra: progress_updates.append(
            (stage, progress, extra)
        ),
    )

    assert pipeline.extract_calls == [(0.0, 420.0), (410.0, 490.0)]
    assert [s.text for s in segments] == ["recovered in next chunk"]
    assert per_chunk[0] == []
    assert progress_updates[-1] == (
        "transcribing",
        1.0,
        {"chunk": 2, "total_chunks": 2},
    )


def test_chunk_temp_file_is_deleted_when_vibevoice_fails(tmp_path):
    pipeline = FakeChunkedPipeline(
        tmp_path,
        [VibevoiceAPIError("generation failed")],
    )

    with pytest.raises(VibevoiceAPIError, match="generation failed"):
        pipeline._call_vibevoice_chunked(
            tmp_path / "source.wav",
            total_duration=500,
            context=None,
            max_new_tokens=1024,
            progress_callback=None,
        )

    chunk_path = tmp_path / "chunk_0.wav"
    assert pipeline.extract_calls == [(0.0, 500.0)]
    assert not chunk_path.exists()
