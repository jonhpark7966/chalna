"""
Regression tests for invalid forced-alignment timestamp guards.
"""

from __future__ import annotations

from dataclasses import dataclass

from chalna.models import Segment
from chalna.pipeline import ChalnaPipeline


@dataclass
class FakeAlignedWord:
    text: str
    start_time: float
    end_time: float


class FakeAligner:
    def __init__(self, words: list[FakeAlignedWord]):
        self.words = words

    def align(self, audio: str, text: str, language: str):
        return [self.words]


def no_op_ffmpeg(*args, **kwargs):
    return None


def test_split_realign_rejects_boundaries_invalidated_by_clamping(monkeypatch, tmp_path):
    monkeypatch.setattr("subprocess.run", no_op_ffmpeg)
    pipeline = ChalnaPipeline(use_alignment=True, use_llm_refinement=False)
    pipeline._aligner = FakeAligner(
        [
            FakeAlignedWord("늦은", 4.00, 4.20),
            FakeAlignedWord("단어", 4.30, 4.50),
        ]
    )

    aligned = pipeline._align_split_segments(
        audio_path=tmp_path / "audio.wav",
        original_start=10.0,
        original_end=11.0,
        split_texts=["늦은 단어"],
        speaker_id="Speaker 1",
        confidence=0.9,
    )

    assert aligned is None


def test_single_realign_rejects_boundaries_invalidated_by_clamping(monkeypatch, tmp_path):
    monkeypatch.setattr("subprocess.run", no_op_ffmpeg)
    pipeline = ChalnaPipeline(use_alignment=True, use_llm_refinement=False)
    pipeline._aligner = FakeAligner(
        [
            FakeAlignedWord("늦은", 4.00, 4.20),
            FakeAlignedWord("단어", 4.30, 4.50),
        ]
    )

    aligned = pipeline._align_single_segment(
        Segment(1, 10.0, 11.0, "늦은 단어", speaker_id="Speaker 1"),
        tmp_path / "audio.wav",
    )

    assert aligned is None
