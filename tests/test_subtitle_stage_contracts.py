"""
Stage-level tests for the subtitle generation pipeline.

These tests intentionally avoid loading VibeVoice, Qwen, or Codex.  The goal is
to lock down the contract for each pipeline stage with lightweight fakes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from chalna.models import Segment
from chalna.pipeline import ChalnaPipeline
from chalna.srt_utils import parse_srt, segments_to_srt


@dataclass
class FakeAlignedWord:
    text: str
    start_time: float
    end_time: float


class FakeAligner:
    def __init__(self, words: list[FakeAlignedWord]):
        self.words = words
        self.calls: list[dict] = []

    def align(self, audio: str, text: str, language: str):
        self.calls.append({"audio": audio, "text": text, "language": language})
        return [self.words]


def no_op_ffmpeg(*args, **kwargs):
    return None


def test_asr_raw_segments_parse_vibevoice_json_and_skip_sound_markers():
    pipeline = ChalnaPipeline(use_alignment=False, use_llm_refinement=False)
    content = json.dumps(
        [
            {
                "Start time": 0.0,
                "End time": 1.5,
                "Speaker ID": "Speaker 1",
                "Content": "안녕하세요.",
            },
            {
                "Start time": 1.5,
                "End time": 2.0,
                "Speaker ID": "Speaker 1",
                "Content": "[music]",
            },
            {
                "Start": 2.0,
                "End": 3.25,
                "Speaker": 2,
                "Content": "반갑습니다.",
            },
        ],
        ensure_ascii=False,
    )

    segments = pipeline._parse_vibevoice_response(content)

    assert [seg.index for seg in segments] == [1, 2]
    assert [seg.text for seg in segments] == ["안녕하세요.", "반갑습니다."]
    assert segments[0].speaker_id == "Speaker 1"
    assert segments[1].speaker_id == "2"
    assert segments[1].start_time == 2.0
    assert segments[1].end_time == 3.25


def test_asr_raw_segments_trim_repeated_token_tail():
    pipeline = ChalnaPipeline(use_alignment=False, use_llm_refinement=False)
    repeated_tail = " ".join(["레"] * 40)
    content = json.dumps(
        [
            {
                "Start time": 0.0,
                "End time": 8.0,
                "Speaker ID": "Speaker 1",
                "Content": f"응. 딥마인드 출신이라고 하시고 그 {repeated_tail}",
            },
        ],
        ensure_ascii=False,
    )

    segments = pipeline._parse_vibevoice_response(content)

    assert len(segments) == 1
    assert segments[0].text == "응. 딥마인드 출신이라고 하시고 그"


def test_output_sanitizer_drops_invalid_timestamps_and_repetition():
    pipeline = ChalnaPipeline(use_alignment=False, use_llm_refinement=False)
    repeated_tail = " ".join(["레"] * 40)

    sanitized = pipeline._sanitize_segments_for_output(
        [
            Segment(1, 0.0, 1.0, "정상 문장입니다."),
            Segment(2, 2.0, 1.9, "시간이 잘못된 문장입니다."),
            Segment(3, 3.0, 4.0, f"의미 있는 prefix {repeated_tail}"),
        ],
        verbose=False,
    )

    assert [seg.index for seg in sanitized] == [1, 2]
    assert [seg.text for seg in sanitized] == [
        "정상 문장입니다.",
        "의미 있는 prefix",
    ]
    assert all(seg.end_time > seg.start_time for seg in sanitized)


def test_final_merge_combines_tight_same_speaker_segments():
    pipeline = ChalnaPipeline(use_alignment=False, use_llm_refinement=False)

    merged = pipeline._merge_tight_same_speaker_segments(
        [
            Segment(1, 0.0, 1.0, "첫 문장입니다.", speaker_id="Speaker 1"),
            Segment(2, 1.0, 2.0, "두 번째 문장입니다.", speaker_id="Speaker 1"),
            Segment(3, 2.5, 3.0, "다른 speaker입니다.", speaker_id="Speaker 2"),
        ],
        verbose=False,
    )

    assert len(merged) == 2
    assert merged[0].start_time == 0.0
    assert merged[0].end_time == 2.0
    assert merged[0].text == "첫 문장입니다. 두 번째 문장입니다."
    assert merged[1].speaker_id == "Speaker 2"


def test_forced_alignment_splits_multi_sentence_segment(monkeypatch, tmp_path):
    monkeypatch.setattr("subprocess.run", no_op_ffmpeg)
    pipeline = ChalnaPipeline(use_alignment=True, use_llm_refinement=False)
    pipeline._aligner = FakeAligner(
        [
            FakeAlignedWord("첫", 2.10, 2.20),
            FakeAlignedWord("문장입니다.", 2.25, 3.00),
            FakeAlignedWord("두", 3.20, 3.30),
            FakeAlignedWord("번째", 3.35, 3.55),
            FakeAlignedWord("문장입니다.", 3.60, 4.40),
        ]
    )
    segments = [
        Segment(
            index=1,
            start_time=10.0,
            end_time=13.0,
            text="첫 문장입니다. 두 번째 문장입니다.",
            speaker_id="Speaker 1",
        )
    ]

    aligned = pipeline._run_alignment(tmp_path / "audio.wav", segments, verbose=False)

    assert [seg.index for seg in aligned] == [1, 2]
    assert [seg.text for seg in aligned] == ["첫 문장입니다.", "두 번째 문장입니다."]
    assert aligned[0].start_time == pytest.approx(9.95)
    assert aligned[0].end_time == pytest.approx(11.10)
    assert aligned[1].start_time == pytest.approx(11.10)
    assert aligned[1].end_time == pytest.approx(12.55)
    assert pipeline._last_alignment_log[0]["status"] == "split"


@pytest.mark.xfail(
    reason="semantic/length/edit-point aware segmentation is not implemented yet",
    raises=(ModuleNotFoundError, ImportError, AttributeError, AssertionError),
)
def test_semantic_length_edit_point_segmentation_preserves_constraints():
    from chalna.subtitle_segmentation import TimedTextUnit, segment_with_constraints

    units = [
        TimedTextUnit(text="오늘은", start_time=0.0, end_time=0.4),
        TimedTextUnit(text="정말", start_time=0.5, end_time=0.8),
        TimedTextUnit(text="수고하셨습니다", start_time=0.9, end_time=1.7),
        TimedTextUnit(text="그리고", start_time=3.2, end_time=3.5),
        TimedTextUnit(text="다음", start_time=3.7, end_time=4.0),
        TimedTextUnit(text="문장입니다.", start_time=4.1, end_time=5.0),
    ]

    segments = segment_with_constraints(
        units,
        edit_points=[3.0],
        max_duration=3.0,
        max_chars=18,
        speaker_id="Speaker 1",
    )

    assert segments
    assert all(not (seg.start_time < 3.0 < seg.end_time) for seg in segments)
    assert all(seg.duration <= 3.0 for seg in segments)
    assert all(seg.text.strip() != "습니다" for seg in segments)
    assert any("수고하셨습니다" in seg.text for seg in segments)


def test_llm_text_refinement_parses_split_markers_and_keeps_origin_map(monkeypatch):
    from chalna import llm_refiner

    response = json.dumps(
        [
            {"index": 1, "text": "첫 문장입니다. |SPLIT| 두 번째 문장입니다."},
            {"index": 2, "text": "맞춤법을 고친 문장입니다."},
        ],
        ensure_ascii=False,
    )
    monkeypatch.setattr(llm_refiner, "call_codex_cli", lambda prompt: response)

    output = llm_refiner.refine_segments(
        [
            Segment(1, 0.0, 6.0, "첫 문장입니다 두 번째 문장입니다."),
            Segment(2, 6.0, 8.0, "맞춤법을 고친 문장입니다"),
        ],
        chunk_size=30,
        max_workers=1,
    )

    assert [seg.text for seg in output.segments] == [
        "첫 문장입니다.",
        "두 번째 문장입니다.",
        "맞춤법을 고친 문장입니다.",
    ]
    assert output.segments[0].start_time == 0.0
    assert output.segments[0].end_time == 3.0
    assert output.segments[1].start_time == 3.0
    assert output.segments[1].end_time == 6.0
    assert output.origin_map == {0: 1, 1: 1, 2: 2}
    assert [entry["status"] for entry in output.log[:2]] == ["split", "refined"]


def test_final_alignment_derives_split_boundaries_from_word_timestamps(monkeypatch, tmp_path):
    monkeypatch.setattr("subprocess.run", no_op_ffmpeg)
    pipeline = ChalnaPipeline(use_alignment=True, use_llm_refinement=False)
    pipeline._aligner = FakeAligner(
        [
            FakeAlignedWord("첫", 2.10, 2.20),
            FakeAlignedWord("문장입니다.", 2.25, 3.00),
            FakeAlignedWord("두", 3.20, 3.30),
            FakeAlignedWord("번째", 3.35, 3.55),
            FakeAlignedWord("문장입니다.", 3.60, 4.40),
        ]
    )

    aligned = pipeline._align_split_segments(
        audio_path=tmp_path / "audio.wav",
        original_start=20.0,
        original_end=24.0,
        split_texts=["첫 문장입니다.", "두 번째 문장입니다."],
        speaker_id="Speaker 1",
        confidence=0.9,
    )

    assert aligned is not None
    assert [seg.text for seg in aligned] == ["첫 문장입니다.", "두 번째 문장입니다."]
    assert aligned[0].start_time == pytest.approx(19.95)
    assert aligned[0].end_time == pytest.approx(21.10)
    assert aligned[1].start_time == pytest.approx(21.10)
    assert aligned[1].end_time == pytest.approx(22.55)


def test_validator_reports_edit_crossing_long_segment_and_orphaned_ending():
    from chalna.subtitle_validator import validate_segments

    issues = validate_segments(
        [
            Segment(1, 0.0, 5.5, "오늘은 정말 수고하셨"),
            Segment(2, 5.5, 5.9, "습니다"),
        ],
        edit_points=[3.0],
        max_duration=4.0,
    )

    issue_codes = {issue.code for issue in issues}
    assert "crosses_edit_point" in issue_codes
    assert "duration_too_long" in issue_codes
    assert "orphaned_korean_ending" in issue_codes


def test_srt_stage_round_trips_final_segments_without_speaker_labels():
    segments = [
        Segment(1, 0.0, 1.5, "첫 문장입니다.", speaker_id="Speaker 1"),
        Segment(2, 1.5, 3.0, "두 번째 문장입니다.", speaker_id="Speaker 1"),
    ]

    srt = segments_to_srt(segments, include_speaker=False)
    parsed = parse_srt(srt)

    assert "00:00:00,000 --> 00:00:01,500" in srt
    assert "00:00:01,500 --> 00:00:03,000" in srt
    assert "[Speaker 1]" not in srt
    assert [seg["text"] for seg in parsed] == ["첫 문장입니다.", "두 번째 문장입니다."]
