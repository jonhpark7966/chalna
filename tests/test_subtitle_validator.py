"""
Tests for subtitle validation and edit-safe audio boundary checks.
"""

from __future__ import annotations

from dataclasses import dataclass

from chalna.models import Segment
from chalna.subtitle_validator import (
    SubtitleValidationIssue,
    summarize_validation_diagnostics,
    validate_audio_boundaries,
    validate_segments,
)


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


def test_validate_audio_boundaries_extracts_source_audio_once(monkeypatch, tmp_path):
    ffmpeg_calls = []

    def record_ffmpeg(cmd, *args, **kwargs):
        ffmpeg_calls.append(cmd)
        return None

    monkeypatch.setattr("subprocess.run", record_ffmpeg)
    aligner = FakeAligner([FakeAlignedWord("문장", 0.75, 1.00)])
    segments = [
        Segment(1, 10.50, 11.50, "첫 문장"),
        Segment(2, 12.50, 13.50, "두 번째 문장"),
    ]

    validate_audio_boundaries(tmp_path / "audio.mp4", segments, aligner)

    full_audio_extracts = [cmd for cmd in ffmpeg_calls if "-ss" not in cmd]
    window_extracts = [cmd for cmd in ffmpeg_calls if "-ss" in cmd]
    assert len(full_audio_extracts) == 1
    assert len(window_extracts) == 2


def test_validate_audio_boundaries_reports_tight_start_and_end(monkeypatch, tmp_path):
    monkeypatch.setattr("subprocess.run", no_op_ffmpeg)
    aligner = FakeAligner(
        [
            FakeAlignedWord("첫", 0.50, 0.70),
            FakeAlignedWord("문장", 1.20, 1.50),
        ]
    )
    segment = Segment(1, 10.50, 11.50, "첫 문장")

    issues = validate_audio_boundaries(
        tmp_path / "audio.wav",
        [segment],
        aligner,
        scan_padding=0.5,
        min_start_padding=0.15,
        min_end_padding=0.15,
    )

    assert {issue.code for issue in issues} == {
        "tight_start_boundary",
        "tight_end_boundary",
    }
    assert issues[0].details["speech_start"] == 10.50
    assert issues[1].details["speech_end"] == 11.50


def test_validate_audio_boundaries_allows_room_around_speech(monkeypatch, tmp_path):
    monkeypatch.setattr("subprocess.run", no_op_ffmpeg)
    aligner = FakeAligner(
        [
            FakeAlignedWord("첫", 0.75, 0.90),
            FakeAlignedWord("문장", 1.05, 1.25),
        ]
    )
    segment = Segment(1, 10.50, 11.50, "첫 문장")

    issues = validate_audio_boundaries(
        tmp_path / "audio.wav",
        [segment],
        aligner,
        scan_padding=0.5,
        min_start_padding=0.15,
        min_end_padding=0.15,
    )

    assert issues == []


def test_validate_audio_boundaries_reports_cutoff_before_tightness(monkeypatch, tmp_path):
    monkeypatch.setattr("subprocess.run", no_op_ffmpeg)
    aligner = FakeAligner(
        [
            FakeAlignedWord("잘린", 0.40, 0.70),
            FakeAlignedWord("문장", 1.20, 1.60),
        ]
    )
    segment = Segment(1, 10.50, 11.50, "잘린 문장")

    issues = validate_audio_boundaries(
        tmp_path / "audio.wav",
        [segment],
        aligner,
        scan_padding=0.5,
        min_start_padding=0.15,
        min_end_padding=0.15,
        cutoff_tolerance=0.03,
    )

    assert {issue.code for issue in issues} == {
        "cuts_speech_start",
        "cuts_speech_end",
    }


def test_validate_audio_boundaries_uses_context_but_checks_target_range(monkeypatch, tmp_path):
    monkeypatch.setattr("subprocess.run", no_op_ffmpeg)
    aligner = FakeAligner(
        [
            FakeAlignedWord("이전", 0.00, 0.20),
            FakeAlignedWord("현재", 1.00, 1.15),
            FakeAlignedWord("문장", 1.85, 2.00),
            FakeAlignedWord("다음", 2.50, 2.70),
        ]
    )
    segments = [
        Segment(1, 9.00, 10.00, "이전"),
        Segment(2, 10.00, 11.00, "현재 문장"),
        Segment(3, 11.00, 12.00, "다음"),
    ]

    issues = validate_audio_boundaries(
        tmp_path / "audio.wav",
        segments,
        aligner,
        scan_padding=0.5,
        min_start_padding=0.15,
        min_end_padding=0.15,
    )

    target_issues = [issue for issue in issues if issue.segment_index == 2]
    assert {issue.code for issue in target_issues} == {
        "tight_start_boundary",
        "tight_end_boundary",
    }
    assert aligner.calls[1]["text"] == "이전 현재 문장 다음"
    assert target_issues[0].details["first_word"] == "현재"
    assert target_issues[0].details["last_word"] == "문장"
    assert target_issues[0].details["context_segment_indices"] == [1, 2, 3]
    assert target_issues[0].details["target_match_ratio"] == 1.0
    assert target_issues[0].details["alignment_relation"] == "inside_segment"


def test_validate_audio_boundaries_uses_text_match_when_char_range_drifts(monkeypatch, tmp_path):
    monkeypatch.setattr("subprocess.run", no_op_ffmpeg)
    aligner = FakeAligner(
        [
            FakeAlignedWord("앞앞앞앞", 0.00, 0.20),
            FakeAlignedWord("노이즈", 0.30, 0.40),
            FakeAlignedWord("현재", 1.00, 1.15),
            FakeAlignedWord("문장", 1.85, 2.00),
            FakeAlignedWord("뒤뒤뒤뒤", 2.50, 2.70),
        ]
    )
    segments = [
        Segment(1, 9.00, 10.00, "앞앞앞앞"),
        Segment(2, 10.00, 11.00, "현재 문장"),
        Segment(3, 11.00, 12.00, "뒤뒤뒤뒤"),
    ]

    issues = validate_audio_boundaries(
        tmp_path / "audio.wav",
        segments,
        aligner,
        scan_padding=0.5,
        min_start_padding=0.15,
        min_end_padding=0.15,
    )

    target_issue = next(issue for issue in issues if issue.segment_index == 2)
    assert target_issue.details["target_word_range_method"] == "text_match"
    assert target_issue.details["first_word"] == "현재"
    assert target_issue.details["last_word"] == "문장"


def test_boundary_diagnostics_summarize_gap_and_alignment_causes():
    segments = [
        Segment(1, 0.0, 1.0, "첫 문장", speaker_id="A"),
        Segment(2, 1.0, 2.0, "두 번째", speaker_id="A"),
    ]
    issues = [
        SubtitleValidationIssue(
            code="cuts_speech_end",
            message="Segment ends before speech.",
            segment_index=1,
            severity="error",
            start_time=0.0,
            end_time=1.0,
            text="첫 문장",
            details={
                "segment_start": 0.0,
                "segment_end": 1.0,
                "speech_start": 0.1,
                "speech_end": 1.4,
                "start_padding": 0.1,
                "end_padding": -0.4,
                "scan_padding": 0.5,
                "alignment_relation": "ends_after_segment",
                "target_word_range_method": "char_range",
                "target_match_ratio": 0.9,
            },
        )
    ]

    diagnostics = summarize_validation_diagnostics(segments, issues)

    assert diagnostics["issue_counts"] == {"cuts_speech_end": 1}
    assert diagnostics["gap"]["zero_gap"] == 1
    assert diagnostics["gap"]["same_speaker_zero_gap"] == 1
    assert diagnostics["gap"]["cuts_speech_end_by_next_gap"]["zero"]["rate"] == 1.0
    assert diagnostics["alignment"]["relation_counts"] == {"ends_after_segment": 1}


def test_validate_segments_reports_common_subtitle_contract_issues():
    issues = validate_segments(
        [
            Segment(1, 0.0, 5.5, "오늘은 정말 수고하셨"),
            Segment(2, 5.5, 5.9, "습니다"),
        ],
        edit_points=[3.0],
        max_duration=4.0,
    )

    assert {issue.code for issue in issues} == {
        "crosses_edit_point",
        "duration_too_long",
        "orphaned_korean_ending",
    }


def test_validate_segments_reports_repeated_token_loop():
    repeated_tail = " ".join(["레"] * 40)

    issues = validate_segments(
        [
            Segment(1, 0.0, 8.0, f"의미 있는 prefix {repeated_tail}"),
        ],
    )

    assert "repeated_token_loop" in {issue.code for issue in issues}
