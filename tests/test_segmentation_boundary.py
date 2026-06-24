from pathlib import Path

from chalna.models import Segment
from chalna.segmentation_boundary import _PcmAudio, apply_boundary_rule


def _segments() -> list[Segment]:
    return [
        Segment(1, 0.0, 1.0, "one"),
        Segment(2, 2.0, 3.0, "two"),
        Segment(3, 5.0, 6.0, "three"),
    ]


def test_word_boundary_rule_keeps_original_times():
    result = apply_boundary_rule(_segments(), rule="word_boundary")

    assert [(s.start_time, s.end_time) for s in result.segments] == [
        (0.0, 1.0),
        (2.0, 3.0),
        (5.0, 6.0),
    ]
    assert result.stats["unchanged_boundaries"] == 2


def test_midpoint_rule_splits_short_gaps_and_caps_long_gaps():
    result = apply_boundary_rule(_segments(), rule="midpoint_gap")

    assert [(s.start_time, s.end_time) for s in result.segments] == [
        (0.0, 1.5),
        (1.5, 3.25),
        (4.75, 6.0),
    ]
    assert result.stats["midpoint_boundaries"] == 1
    assert result.stats["capped_gap_boundaries"] == 1


def test_low_energy_rule_uses_quiet_window_for_short_gap(monkeypatch):
    class FakeAudio:
        def quiet_boundary_seconds(self, *args, **kwargs):
            return 1.25

    monkeypatch.setattr(
        _PcmAudio,
        "decode",
        classmethod(lambda cls, audio_path, sample_rate: FakeAudio()),
    )

    result = apply_boundary_rule(
        [Segment(1, 0.0, 1.0, "one"), Segment(2, 2.0, 3.0, "two")],
        rule="low_energy_gap_v1",
        audio_path=Path("source.wav"),
    )

    assert [(s.start_time, s.end_time) for s in result.segments] == [
        (0.0, 1.25),
        (1.25, 3.0),
    ]
    assert result.stats["low_energy_boundaries"] == 1
    assert result.stats["fallback_boundaries"] == 0
