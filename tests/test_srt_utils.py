"""
Tests for SRT utilities.
"""

import pytest

from chalna.models import Segment
from chalna.srt_utils import (
    format_timestamp,
    parse_srt_timestamp,
    segments_to_srt,
    parse_srt,
)


class TestFormatTimestamp:
    def test_zero(self):
        assert format_timestamp(0) == "00:00:00,000"

    def test_seconds(self):
        assert format_timestamp(5.5) == "00:00:05,500"

    def test_minutes(self):
        assert format_timestamp(65.123) == "00:01:05,123"

    def test_hours(self):
        assert format_timestamp(3661.999) == "01:01:01,999"

    def test_negative(self):
        assert format_timestamp(-5) == "00:00:00,000"


class TestParseSrtTimestamp:
    def test_zero(self):
        assert parse_srt_timestamp("00:00:00,000") == 0.0

    def test_seconds(self):
        assert parse_srt_timestamp("00:00:05,500") == 5.5

    def test_minutes(self):
        assert abs(parse_srt_timestamp("00:01:05,123") - 65.123) < 0.001

    def test_hours(self):
        assert abs(parse_srt_timestamp("01:01:01,999") - 3661.999) < 0.001

    def test_invalid(self):
        assert parse_srt_timestamp("invalid") == 0.0


class TestSegmentsToSrt:
    def test_basic(self):
        segments = [
            Segment(index=1, start_time=0.0, end_time=2.0, text="Hello"),
            Segment(index=2, start_time=2.5, end_time=4.0, text="World"),
        ]
        srt = segments_to_srt(segments, include_speaker=False)

        assert "1\n" in srt
        assert "00:00:00,000 --> 00:00:02,000" in srt
        assert "Hello" in srt
        assert "2\n" in srt
        assert "World" in srt

    def test_with_speaker(self):
        segments = [
            Segment(index=1, start_time=0.0, end_time=2.0, text="Hello", speaker_id="Speaker 0"),
        ]
        srt = segments_to_srt(segments, include_speaker=True)

        assert "[Speaker 0] Hello" in srt

    def test_without_speaker(self):
        segments = [
            Segment(index=1, start_time=0.0, end_time=2.0, text="Hello", speaker_id="Speaker 0"),
        ]
        srt = segments_to_srt(segments, include_speaker=False)

        assert "[Speaker 0]" not in srt
        assert "Hello" in srt


class TestParseSrt:
    def test_basic(self):
        srt_content = """1
00:00:00,000 --> 00:00:02,000
Hello

2
00:00:02,500 --> 00:00:04,000
World
"""
        segments = parse_srt(srt_content)

        assert len(segments) == 2
        assert segments[0]["index"] == 1
        assert segments[0]["start_time"] == 0.0
        assert segments[0]["end_time"] == 2.0
        assert segments[0]["text"] == "Hello"

    def test_with_speaker(self):
        srt_content = """1
00:00:00,000 --> 00:00:02,000
[Speaker 0] Hello
"""
        segments = parse_srt(srt_content)

        assert segments[0]["speaker_id"] == "Speaker 0"
        assert segments[0]["text"] == "Hello"
