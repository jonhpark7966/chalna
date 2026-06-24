from chalna.models import Segment
from chalna.overlap_protection import protect_overlapped_segments


def test_merges_only_consecutive_segments_that_intersect_overlap():
    segments = [
        Segment(index=1, start_time=0.0, end_time=1.0, text="a", speaker_id="A"),
        Segment(index=2, start_time=1.0, end_time=2.0, text="b", speaker_id="B"),
        Segment(index=3, start_time=2.5, end_time=3.0, text="c", speaker_id="A"),
        Segment(index=4, start_time=3.0, end_time=4.0, text="d", speaker_id="A"),
    ]
    payload = {
        "intervals": [
            {"start_ms": 900, "end_ms": 1600, "models": ["osd"]},
            {"start_ms": 3200, "end_ms": 3400, "models": ["community1"]},
        ]
    }

    protected, summary = protect_overlapped_segments(segments, payload)

    assert [segment.text for segment in protected] == ["a b", "c", "d"]
    assert protected[0].start_time == 0.0
    assert protected[0].end_time == 2.0
    assert protected[0].speaker_id == "mixed"
    assert protected[0].overlap_protection["merged"] is True
    assert protected[0].overlap_protection["source_segment_indices"] == [1, 2]
    assert protected[0].overlap_protection["overlap_models"] == ["osd"]
    assert protected[1].overlap_protection is None
    assert protected[2].overlap_protection is None
    assert summary["affected_segments"] == 3
    assert summary["merged_runs"] == 1


def test_non_overlap_segment_between_affected_runs_is_not_bridged():
    segments = [
        Segment(index=1, start_time=0.0, end_time=1.0, text="a"),
        Segment(index=2, start_time=1.0, end_time=2.0, text="b"),
        Segment(index=3, start_time=2.0, end_time=3.0, text="c"),
    ]
    payload = {
        "intervals": [
            {"start_ms": 100, "end_ms": 200, "models": ["osd"]},
            {"start_ms": 2100, "end_ms": 2200, "models": ["community1"]},
        ]
    }

    protected, summary = protect_overlapped_segments(segments, payload)

    assert [segment.text for segment in protected] == ["a", "b", "c"]
    assert all(segment.overlap_protection is None for segment in protected)
    assert summary["affected_segments"] == 2
    assert summary["merged_runs"] == 0
