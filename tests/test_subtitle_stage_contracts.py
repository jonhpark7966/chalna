"""
Stage-level tests for the Scribe v2 subtitle pipeline.
"""

from __future__ import annotations

import json

from chalna.models import Segment
from chalna.pipeline import ChalnaPipeline


def test_align_segments_compatibility_only_fixes_overlaps(tmp_path):
    pipeline = ChalnaPipeline(use_alignment=True, use_llm_refinement=False)

    segments = [
        Segment(1, 0.0, 2.0, "첫 문장입니다.", speaker_id="A"),
        Segment(2, 1.5, 3.0, "두 번째 문장입니다.", speaker_id="A"),
    ]

    aligned = pipeline.align_segments(tmp_path / "audio.wav", segments, verbose=False)

    assert pipeline.get_alignment_log() == []
    assert aligned[0].end_time == 1.75
    assert aligned[1].start_time == 1.75


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
