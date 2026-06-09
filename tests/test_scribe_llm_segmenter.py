import json

from chalna.models import LlmSegmentationOptions, ScribeOptions
from chalna.scribe_llm_segmenter import LlmScribeSegmenter
from chalna.segment_cache import SegmentPlanCache

SCRIBE_RESPONSE = {
    "language_code": "ko",
    "words": [
        {"type": "word", "text": "안녕하세요", "start": 0.0, "end": 0.5, "speaker_id": "A"},
        {"type": "spacing", "text": " "},
        {"type": "word", "text": "반갑습니다.", "start": 0.6, "end": 1.1, "speaker_id": "A"},
        {"type": "spacing", "text": " "},
        {"type": "word", "text": "다음", "start": 1.2, "end": 1.5, "speaker_id": "A"},
        {"type": "spacing", "text": " "},
        {"type": "word", "text": "문장입니다.", "start": 1.6, "end": 2.1, "speaker_id": "A"},
        {"type": "audio_event", "text": "laughter", "start": 2.2, "end": 2.4},
    ],
}


def test_llm_segmenter_uses_word_ranges_and_caches_plan(tmp_path, monkeypatch):
    calls = []

    def fake_call_codex_cli(prompt, model="gpt-5.5", reasoning_effort="xhigh", timeout=120):
        calls.append({
            "prompt": prompt,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "timeout": timeout,
        })
        return json.dumps(
            [
                {"start_word_index": 0, "end_word_index": 1},
                {"start_word_index": 2, "end_word_index": 3},
            ],
            ensure_ascii=False,
        )

    monkeypatch.setattr("chalna.scribe_llm_segmenter.call_codex_cli", fake_call_codex_cli)
    segmenter = LlmScribeSegmenter(
        cache=SegmentPlanCache(tmp_path / "segment_cache"),
        timeout=30,
    )
    options = LlmSegmentationOptions(
        enabled=True,
        model="gpt-5.5",
        reasoning_effort="xhigh",
        max_words_per_call=20,
    )

    first = segmenter.segment(
        SCRIBE_RESPONSE,
        scribe_cache_key="scribe-key",
        language_code="ko",
        context=None,
        scribe_options=ScribeOptions(diarize=True, tag_audio_events=True),
        segmentation_options=options,
    )
    second = segmenter.segment(
        SCRIBE_RESPONSE,
        scribe_cache_key="scribe-key",
        language_code="ko",
        context=None,
        scribe_options=ScribeOptions(diarize=True, tag_audio_events=True),
        segmentation_options=options,
    )

    assert len(calls) == 1
    assert calls[0]["model"] == "gpt-5.5"
    assert calls[0]["reasoning_effort"] == "xhigh"
    assert [segment.text for segment in first.segments] == [
        "안녕하세요 반갑습니다.",
        "다음 문장입니다.",
        "[laughter]",
    ]
    assert first.segments[0].start_time == 0.0
    assert first.segments[0].end_time == 1.1
    assert first.words_by_segment_index[1][0]["text"] == "안녕하세요"
    assert second.cache_hit is True


def test_llm_segmenter_rejects_non_contiguous_ranges(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "chalna.scribe_llm_segmenter.call_codex_cli",
        lambda *args, **kwargs: json.dumps([{"start_word_index": 0, "end_word_index": 0}]),
    )
    segmenter = LlmScribeSegmenter(cache=SegmentPlanCache(tmp_path / "segment_cache"))

    try:
        segmenter.segment(
            SCRIBE_RESPONSE,
            scribe_cache_key="scribe-key",
            language_code="ko",
            context=None,
            scribe_options=ScribeOptions(),
            segmentation_options=LlmSegmentationOptions(max_words_per_call=20),
        )
    except ValueError as e:
        assert "cover all words" in str(e)
    else:
        raise AssertionError("expected invalid LLM range plan to be rejected")
