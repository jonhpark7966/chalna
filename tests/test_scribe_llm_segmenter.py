import json

from chalna.models import LlmSegmentationOptions, ScribeOptions
from chalna.scribe_llm_segmenter import LlmScribeSegmenter
from chalna.segment_cache import SegmentPlanCache, build_segment_cache_metadata

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


def _scribe_response_from_words(words):
    raw_words = []
    current_time = 0.0
    for index, word in enumerate(words):
        if len(word) == 3:
            text, speaker_id, pause_after = word
        else:
            text, speaker_id = word
            pause_after = 0.1
        raw_words.append({
            "type": "word",
            "text": text,
            "start": round(current_time, 3),
            "end": round(current_time + 0.2, 3),
            "speaker_id": speaker_id,
        })
        current_time += 0.2 + pause_after
        if index < len(words) - 1:
            raw_words.append({"type": "spacing", "text": " "})
    return {"language_code": "ko", "words": raw_words}


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
    prompt = calls[0]["prompt"]
    assert "Word table format: index|text|speaker_id|next_gap" in prompt
    assert "index|text|speaker_id|next_gap" in prompt
    assert "0|안녕하세요|A|0.100" in prompt
    assert "3|문장입니다.|A|-" in prompt
    assert "Sentence-ending punctuation is a mandatory boundary" in prompt
    assert "Do not merge across sentence-ending" in prompt
    assert "Commas, semicolons, colons" in prompt
    assert '"start":' not in prompt
    assert calls[0]["model"] == "gpt-5.5"
    assert calls[0]["reasoning_effort"] == "xhigh"
    assert first.log[0]["mode"] == "compact_full_words"
    assert [segment.text for segment in first.segments] == [
        "안녕하세요 반갑습니다.",
        "다음 문장입니다.",
        "[laughter]",
    ]
    assert first.segments[0].start_time == 0.0
    assert first.segments[0].end_time == 1.1
    assert first.words_by_segment_index[1][0]["text"] == "안녕하세요"
    assert second.cache_hit is True


def test_llm_segmenter_bypasses_cached_plan_when_requested(tmp_path, monkeypatch):
    calls = []

    def fake_call_codex_cli(prompt, **kwargs):
        calls.append(prompt)
        return json.dumps(
            [
                {"start_word_index": 0, "end_word_index": 1},
                {"start_word_index": 2, "end_word_index": 3},
            ],
            ensure_ascii=False,
        )

    monkeypatch.setattr("chalna.scribe_llm_segmenter.call_codex_cli", fake_call_codex_cli)
    segmenter = LlmScribeSegmenter(cache=SegmentPlanCache(tmp_path / "segment_cache"))

    cached_options = LlmSegmentationOptions(max_words_per_call=20)
    bypass_options = LlmSegmentationOptions(max_words_per_call=20, bypass_cache=True)

    first = segmenter.segment(
        SCRIBE_RESPONSE,
        scribe_cache_key="scribe-key",
        language_code="ko",
        context=None,
        scribe_options=ScribeOptions(),
        segmentation_options=cached_options,
    )
    second = segmenter.segment(
        SCRIBE_RESPONSE,
        scribe_cache_key="scribe-key",
        language_code="ko",
        context=None,
        scribe_options=ScribeOptions(),
        segmentation_options=bypass_options,
    )

    assert len(calls) == 2
    assert first.cache_hit is False
    assert second.cache_hit is False
    assert second.log[0] == {"status": "cache_bypassed", "source": "llm_segmentation"}
    assert second.log[1]["mode"] == "compact_full_words"


def test_llm_segmenter_repairs_mixed_speaker_compact_ranges(tmp_path, monkeypatch):
    calls = []
    response = _scribe_response_from_words([
        ("응.", "speaker_1"),
        ("그렇죠.", "speaker_0"),
        ("다음", "speaker_0"),
        ("문장입니다.", "speaker_0"),
    ])

    def fake_call_codex_cli(prompt, **kwargs):
        calls.append(prompt)
        return json.dumps(
            [
                {"start_word_index": 0, "end_word_index": 1},
                {"start_word_index": 2, "end_word_index": 3},
            ],
            ensure_ascii=False,
        )

    monkeypatch.setattr("chalna.scribe_llm_segmenter.call_codex_cli", fake_call_codex_cli)
    segmenter = LlmScribeSegmenter(cache=SegmentPlanCache(tmp_path / "segment_cache"))

    result = segmenter.segment(
        response,
        scribe_cache_key="scribe-key",
        language_code="ko",
        context=None,
        scribe_options=ScribeOptions(),
        segmentation_options=LlmSegmentationOptions(max_words_per_call=20),
    )

    assert len(calls) == 1
    assert not any(item.get("status") == "fallback_to_legacy_chunks" for item in result.log)
    assert result.log[0]["mode"] == "compact_full_words"
    assert result.log[0]["range_repair_count"] == 1
    assert result.log[1]["output"]["range_repairs"] == [
        {
            "reason": "mixed_speaker_range",
            "original": {"start_word_index": 0, "end_word_index": 1},
            "replacements": [
                {"start_word_index": 0, "end_word_index": 0},
                {"start_word_index": 1, "end_word_index": 1},
            ],
        }
    ]
    assert [segment.text for segment in result.segments] == [
        "응.",
        "그렇죠.",
        "다음 문장입니다.",
    ]


def test_llm_segmenter_hard_splits_sentence_endings(tmp_path, monkeypatch):
    response = _scribe_response_from_words([
        ("첫", "A"),
        ("문장입니다.", "A"),
        ("다음", "A"),
        ("질문인가요?", "A"),
        ("마지막", "A"),
    ])

    monkeypatch.setattr(
        "chalna.scribe_llm_segmenter.call_codex_cli",
        lambda *args, **kwargs: json.dumps([{"start_word_index": 0, "end_word_index": 4}]),
    )
    segmenter = LlmScribeSegmenter(cache=SegmentPlanCache(tmp_path / "segment_cache"))

    result = segmenter.segment(
        response,
        scribe_cache_key="scribe-key",
        language_code="ko",
        context=None,
        scribe_options=ScribeOptions(),
        segmentation_options=LlmSegmentationOptions(max_words_per_call=20),
        include_audio_events=False,
    )

    assert [segment.text for segment in result.segments] == [
        "첫 문장입니다.",
        "다음 질문인가요?",
        "마지막",
    ]
    assert result.log[0]["range_repair_count"] == 1
    assert result.log[1]["output"]["range_repairs"] == [
        {
            "reason": "sentence_ending_punctuation",
            "original": {"start_word_index": 0, "end_word_index": 4},
            "replacements": [
                {"start_word_index": 0, "end_word_index": 1},
                {"start_word_index": 2, "end_word_index": 3},
                {"start_word_index": 4, "end_word_index": 4},
            ],
        }
    ]


def test_llm_segmenter_does_not_hard_split_commas(tmp_path, monkeypatch):
    response = _scribe_response_from_words([
        ("네,", "A"),
        ("좋습니다.", "A"),
    ])

    monkeypatch.setattr(
        "chalna.scribe_llm_segmenter.call_codex_cli",
        lambda *args, **kwargs: json.dumps([{"start_word_index": 0, "end_word_index": 1}]),
    )
    segmenter = LlmScribeSegmenter(cache=SegmentPlanCache(tmp_path / "segment_cache"))

    result = segmenter.segment(
        response,
        scribe_cache_key="scribe-key",
        language_code="ko",
        context=None,
        scribe_options=ScribeOptions(),
        segmentation_options=LlmSegmentationOptions(max_words_per_call=20),
        include_audio_events=False,
    )

    assert [segment.text for segment in result.segments] == ["네, 좋습니다."]
    assert result.log[0]["range_repair_count"] == 0
    assert "range_repairs" not in result.log[1]["output"]


def test_llm_segmenter_repairs_speaker_and_sentence_boundaries(tmp_path, monkeypatch):
    response = _scribe_response_from_words([
        ("첫", "A"),
        ("문장입니다.", "A"),
        ("응.", "B"),
        ("다음", "B"),
        ("말입니다.", "B"),
    ])

    monkeypatch.setattr(
        "chalna.scribe_llm_segmenter.call_codex_cli",
        lambda *args, **kwargs: json.dumps([{"start_word_index": 0, "end_word_index": 4}]),
    )
    segmenter = LlmScribeSegmenter(cache=SegmentPlanCache(tmp_path / "segment_cache"))

    result = segmenter.segment(
        response,
        scribe_cache_key="scribe-key",
        language_code="ko",
        context=None,
        scribe_options=ScribeOptions(),
        segmentation_options=LlmSegmentationOptions(max_words_per_call=20),
        include_audio_events=False,
    )

    assert [segment.text for segment in result.segments] == [
        "첫 문장입니다.",
        "응.",
        "다음 말입니다.",
    ]
    assert result.log[0]["range_repair_count"] == 2
    assert result.log[1]["output"]["range_repairs"] == [
        {
            "reason": "mixed_speaker_range",
            "original": {"start_word_index": 0, "end_word_index": 4},
            "replacements": [
                {"start_word_index": 0, "end_word_index": 1},
                {"start_word_index": 2, "end_word_index": 4},
            ],
        },
        {
            "reason": "sentence_ending_punctuation",
            "original": {"start_word_index": 2, "end_word_index": 4},
            "replacements": [
                {"start_word_index": 2, "end_word_index": 2},
                {"start_word_index": 3, "end_word_index": 4},
            ],
        },
    ]


def test_llm_segmenter_applies_sentence_split_to_cached_broad_ranges(tmp_path, monkeypatch):
    cache = SegmentPlanCache(tmp_path / "segment_cache")
    scribe_options = ScribeOptions()
    segmentation_options = LlmSegmentationOptions(max_words_per_call=20)
    metadata = build_segment_cache_metadata(
        scribe_cache_key="scribe-key",
        language_code="ko",
        scribe_options=scribe_options,
        segmentation_options=segmentation_options,
    )
    cache.put(metadata, {
        "ranges": [{"start_word_index": 0, "end_word_index": 3}],
        "log": [{"status": "planned", "mode": "compact_full_words"}],
    })
    monkeypatch.setattr(
        "chalna.scribe_llm_segmenter.call_codex_cli",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache should be used")),
    )
    segmenter = LlmScribeSegmenter(cache=cache)

    result = segmenter.segment(
        SCRIBE_RESPONSE,
        scribe_cache_key="scribe-key",
        language_code="ko",
        context=None,
        scribe_options=scribe_options,
        segmentation_options=segmentation_options,
        include_audio_events=False,
    )

    assert result.cache_hit is True
    assert [segment.text for segment in result.segments] == [
        "안녕하세요 반갑습니다.",
        "다음 문장입니다.",
    ]


def test_llm_segmenter_escapes_compact_table_cells(tmp_path, monkeypatch):
    calls = []
    response = {
        "language_code": "ko",
        "words": [
            {
                "type": "word",
                "text": "파|이프\n문장",
                "start": 0.0,
                "end": 0.5,
            },
            {"type": "spacing", "text": " "},
            {"type": "word", "text": "끝", "start": 1.0, "end": 1.5},
        ],
    }

    def fake_call_codex_cli(prompt, **kwargs):
        calls.append(prompt)
        return json.dumps([{"start_word_index": 0, "end_word_index": 1}], ensure_ascii=False)

    monkeypatch.setattr("chalna.scribe_llm_segmenter.call_codex_cli", fake_call_codex_cli)
    segmenter = LlmScribeSegmenter(cache=SegmentPlanCache(tmp_path / "segment_cache"))

    segmenter.segment(
        response,
        scribe_cache_key="scribe-key",
        language_code="ko",
        context=None,
        scribe_options=ScribeOptions(),
        segmentation_options=LlmSegmentationOptions(max_words_per_call=20),
    )

    assert "0|파／이프 문장|-|0.500" in calls[0]
    assert "1|끝|-|-" in calls[0]


def test_llm_segmenter_falls_back_to_legacy_json_chunks(tmp_path, monkeypatch):
    calls = []

    def fake_call_codex_cli(prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps([{"start_word_index": 0, "end_word_index": 0}], ensure_ascii=False)
        return json.dumps([{"start_word_index": 0, "end_word_index": 3}], ensure_ascii=False)

    monkeypatch.setattr("chalna.scribe_llm_segmenter.call_codex_cli", fake_call_codex_cli)
    segmenter = LlmScribeSegmenter(cache=SegmentPlanCache(tmp_path / "segment_cache"))

    result = segmenter.segment(
        SCRIBE_RESPONSE,
        scribe_cache_key="scribe-key",
        language_code="ko",
        context=None,
        scribe_options=ScribeOptions(),
        segmentation_options=LlmSegmentationOptions(max_words_per_call=20),
    )

    assert len(calls) == 2
    assert "index|text|speaker_id|next_gap" in calls[0]
    assert '"start": 0.0' in calls[1]
    assert result.log[0]["status"] == "llm_io"
    assert (
        result.log[0]["output"]["validation_error"]
        == "Word ranges did not cover all words through 3"
    )
    assert result.log[1]["status"] == "fallback_to_legacy_chunks"
    assert result.log[2]["mode"] == "legacy_json_word_chunks"
    assert [segment.text for segment in result.segments] == [
        "안녕하세요 반갑습니다.",
        "다음 문장입니다.",
        "[laughter]",
    ]


def test_legacy_chunking_uses_safe_boundary_to_avoid_continuation_split(tmp_path, monkeypatch):
    calls = []
    words = [(f"단어{i}", "A") for i in range(15)]
    words.extend([
        ("문장입니다.", "A"),
        ("이제", "A"),
        ("어떤", "A"),
        ("부작용이", "A"),
        ("있을", "A"),
        ("수", "A"),
        ("있는지.", "A"),
        ("다음", "A"),
        ("문장입니다.", "A"),
        ("끝입니다.", "A"),
    ])
    response = _scribe_response_from_words(words)

    def fake_call_codex_cli(prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps([{"start_word_index": 0, "end_word_index": 0}], ensure_ascii=False)

        range_line = next(
            line for line in prompt.splitlines()
            if line.startswith("Chunk word index range:")
        )
        start_end = range_line.removeprefix("Chunk word index range: ").split("..")
        start_word = int(start_end[0])
        end_word = int(start_end[1])
        return json.dumps(
            [{"start_word_index": start_word, "end_word_index": end_word}],
            ensure_ascii=False,
        )

    monkeypatch.setattr("chalna.scribe_llm_segmenter.call_codex_cli", fake_call_codex_cli)
    segmenter = LlmScribeSegmenter(cache=SegmentPlanCache(tmp_path / "segment_cache"))

    result = segmenter.segment(
        response,
        scribe_cache_key="scribe-key",
        language_code="ko",
        context=None,
        scribe_options=ScribeOptions(),
        segmentation_options=LlmSegmentationOptions(max_words_per_call=20),
        include_audio_events=False,
    )

    assert len(calls) == 3
    assert "Chunk word index range: 0..15" in calls[1]
    assert "Chunk word index range: 16..24" in calls[2]
    segment_texts = [segment.text for segment in result.segments]
    assert not any(text.endswith("어떤 부작용이 있을") for text in segment_texts)
    assert any("어떤 부작용이 있을 수 있는지." in text for text in segment_texts)


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
