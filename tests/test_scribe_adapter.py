from chalna.scribe_adapter import scribe_response_to_segments


def test_scribe_adapter_segments_by_speaker_pause_and_audio_event():
    response = {
        "language_code": "ko",
        "words": [
            {"type": "word", "text": "안녕하세요", "start": 0.0, "end": 0.5, "speaker_id": "A"},
            {"type": "spacing", "text": " "},
            {"type": "word", "text": "반갑습니다.", "start": 0.55, "end": 1.1, "speaker_id": "A"},
            {"type": "word", "text": "저는", "start": 2.2, "end": 2.5, "speaker_id": "B"},
            {"type": "spacing", "text": " "},
            {"type": "word", "text": "영희입니다.", "start": 2.55, "end": 3.1, "speaker_id": "B"},
            {"type": "audio_event", "text": "laughter", "start": 3.2, "end": 3.6},
        ],
    }

    result = scribe_response_to_segments(response, include_audio_events=True)

    assert [segment.text for segment in result.segments] == [
        "안녕하세요 반갑습니다.",
        "저는 영희입니다.",
        "[laughter]",
    ]
    assert result.segments[0].speaker_id == "A"
    assert result.segments[1].speaker_id == "B"
    assert result.language_code == "ko"
    assert result.words_by_segment_index[1][0]["text"] == "안녕하세요"


def test_scribe_adapter_handles_diarize_false_without_speakers():
    response = {
        "words": [
            {"type": "word", "text": "hello", "start": 0.0, "end": 0.2},
            {"type": "spacing", "text": " "},
            {"type": "word", "text": "world.", "start": 0.25, "end": 0.6},
        ],
    }

    result = scribe_response_to_segments(response)

    assert len(result.segments) == 1
    assert result.segments[0].text == "hello world."
    assert result.segments[0].speaker_id is None
