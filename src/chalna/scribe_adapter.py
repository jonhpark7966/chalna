"""Convert ElevenLabs Scribe responses into Chalna segments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from chalna.models import Segment

SENTENCE_ENDINGS = (".", "?", "!", "。", "？", "！")
PHRASE_ENDINGS = SENTENCE_ENDINGS + (",", ";", ":", "，", "、")


@dataclass
class ScribeAdapterResult:
    segments: list[Segment]
    words_by_segment_index: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    language_code: Optional[str] = None


def _item_text(item: dict[str, Any]) -> str:
    value = item.get("text", item.get("word", ""))
    return str(value) if value is not None else ""


def _item_time(item: dict[str, Any], key: str) -> Optional[float]:
    value = item.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class _SegmentBuilder:
    def __init__(self) -> None:
        self.text_parts: list[str] = []
        self.words: list[dict[str, Any]] = []
        self.start: Optional[float] = None
        self.end: Optional[float] = None
        self.speaker_id: Optional[str] = None

    @property
    def has_words(self) -> bool:
        return bool(self.words)

    @property
    def duration(self) -> float:
        if self.start is None or self.end is None:
            return 0.0
        return max(0.0, self.end - self.start)

    def append_spacing(self, text: str) -> None:
        if self.has_words and text:
            self.text_parts.append(text)

    def append_word(self, item: dict[str, Any]) -> None:
        start = _item_time(item, "start")
        end = _item_time(item, "end")
        if start is None or end is None:
            return

        if self.start is None:
            self.start = start
        self.end = end

        if self.speaker_id is None and item.get("speaker_id") is not None:
            self.speaker_id = str(item["speaker_id"])

        text = _item_text(item)
        self.text_parts.append(text)
        self.words.append(dict(item))

    def flush(self, index: int) -> tuple[Optional[Segment], list[dict[str, Any]]]:
        text = "".join(self.text_parts).strip()
        if not text or self.start is None or self.end is None or self.end <= self.start:
            self.reset()
            return None, []

        segment = Segment(
            index=index,
            start_time=self.start,
            end_time=self.end,
            text=text,
            speaker_id=self.speaker_id,
        )
        words = list(self.words)
        self.reset()
        return segment, words

    def reset(self) -> None:
        self.text_parts = []
        self.words = []
        self.start = None
        self.end = None
        self.speaker_id = None


def scribe_response_to_segments(
    response: dict[str, Any],
    *,
    include_audio_events: bool = True,
    max_segment_duration: float = 5.0,
    pause_threshold: float = 0.8,
) -> ScribeAdapterResult:
    """Convert a raw Scribe response into Chalna subtitle segments."""
    raw_words = response.get("words") or []
    if not isinstance(raw_words, list):
        raw_words = []

    segments: list[Segment] = []
    words_by_segment_index: dict[int, list[dict[str, Any]]] = {}
    builder = _SegmentBuilder()

    def flush_current() -> None:
        segment, words = builder.flush(len(segments) + 1)
        if segment is None:
            return
        segments.append(segment)
        words_by_segment_index[segment.index] = words

    def add_audio_event(item: dict[str, Any]) -> None:
        if not include_audio_events:
            return
        start = _item_time(item, "start")
        end = _item_time(item, "end")
        text = _item_text(item).strip()
        if not text or start is None or end is None or end <= start:
            return
        segment = Segment(
            index=len(segments) + 1,
            start_time=start,
            end_time=end,
            text=f"[{text.strip('[]')}]",
            speaker_id=str(item["speaker_id"]) if item.get("speaker_id") is not None else None,
            confidence=0.8,
        )
        segments.append(segment)
        words_by_segment_index[segment.index] = [dict(item)]

    for item in raw_words:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "word"))

        if item_type == "spacing":
            builder.append_spacing(_item_text(item))
            continue

        if item_type == "audio_event":
            flush_current()
            add_audio_event(item)
            continue

        if item_type != "word":
            continue

        start = _item_time(item, "start")
        end = _item_time(item, "end")
        if start is None or end is None:
            continue

        speaker_id = str(item["speaker_id"]) if item.get("speaker_id") is not None else None
        previous_end = builder.end
        should_flush = False

        if builder.has_words:
            if speaker_id is not None and builder.speaker_id not in (None, speaker_id):
                should_flush = True
            elif previous_end is not None and start - previous_end >= pause_threshold:
                should_flush = True
            elif builder.duration >= max_segment_duration:
                should_flush = True

        if should_flush:
            flush_current()

        builder.append_word(item)

        token_text = _item_text(item).rstrip()
        if token_text.endswith(SENTENCE_ENDINGS) and builder.duration >= 1.0:
            flush_current()
        elif token_text.endswith(PHRASE_ENDINGS) and builder.duration >= max_segment_duration:
            flush_current()
        elif builder.duration >= max_segment_duration + 1.0:
            flush_current()

    flush_current()

    # Re-index after any skipped items.
    reindexed_words: dict[int, list[dict[str, Any]]] = {}
    for new_index, segment in enumerate(segments, start=1):
        old_index = segment.index
        segment.index = new_index
        reindexed_words[new_index] = words_by_segment_index.get(old_index, [])

    language_code = response.get("language_code")
    return ScribeAdapterResult(
        segments=segments,
        words_by_segment_index=reindexed_words,
        language_code=str(language_code) if language_code else None,
    )


fallback_scribe_response_to_segments = scribe_response_to_segments
