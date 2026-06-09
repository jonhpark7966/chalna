"""LLM-based word boundary planner for Scribe responses."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from chalna.exceptions import CodexAPIError, CodexRateLimitError
from chalna.llm_refiner import call_codex_cli
from chalna.models import LlmSegmentationOptions, ScribeOptions, Segment
from chalna.segment_cache import (
    SEGMENTATION_PROMPT_VERSION,
    SegmentPlanCache,
    build_segment_cache_metadata,
)
from chalna.settings import settings


@dataclass
class LlmSegmentationResult:
    """Result of converting Scribe words to Chalna segments using an LLM plan."""

    segments: list[Segment]
    words_by_segment_index: dict[int, list[dict[str, Any]]]
    log: list[dict[str, Any]] = field(default_factory=list)
    cache_hit: bool = False


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


def _extract_json_payload(response: str) -> Any:
    response_clean = response.strip()
    if response_clean.startswith("```"):
        lines = response_clean.splitlines()
        start_idx = 1 if lines and lines[0].startswith("```") else 0
        end_idx = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break
        response_clean = "\n".join(lines[start_idx:end_idx])

    array_start = response_clean.find("[")
    object_start = response_clean.find("{")
    if array_start == -1 and object_start == -1:
        raise ValueError("No JSON payload found in LLM response")

    if array_start != -1 and (object_start == -1 or array_start < object_start):
        start = array_start
        end = response_clean.rfind("]") + 1
    else:
        start = object_start
        end = response_clean.rfind("}") + 1

    if start < 0 or end <= start:
        raise ValueError("Incomplete JSON payload in LLM response")
    return json.loads(response_clean[start:end])


def _speech_words(raw_words: list[Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for raw_index, item in enumerate(raw_words):
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "word")) != "word":
            continue
        start = _item_time(item, "start")
        end = _item_time(item, "end")
        if start is None or end is None or end <= start:
            continue
        words.append({
            "index": len(words),
            "raw_index": raw_index,
            "text": _item_text(item),
            "start": start,
            "end": end,
            "speaker_id": str(item["speaker_id"]) if item.get("speaker_id") is not None else None,
            "type": "word",
            "item": dict(item),
        })
    return words


def _speaker_id_for_range(words: list[dict[str, Any]], start: int, end: int) -> Optional[str]:
    speakers = [
        str(word["speaker_id"])
        for word in words[start:end + 1]
        if word.get("speaker_id") is not None
    ]
    if not speakers:
        return None
    return Counter(speakers).most_common(1)[0][0]


def _text_for_range(
    raw_words: list[Any],
    speech_words: list[dict[str, Any]],
    start: int,
    end: int,
) -> str:
    raw_start = speech_words[start]["raw_index"]
    raw_end = speech_words[end]["raw_index"]
    parts: list[str] = []
    for item in raw_words[raw_start:raw_end + 1]:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "word"))
        if item_type == "spacing":
            if parts:
                parts.append(_item_text(item))
        elif item_type == "word":
            parts.append(_item_text(item))

    text = "".join(parts).strip()
    if text:
        return text
    return " ".join(word["text"] for word in speech_words[start:end + 1]).strip()


def _audio_event_segments(
    raw_words: list[Any],
    include_audio_events: bool,
) -> list[tuple[Segment, list[dict[str, Any]]]]:
    if not include_audio_events:
        return []

    segments: list[tuple[Segment, list[dict[str, Any]]]] = []
    for item in raw_words:
        if not isinstance(item, dict) or str(item.get("type")) != "audio_event":
            continue
        start = _item_time(item, "start")
        end = _item_time(item, "end")
        text = _item_text(item).strip()
        if not text or start is None or end is None or end <= start:
            continue
        segment = Segment(
            index=0,
            start_time=start,
            end_time=end,
            text=f"[{text.strip('[]')}]",
            speaker_id=str(item["speaker_id"]) if item.get("speaker_id") is not None else None,
            confidence=0.8,
        )
        segments.append((segment, [dict(item)]))
    return segments


def _items_from_plan_payload(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("segments", "ranges", "plan"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("LLM response must be a JSON array or contain a segments array")


def _validate_ranges(
    items: list[Any],
    *,
    expected_start: int,
    expected_end: int,
    speech_words: list[dict[str, Any]],
    max_segment_duration: float,
) -> list[dict[str, int]]:
    if expected_start > expected_end:
        return []

    ranges: list[dict[str, int]] = []
    next_index = expected_start
    hard_max_duration = max_segment_duration + 1.0

    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each segment range must be an object")
        try:
            start = int(item["start_word_index"])
            end = int(item["end_word_index"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError("Each segment needs start_word_index and end_word_index") from e

        if start != next_index:
            raise ValueError(f"Word ranges must be contiguous: expected {next_index}, got {start}")
        if start > end:
            raise ValueError(f"Invalid descending word range: {start}>{end}")
        if end > expected_end:
            raise ValueError(f"Word range end exceeds chunk: {end}>{expected_end}")

        duration = speech_words[end]["end"] - speech_words[start]["start"]
        if duration > hard_max_duration and end > start:
            raise ValueError(
                f"Word range {start}-{end} is too long: {duration:.2f}s "
                f"(max {hard_max_duration:.2f}s)"
            )

        speakers = {
            word["speaker_id"]
            for word in speech_words[start:end + 1]
            if word.get("speaker_id") is not None
        }
        if len(speakers) > 1:
            raise ValueError(f"Word range {start}-{end} mixes speakers: {sorted(speakers)}")

        ranges.append({"start_word_index": start, "end_word_index": end})
        next_index = end + 1

    if next_index != expected_end + 1:
        raise ValueError(f"Word ranges did not cover all words through {expected_end}")

    return ranges


def _ranges_to_segments(
    *,
    raw_words: list[Any],
    speech_words: list[dict[str, Any]],
    ranges: list[dict[str, int]],
    include_audio_events: bool,
) -> tuple[list[Segment], dict[int, list[dict[str, Any]]]]:
    paired: list[tuple[Segment, list[dict[str, Any]]]] = []

    for item in ranges:
        start_idx = item["start_word_index"]
        end_idx = item["end_word_index"]
        start_word = speech_words[start_idx]
        end_word = speech_words[end_idx]
        text = _text_for_range(raw_words, speech_words, start_idx, end_idx)
        if not text:
            continue

        segment = Segment(
            index=0,
            start_time=float(start_word["start"]),
            end_time=float(end_word["end"]),
            text=text,
            speaker_id=_speaker_id_for_range(speech_words, start_idx, end_idx),
        )
        words = [dict(word["item"]) for word in speech_words[start_idx:end_idx + 1]]
        paired.append((segment, words))

    paired.extend(_audio_event_segments(raw_words, include_audio_events))
    paired.sort(key=lambda pair: (pair[0].start_time, pair[0].end_time))

    segments: list[Segment] = []
    words_by_segment_index: dict[int, list[dict[str, Any]]] = {}
    for index, (segment, words) in enumerate(paired, start=1):
        segment.index = index
        segments.append(segment)
        words_by_segment_index[index] = words

    return segments, words_by_segment_index


class LlmScribeSegmenter:
    """Plan subtitle boundaries with an LLM while preserving Scribe timestamps."""

    def __init__(
        self,
        *,
        cache: Optional[SegmentPlanCache] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.cache = cache or SegmentPlanCache(settings.llm_segmentation_cache_dir)
        self.model = model or settings.llm_segmentation_model
        self.reasoning_effort = reasoning_effort or settings.llm_segmentation_reasoning_effort
        self.timeout = timeout if timeout is not None else settings.llm_segmentation_timeout

    def segment(
        self,
        response: dict[str, Any],
        *,
        scribe_cache_key: str,
        language_code: Optional[str],
        context: Optional[str],
        scribe_options: ScribeOptions,
        segmentation_options: LlmSegmentationOptions,
        include_audio_events: bool = True,
    ) -> LlmSegmentationResult:
        raw_words = response.get("words") or []
        if not isinstance(raw_words, list):
            raise ValueError("Scribe response words must be a list")

        speech_words = _speech_words(raw_words)
        if not speech_words:
            raise ValueError("Scribe response does not contain timed speech words")

        effective_options = LlmSegmentationOptions(
            enabled=segmentation_options.enabled,
            model=segmentation_options.model or self.model,
            reasoning_effort=segmentation_options.reasoning_effort or self.reasoning_effort,
            max_segment_duration=segmentation_options.max_segment_duration,
            max_words_per_call=segmentation_options.max_words_per_call,
        )
        metadata = build_segment_cache_metadata(
            scribe_cache_key=scribe_cache_key,
            language_code=language_code,
            scribe_options=scribe_options,
            segmentation_options=effective_options,
        )

        log: list[dict[str, Any]] = []
        cache_hit = False
        plan = self.cache.get(metadata)
        if plan is not None:
            try:
                ranges = _validate_ranges(
                    plan.get("ranges", []),
                    expected_start=0,
                    expected_end=len(speech_words) - 1,
                    speech_words=speech_words,
                    max_segment_duration=effective_options.max_segment_duration,
                )
                cache_hit = True
                log.extend(plan.get("log", []))
                log.append({"status": "cache_hit", "source": "llm_segmentation"})
            except ValueError as e:
                log.append({"status": "cache_invalid", "error": str(e)})
                ranges = self._plan_ranges(
                    speech_words=speech_words,
                    language_code=language_code,
                    context=context,
                    options=effective_options,
                    log=log,
                )
                self.cache.put(metadata, {"ranges": ranges, "log": log})
        else:
            ranges = self._plan_ranges(
                speech_words=speech_words,
                language_code=language_code,
                context=context,
                options=effective_options,
                log=log,
            )
            self.cache.put(metadata, {"ranges": ranges, "log": log})

        segments, words_by_segment_index = _ranges_to_segments(
            raw_words=raw_words,
            speech_words=speech_words,
            ranges=ranges,
            include_audio_events=include_audio_events,
        )
        if not segments:
            raise ValueError("LLM segmentation produced no segments")

        return LlmSegmentationResult(
            segments=segments,
            words_by_segment_index=words_by_segment_index,
            log=log,
            cache_hit=cache_hit,
        )

    def _plan_ranges(
        self,
        *,
        speech_words: list[dict[str, Any]],
        language_code: Optional[str],
        context: Optional[str],
        options: LlmSegmentationOptions,
        log: list[dict[str, Any]],
    ) -> list[dict[str, int]]:
        ranges: list[dict[str, int]] = []
        max_words = options.max_words_per_call
        for chunk_start in range(0, len(speech_words), max_words):
            chunk_end = min(chunk_start + max_words, len(speech_words)) - 1
            prompt = self._build_prompt(
                speech_words=speech_words,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                language_code=language_code,
                context=context,
                options=options,
            )
            try:
                response = call_codex_cli(
                    prompt,
                    model=options.model,
                    reasoning_effort=options.reasoning_effort,
                    timeout=self.timeout,
                )
            except (CodexAPIError, CodexRateLimitError):
                raise

            payload = _extract_json_payload(response)
            items = _items_from_plan_payload(payload)
            chunk_ranges = _validate_ranges(
                items,
                expected_start=chunk_start,
                expected_end=chunk_end,
                speech_words=speech_words,
                max_segment_duration=options.max_segment_duration,
            )
            ranges.extend(chunk_ranges)
            log.append({
                "status": "planned",
                "chunk_start_word_index": chunk_start,
                "chunk_end_word_index": chunk_end,
                "segment_count": len(chunk_ranges),
                "model": options.model,
                "reasoning_effort": options.reasoning_effort,
                "prompt_version": SEGMENTATION_PROMPT_VERSION,
            })
        return ranges

    def _build_prompt(
        self,
        *,
        speech_words: list[dict[str, Any]],
        chunk_start: int,
        chunk_end: int,
        language_code: Optional[str],
        context: Optional[str],
        options: LlmSegmentationOptions,
    ) -> str:
        words_payload = [
            {
                "index": word["index"],
                "text": word["text"],
                "start": round(float(word["start"]), 3),
                "end": round(float(word["end"]), 3),
                "speaker_id": word["speaker_id"],
                "type": word["type"],
            }
            for word in speech_words[chunk_start:chunk_end + 1]
        ]
        language_line = (
            f"Language hint: {language_code}"
            if language_code
            else "Language hint: auto"
        )
        context_block = f"\nContext:\n{context}\n" if context else ""
        words_json = json.dumps(words_payload, ensure_ascii=False, indent=2)

        return f"""You are a subtitle segmentation planner.

Task: Group the provided Scribe word tokens into subtitle segments.
You must only decide word index ranges. Do not rewrite words. Do not invent timestamps.

Rules:
- Return JSON only.
- Output a JSON array of objects with start_word_index and end_word_index.
- Cover every input word index exactly once, in ascending order.
- Do not skip, duplicate, overlap, or reorder words.
- Prefer natural subtitle boundaries by sentence meaning, clauses, pauses, and speaker changes.
- Never mix different speaker_id values inside one segment when speaker_id is present.
- Target each segment to be 2-4 seconds.
- Hard limit is {options.max_segment_duration + 1.0:.1f} seconds unless a single word is longer.
- Keep segments readable as subtitles; avoid one-word segments unless needed.

{language_line}
Chunk word index range: {chunk_start}..{chunk_end}
{context_block}
Words:
{words_json}

JSON response:
"""
