"""LLM-based word boundary planner for Scribe responses."""

from __future__ import annotations

import hashlib
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


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _llm_io_log_entry(
    *,
    mode: str,
    prompt: str,
    model: str,
    reasoning_effort: str,
    cache_hit: bool,
    response: str | None = None,
    ranges: list[dict[str, int]] | None = None,
    prompt_version: str = SEGMENTATION_PROMPT_VERSION,
    extra_input: dict[str, Any] | None = None,
    extra_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "status": "llm_io",
        "stage": "segmentation",
        "provider": "codex",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "mode": mode,
        "prompt_version": prompt_version,
        "cache_hit": cache_hit,
        "input": {
            "prompt": prompt,
            "prompt_sha256": _hash_text(prompt),
            **(extra_input or {}),
        },
    }
    output: dict[str, Any] = {**(extra_output or {})}
    if response is not None:
        output["response"] = response
        output["response_sha256"] = _hash_text(response)
    if ranges is not None:
        output["ranges"] = ranges
        output["range_count"] = len(ranges)
    if output:
        entry["output"] = output
    return entry


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


def _compact_table_cell(value: Any, *, fallback: str = "-") -> str:
    if value is None:
        return fallback
    text = " ".join(str(value).replace("\r", " ").replace("\n", " ").split())
    text = text.replace("|", "／")
    return text if text else fallback


def _compact_next_gap(speech_words: list[dict[str, Any]], index: int) -> str:
    if index >= len(speech_words) - 1:
        return "-"
    gap = float(speech_words[index + 1]["start"]) - float(speech_words[index]["end"])
    if abs(gap) < 0.0005:
        gap = 0.0
    return f"{gap:.3f}"


def _compact_word_table(
    speech_words: list[dict[str, Any]],
    *,
    start: int,
    end: int,
) -> str:
    lines = ["index|text|speaker_id|next_gap"]
    for index in range(start, end + 1):
        word = speech_words[index]
        lines.append(
            "|".join((
                str(word["index"]),
                _compact_table_cell(word.get("text"), fallback=""),
                _compact_table_cell(word.get("speaker_id")),
                _compact_next_gap(speech_words, index),
            ))
        )
    return "\n".join(lines)


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


def _is_recoverable_full_call_error(exc: Exception) -> bool:
    if isinstance(exc, CodexRateLimitError):
        return False

    message = str(exc).lower()
    nonrecoverable_markers = (
        "rate limit",
        "quota",
        "auth",
        "unauthorized",
        "permission",
        "forbidden",
        "api key",
        "login",
        "not found",
    )
    if any(marker in message for marker in nonrecoverable_markers):
        return False

    if isinstance(exc, ValueError):
        return True

    recoverable_markers = (
        "timeout",
        "context",
        "token",
        "too large",
        "maximum",
        "json",
        "parse",
        "incomplete",
    )
    return isinstance(exc, CodexAPIError) and any(
        marker in message for marker in recoverable_markers
    )


_BOUNDARY_PUNCTUATION = tuple(".?!,;:。？！…，、；：")
_SENTENCE_ENDING_PUNCTUATION = tuple(".?!。？！…")
_TRAILING_PUNCTUATION = "\"'”’)]}.,?!;:。？！…，、；："
_CONTINUATION_START_WORDS = {
    "수",
    "것",
    "거",
    "게",
    "정도",
    "때",
    "중",
    "지",
}
_CONTINUATION_PREV_SUFFIXES = (
    "할",
    "될",
    "있을",
    "없을",
    "하는",
    "되는",
    "있는",
    "없는",
    "같은",
    "라는",
    "이라는",
)
_LEGACY_CHUNK_BOUNDARY_LOOKBACK_WORDS = 40
_LEGACY_CHUNK_MIN_LOOKBACK_WORDS = 10
_LEGACY_CHUNK_PAUSE_SECONDS = 0.3


def _boundary_token(text: Any) -> str:
    return str(text or "").strip()


def _token_without_trailing_punctuation(text: Any) -> str:
    return _boundary_token(text).strip(_TRAILING_PUNCTUATION)


def _ends_with_boundary_punctuation(text: Any) -> bool:
    token = _boundary_token(text).rstrip("\"'”’)]}")
    return bool(token) and token.endswith(_BOUNDARY_PUNCTUATION)


def _ends_with_sentence_ending_punctuation(text: Any) -> bool:
    token = _boundary_token(text).rstrip("\"'”’)]}")
    return bool(token) and token.endswith(_SENTENCE_ENDING_PUNCTUATION)


def _is_poor_legacy_chunk_boundary(
    current_word: dict[str, Any],
    next_word: dict[str, Any],
) -> bool:
    current_text = _token_without_trailing_punctuation(current_word.get("text"))
    next_text = _token_without_trailing_punctuation(next_word.get("text"))
    if not current_text or not next_text:
        return False
    if _ends_with_boundary_punctuation(current_word.get("text")):
        return False
    if next_text in _CONTINUATION_START_WORDS:
        return True
    return current_text.endswith(_CONTINUATION_PREV_SUFFIXES)


def _gap_after_word(speech_words: list[dict[str, Any]], index: int) -> float:
    if index >= len(speech_words) - 1:
        return 0.0
    return float(speech_words[index + 1]["start"]) - float(speech_words[index]["end"])


def _is_safe_legacy_chunk_boundary(
    speech_words: list[dict[str, Any]],
    index: int,
) -> bool:
    if index >= len(speech_words) - 1:
        return True

    current_word = speech_words[index]
    next_word = speech_words[index + 1]
    if _is_poor_legacy_chunk_boundary(current_word, next_word):
        return False

    current_speaker = current_word.get("speaker_id")
    next_speaker = next_word.get("speaker_id")
    if current_speaker is not None and next_speaker is not None and current_speaker != next_speaker:
        return True
    if _ends_with_boundary_punctuation(current_word.get("text")):
        return True
    return _gap_after_word(speech_words, index) >= _LEGACY_CHUNK_PAUSE_SECONDS


def _choose_legacy_chunk_end(
    speech_words: list[dict[str, Any]],
    *,
    chunk_start: int,
    hard_end: int,
    max_words: int,
) -> int:
    if hard_end >= len(speech_words) - 1:
        return hard_end

    lookback = min(
        _LEGACY_CHUNK_BOUNDARY_LOOKBACK_WORDS,
        max(_LEGACY_CHUNK_MIN_LOOKBACK_WORDS, max_words // 2),
    )
    min_end = max(chunk_start, hard_end - lookback)
    for index in range(hard_end, min_end - 1, -1):
        if _is_safe_legacy_chunk_boundary(speech_words, index):
            return index
    return hard_end


def _legacy_chunk_ranges(
    speech_words: list[dict[str, Any]],
    *,
    max_words: int,
) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    chunk_start = 0
    last_index = len(speech_words) - 1
    while chunk_start <= last_index:
        hard_end = min(chunk_start + max_words, len(speech_words)) - 1
        chunk_end = _choose_legacy_chunk_end(
            speech_words,
            chunk_start=chunk_start,
            hard_end=hard_end,
            max_words=max_words,
        )
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + 1
    return chunks


def _split_range_on_speaker_changes(
    speech_words: list[dict[str, Any]],
    *,
    start: int,
    end: int,
) -> list[dict[str, int]]:
    splits: list[dict[str, int]] = []
    current_start = start
    current_speaker: str | None = None

    for index in range(start, end + 1):
        speaker_id = speech_words[index].get("speaker_id")
        speaker = str(speaker_id) if speaker_id is not None else None
        if current_speaker is None:
            current_speaker = speaker
            continue
        if speaker is not None and speaker != current_speaker:
            splits.append({
                "start_word_index": current_start,
                "end_word_index": index - 1,
            })
            current_start = index
            current_speaker = speaker

    splits.append({
        "start_word_index": current_start,
        "end_word_index": end,
    })
    return splits


def _split_range_on_sentence_endings(
    speech_words: list[dict[str, Any]],
    *,
    start: int,
    end: int,
) -> list[dict[str, int]]:
    splits: list[dict[str, int]] = []
    current_start = start

    for index in range(start, end):
        if not _ends_with_sentence_ending_punctuation(speech_words[index].get("text")):
            continue
        splits.append({
            "start_word_index": current_start,
            "end_word_index": index,
        })
        current_start = index + 1

    splits.append({
        "start_word_index": current_start,
        "end_word_index": end,
    })
    return splits


def _validate_ranges(
    items: list[Any],
    *,
    expected_start: int,
    expected_end: int,
    speech_words: list[dict[str, Any]],
    repair_log: list[dict[str, Any]] | None = None,
) -> list[dict[str, int]]:
    if expected_start > expected_end:
        return []

    ranges: list[dict[str, int]] = []
    next_index = expected_start

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

        split_ranges = _split_range_on_speaker_changes(
            speech_words,
            start=start,
            end=end,
        )
        if len(split_ranges) > 1 and repair_log is not None:
            repair_log.append({
                "reason": "mixed_speaker_range",
                "original": {
                    "start_word_index": start,
                    "end_word_index": end,
                },
                "replacements": split_ranges,
            })

        repaired_ranges: list[dict[str, int]] = []
        for split_range in split_ranges:
            sentence_ranges = _split_range_on_sentence_endings(
                speech_words,
                start=split_range["start_word_index"],
                end=split_range["end_word_index"],
            )
            if len(sentence_ranges) > 1 and repair_log is not None:
                repair_log.append({
                    "reason": "sentence_ending_punctuation",
                    "original": split_range,
                    "replacements": sentence_ranges,
                })
            repaired_ranges.extend(sentence_ranges)

        ranges.extend(repaired_ranges)
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
            bypass_cache=segmentation_options.bypass_cache,
            boundary_rule=segmentation_options.boundary_rule,
        )
        metadata = build_segment_cache_metadata(
            scribe_cache_key=scribe_cache_key,
            language_code=language_code,
            scribe_options=scribe_options,
            segmentation_options=effective_options,
        )

        log: list[dict[str, Any]] = []
        cache_hit = False
        plan = None if effective_options.bypass_cache else self.cache.get(metadata)
        if effective_options.bypass_cache:
            log.append({"status": "cache_bypassed", "source": "llm_segmentation"})
        if plan is not None:
            try:
                ranges = _validate_ranges(
                    plan.get("ranges", []),
                    expected_start=0,
                    expected_end=len(speech_words) - 1,
                    speech_words=speech_words,
                )
                cache_hit = True
                cached_log = plan.get("log", [])
                log.extend(cached_log)
                log.append({"status": "cache_hit", "source": "llm_segmentation"})
                cached_log_entry_count = (
                    len(cached_log) if isinstance(cached_log, list) else 0
                )
                cache_prompt = self._build_prompt(
                    speech_words=speech_words,
                    chunk_start=0,
                    chunk_end=len(speech_words) - 1,
                    language_code=language_code,
                    context=context,
                    options=effective_options,
                    compact_table=True,
                )
                log.append(_llm_io_log_entry(
                    mode="cache_hit",
                    prompt=cache_prompt,
                    model=effective_options.model,
                    reasoning_effort=effective_options.reasoning_effort,
                    cache_hit=True,
                    ranges=ranges,
                    extra_input={
                        "cache_metadata": metadata,
                        "reconstructed_prompt": True,
                    },
                    extra_output={
                        "cached_log_entry_count": cached_log_entry_count,
                    },
                ))
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
        try:
            return self._plan_ranges_compact_full(
                speech_words=speech_words,
                language_code=language_code,
                context=context,
                options=options,
                log=log,
            )
        except (CodexAPIError, ValueError) as exc:
            if not _is_recoverable_full_call_error(exc):
                raise
            log.append({
                "status": "fallback_to_legacy_chunks",
                "source_mode": "compact_full_words",
                "fallback_mode": "legacy_json_word_chunks",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "prompt_version": SEGMENTATION_PROMPT_VERSION,
            })
            return self._plan_ranges_legacy_chunks(
                speech_words=speech_words,
                language_code=language_code,
                context=context,
                options=options,
                log=log,
            )

    def _plan_ranges_compact_full(
        self,
        *,
        speech_words: list[dict[str, Any]],
        language_code: Optional[str],
        context: Optional[str],
        options: LlmSegmentationOptions,
        log: list[dict[str, Any]],
    ) -> list[dict[str, int]]:
        chunk_start = 0
        chunk_end = len(speech_words) - 1
        prompt = self._build_prompt(
            speech_words=speech_words,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            language_code=language_code,
            context=context,
            options=options,
            compact_table=True,
        )
        response = call_codex_cli(
            prompt,
            model=options.model,
            reasoning_effort=options.reasoning_effort,
            timeout=self.timeout,
        )
        range_repairs: list[dict[str, Any]] = []
        try:
            payload = _extract_json_payload(response)
            items = _items_from_plan_payload(payload)
            ranges = _validate_ranges(
                items,
                expected_start=chunk_start,
                expected_end=chunk_end,
                speech_words=speech_words,
                repair_log=range_repairs,
            )
        except ValueError as exc:
            log.append(_llm_io_log_entry(
                mode="compact_full_words",
                prompt=prompt,
                model=options.model,
                reasoning_effort=options.reasoning_effort,
                cache_hit=False,
                response=response,
                extra_input={
                    "chunk_start_word_index": chunk_start,
                    "chunk_end_word_index": chunk_end,
                    "word_count": len(speech_words),
                    "has_context": bool(context),
                },
                extra_output={
                    "validation_error": str(exc),
                    "error_type": type(exc).__name__,
                },
            ))
            raise

        extra_output = (
            {"range_repairs": range_repairs, "range_repair_count": len(range_repairs)}
            if range_repairs
            else None
        )
        log.append({
            "status": "planned",
            "mode": "compact_full_words",
            "chunk_start_word_index": chunk_start,
            "chunk_end_word_index": chunk_end,
            "word_count": len(speech_words),
            "segment_count": len(ranges),
            "range_repair_count": len(range_repairs),
            "model": options.model,
            "reasoning_effort": options.reasoning_effort,
            "prompt_version": SEGMENTATION_PROMPT_VERSION,
        })
        log.append(_llm_io_log_entry(
            mode="compact_full_words",
            prompt=prompt,
            model=options.model,
            reasoning_effort=options.reasoning_effort,
            cache_hit=False,
            response=response,
            ranges=ranges,
            extra_input={
                "chunk_start_word_index": chunk_start,
                "chunk_end_word_index": chunk_end,
                "word_count": len(speech_words),
                "has_context": bool(context),
            },
            extra_output=extra_output,
        ))
        return ranges

    def _plan_ranges_legacy_chunks(
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
        for chunk_start, chunk_end in _legacy_chunk_ranges(speech_words, max_words=max_words):
            prompt = self._build_prompt(
                speech_words=speech_words,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                language_code=language_code,
                context=context,
                options=options,
                compact_table=False,
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
            range_repairs: list[dict[str, Any]] = []
            chunk_ranges = _validate_ranges(
                items,
                expected_start=chunk_start,
                expected_end=chunk_end,
                speech_words=speech_words,
                repair_log=range_repairs,
            )
            ranges.extend(chunk_ranges)
            log.append({
                "status": "planned",
                "mode": "legacy_json_word_chunks",
                "chunk_start_word_index": chunk_start,
                "chunk_end_word_index": chunk_end,
                "segment_count": len(chunk_ranges),
                "range_repair_count": len(range_repairs),
                "model": options.model,
                "reasoning_effort": options.reasoning_effort,
                "prompt_version": SEGMENTATION_PROMPT_VERSION,
            })
            log.append(_llm_io_log_entry(
                mode="legacy_json_word_chunks",
                prompt=prompt,
                model=options.model,
                reasoning_effort=options.reasoning_effort,
                cache_hit=False,
                response=response,
                ranges=chunk_ranges,
                extra_input={
                    "chunk_start_word_index": chunk_start,
                    "chunk_end_word_index": chunk_end,
                    "word_count": chunk_end - chunk_start + 1,
                    "has_context": bool(context),
                },
                extra_output=(
                    {"range_repairs": range_repairs, "range_repair_count": len(range_repairs)}
                    if range_repairs
                    else None
                ),
            ))
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
        compact_table: bool = True,
    ) -> str:
        language_line = (
            f"Language hint: {language_code}"
            if language_code
            else "Language hint: auto"
        )
        context_block = f"\nContext:\n{context}\n" if context else ""
        if compact_table:
            input_format = """Input format:
- Words is a compact pipe-delimited table.
- Word table format: index|text|speaker_id|next_gap
- index is the zero-based Scribe speech word index.
- text is recognized word text. Do not rewrite it.
- speaker_id is the speaker label, or - when unavailable.
- next_gap is seconds from this word's end to the next speech word's start.
- next_gap is only a pause hint. Larger values are boundary candidates, not timestamps.
- If next_gap is negative, the next word starts before this word ends.
- The last word's next_gap is -.
"""
            words_body = _compact_word_table(
                speech_words,
                start=chunk_start,
                end=chunk_end,
            )
        else:
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
            input_format = """Input format:
- Words is a JSON array of Scribe speech word objects.
- index is the zero-based Scribe speech word index.
- text is recognized word text.
- start and end are word timestamps in seconds.
- speaker_id is the speaker label, or null when unavailable.
"""
            words_body = json.dumps(words_payload, ensure_ascii=False, indent=2)

        return f"""You are a semantic segmentation planner for editing review.

Task:
Group the provided Scribe word tokens into edit-decision segments.
You must only decide word index ranges. Do not rewrite words. Do not invent timestamps.

Output:
- Return JSON only.
- Output a JSON array of objects with start_word_index and end_word_index.
- Cover every input word index exactly once, in ascending order.
- Do not skip, duplicate, overlap, or reorder words.

Core goal:
Each segment should be one atomic edit-decision unit.
The unit is not a topic, paragraph, or subtitle. It is the smallest useful span
that can receive one KEEP/CUT label.
A reviewer should be able to mark the whole segment as KEEP or CUT without
needing to split it again.
Do not merge material just because it belongs to the same topic or sounds coherent together.

Granularity target:
- Prefer many precise segments over fewer broad segments. For a long transcript,
  hundreds of segments are expected.
- Default to one independent claim, clause, answer, question, observation, example,
  reaction, correction, or explanation per segment.
- A segment should rarely contain more than one independently editable clause.
- Multiple short sentences should be separate segments when each sentence is
  independently understandable or independently removable.
- A coherent answer can still be multiple segments: direct answer, restatement,
  reason, implication, and evaluation are separate edit decisions.
- Do not merge separate edit decisions only because each would be short.
- Avoid one-word or filler-only segments unless the utterance is truly standalone.

Primary segmentation rules:
- Prefer smaller segments over broad topic-level segments when a span contains
  multiple actions, claims, examples, asides, or corrections.
- A segment should normally contain one primary speech act: introducing a topic,
  stating one claim, giving one example, asking one question, answering one question,
  reacting to something, navigating or searching on screen, correcting a previous
  phrase, abandoning a previous phrase, or transitioning to the next topic.
- If a span contains two or more primary speech acts, split them.
- Even if a span is coherent, split it when it contains setup, answer, restatement,
  reason, implication, aside, correction, search/navigation, and final point as
  separate parts.
- Short acknowledgement plus immediate same-answer restatement may stay together,
  but the following reason or explanation should start a new segment.

Must-split triggers:
Start a new segment whenever any of these occur:
- A quoted, read, or repeated audience comment/question ends and the speaker begins responding.
- The speaker moves from acknowledgement/restatement to the reason, evidence,
  implication, or explanation.
- The speaker moves from negative classification to positive classification,
  or from classification to definition/restatement.
- The speaker moves from setup/condition to contrast/qualification,
  or from contrast/qualification to consequence/conclusion.
- The speaker moves from direct observation to evaluation or hedged inference.
- The speaker moves from main content to process talk, or from process talk back to main content.
- The speaker searches for something, looks for an item, comments on the screen,
  UI, window, document, article, or source.
- The speaker abandons a phrase and restarts with a different sentence.
- The speaker says that the previous phrase was wrong, irrelevant, not the right item,
  or should be skipped.
- The speaker moves from setup to the actual question or point.
- The speaker moves from one example to another example.
- The speaker moves from factual explanation to personal aside, or from personal aside
  back to factual explanation.
- The speaker moves from reading or quoting visible content to explaining it.
- A long span contains multiple independently removable parts.

Clause-level edit decisions:
- Segment at clause-level granularity. A segment should usually contain one claim,
  one condition, one contrast, one consequence, one observation, one answer,
  one reason, or one evaluation.
- Split negative/positive contrast pairs into separate segments when both sides are complete claims.
- For long sentences joined by contrast, condition, cause, consequence,
  or inference markers, split the sentence into clause-level segments.
- Do not keep multiple clauses in one segment when each clause can be independently understood.
- Split chained reasoning into separate segments when it contains distinct setup,
  condition, contrast, cause, consequence, conclusion, or evaluation parts.
- Split before contrast or qualification clauses when they introduce a separate
  edit decision, such as "but", "however", "although", "while", or equivalent words
  in the spoken language.
- Split before cause, result, or inference clauses when they can stand as separate
  reasoning steps, such as "because", "so", "therefore", "that is why", "I guess",
  "I think", or equivalent words in the spoken language.
- Transition/setup, comparison, observation, and evaluation must be separate segments.
- If the speaker states multiple equivalent formulations of the same idea,
  split each formulation unless they are only a few words.
- Keep short closing inference or evaluation phrases as their own segment when
  they summarize or hedge the previous explanation.
- Sentence-ending punctuation is a mandatory boundary. If a word token ends with
  punctuation that closes a sentence, end the current segment at that word and
  start a new segment at the next word. Do not merge across sentence-ending
  punctuation, even when the meaning or topic continues.
- Commas, semicolons, colons, and phrase-separating punctuation are preferred
  boundary candidates, but they are not mandatory boundaries.
- Prefer splitting at punctuation or natural pause boundaries, but split at clause
  boundaries even when punctuation is missing.
- If the same sentence contains a condition, a qualification, and a conclusion,
  it should normally produce at least two or three segments.

Keep/Cut separation:
- Likely KEEP main content and likely CUT process artifacts must be separate segments.
- Search, navigation, and meta-commentary must be separate segments, even if short.
- False starts, abandoned attempts, and self-corrections must be separate from
  the final usable sentence.
- Do not attach process/search/correction artifacts to nearby main content for readability.
- If the speaker returns to the useful point after a removable aside, start a new
  segment at the return point.

Boundary preferences:
- Prefer boundaries at sentence endings, completed clauses, topic shifts,
  rhetorical transitions, and clear pauses.
- Use speaker changes as mandatory boundaries when speaker_id is present.
- Never mix different speaker_id values inside one segment when speaker_id is present.
- Keep discourse markers, fillers, and transition words with the following or previous
  semantic unit unless they clearly stand alone.
- Keep technical terms, named entities, and their immediate explanation together
  when they form one concept.

Maximum granularity guidance:
- When unsure between one broad segment and several smaller edit-decision segments,
  choose the smaller segments.
- Avoid segments that contain more than one independently editable sentence group.
- Avoid segments longer than roughly one paragraph of spoken content.
- Avoid one-word or filler-only segments unless the utterance is truly standalone.

Pattern example:
- Bad: [main point + search/navigation aside + correction + final question]
- Good: [main point] [search/navigation aside] [correction] [final question]
- Bad: [condition/setup + contrast/qualification + conclusion/evaluation]
- Good: [condition/setup] [contrast/qualification] [conclusion] [evaluation or inference]
- Bad: [transition + prior comparison + current observation + evaluation]
- Good: [transition/current item] [prior comparison] [current observation] [evaluation]
- Bad: [quoted/read comment/question + acknowledgement/restatement + explanation]
- Good: [quoted/read comment/question] [acknowledgement/restatement] [explanation]
- Bad: [negative claim + positive claim + restatement]
- Good: [negative claim] [positive claim] [restatement]

{input_format}

Validation constraints:
- Each returned range must use the provided word indexes.
- Ranges must be contiguous and ordered.
- Ranges must stay within the chunk word index range.
- Prefer not to create segments shorter than a meaningful spoken unit.
- Prefer not to create very long segments unless they are semantically indivisible.

{language_line}
Chunk word index range: {chunk_start}..{chunk_end}
{context_block}
Words:
{words_body}

JSON response:
"""
