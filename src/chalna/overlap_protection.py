"""Final segment protection for detected overlapped speech intervals."""

from __future__ import annotations

from typing import Any

from chalna.models import Segment


def protect_overlapped_segments(
    segments: list[Segment],
    overlap_payload: dict[str, Any] | list[Any] | None,
) -> tuple[list[Segment], dict[str, Any] | None]:
    """Merge consecutive final segments that intersect detected overlap intervals."""
    intervals = _normalize_intervals(overlap_payload)
    if not intervals:
        if overlap_payload:
            return segments, {
                "enabled": True,
                "status": "no_intervals",
                "input_intervals": 0,
                "merged_segments": 0,
                "merged_runs": 0,
            }
        return segments, None

    affected = [_segment_intervals(segment, intervals) for segment in segments]
    output: list[Segment] = []
    merged_runs = 0
    merged_segment_count = 0
    affected_segment_count = sum(1 for items in affected if items)
    i = 0
    while i < len(segments):
        if not affected[i]:
            output.append(segments[i])
            i += 1
            continue

        start = i
        while i + 1 < len(segments) and affected[i + 1]:
            i += 1
        end = i

        if end == start:
            output.append(segments[start])
            i += 1
            continue

        run = segments[start : end + 1]
        run_intervals = _dedupe_intervals(
            interval
            for affected_intervals in affected[start : end + 1]
            for interval in affected_intervals
        )
        output.append(_merge_run(run, run_intervals))
        merged_runs += 1
        merged_segment_count += len(run)
        i += 1

    summary = {
        "enabled": True,
        "status": "complete",
        "input_intervals": len(intervals),
        "affected_segments": affected_segment_count,
        "merged_runs": merged_runs,
        "merged_segments": merged_segment_count,
        "output_segments": len(output),
    }
    return output, summary


def _normalize_intervals(payload: dict[str, Any] | list[Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []

    raw_intervals: Any
    if isinstance(payload, dict):
        raw_intervals = payload.get("intervals") or []
    else:
        raw_intervals = payload

    intervals: list[dict[str, Any]] = []
    if not isinstance(raw_intervals, list):
        return intervals

    for item in raw_intervals:
        if not isinstance(item, dict):
            continue
        start_ms = _interval_ms(item, "start_ms", "start_time", "start")
        end_ms = _interval_ms(item, "end_ms", "end_time", "end")
        if start_ms is None or end_ms is None or end_ms <= start_ms:
            continue
        models = item.get("models") or item.get("overlap_models") or []
        if isinstance(models, str):
            models = [models]
        if not isinstance(models, list):
            models = []
        intervals.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "start_time": start_ms / 1000.0,
            "end_time": end_ms / 1000.0,
            "models": sorted({str(model) for model in models if model}),
        })

    intervals.sort(key=lambda interval: (interval["start_ms"], interval["end_ms"]))
    return intervals


def _interval_ms(item: dict[str, Any], ms_key: str, seconds_key: str, fallback_key: str) -> int | None:
    value = item.get(ms_key)
    if value is not None:
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            return None

    value = item.get(seconds_key, item.get(fallback_key))
    if value is None:
        return None
    try:
        return int(round(float(value) * 1000.0))
    except (TypeError, ValueError):
        return None


def _segment_intervals(segment: Segment, intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for interval in intervals:
        if segment.start_time < interval["end_time"] and segment.end_time > interval["start_time"]:
            matches.append(interval)
    return matches


def _dedupe_intervals(intervals: Any) -> list[dict[str, Any]]:
    deduped: dict[tuple[int, int], dict[str, Any]] = {}
    for interval in intervals:
        key = (int(interval["start_ms"]), int(interval["end_ms"]))
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = dict(interval)
            continue
        models = set(existing.get("models") or [])
        models.update(interval.get("models") or [])
        existing["models"] = sorted(models)
    return list(deduped.values())


def _merge_run(run: list[Segment], intervals: list[dict[str, Any]]) -> Segment:
    speakers = sorted({segment.speaker_id for segment in run if segment.speaker_id})
    speaker_id = speakers[0] if len(speakers) == 1 else ("mixed" if speakers else None)
    models = sorted({
        model
        for interval in intervals
        for model in (interval.get("models") or [])
        if model
    })
    source_indices = [segment.index for segment in run]
    return Segment(
        index=run[0].index,
        start_time=run[0].start_time,
        end_time=run[-1].end_time,
        text=" ".join(segment.text.strip() for segment in run if segment.text.strip()),
        speaker_id=speaker_id,
        confidence=min(segment.confidence for segment in run),
        overlap_protection={
            "enabled": True,
            "merged": True,
            "reason": "overlap_protection",
            "source_segment_indices": source_indices,
            "source_segment_count": len(source_indices),
            "speaker_ids": speakers,
            "overlap_models": models,
            "overlap_intervals_ms": [
                {
                    "start_ms": interval["start_ms"],
                    "end_ms": interval["end_ms"],
                    "models": interval.get("models") or [],
                }
                for interval in intervals
            ],
        },
    )
