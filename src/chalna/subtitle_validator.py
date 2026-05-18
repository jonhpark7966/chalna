"""
Subtitle validation helpers for edit-safe segment timing.
"""

from __future__ import annotations

import subprocess
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, List, Optional

from chalna.models import Segment

REPEATED_TOKEN_TAIL_MIN_RUN = 16
REPEATED_TOKEN_LOOP_MIN_TOKENS = 50
REPEATED_TOKEN_LOOP_MIN_RATIO = 0.6


@dataclass
class SubtitleValidationIssue:
    """A single subtitle validation issue."""

    code: str
    message: str
    segment_index: int
    severity: str = "warning"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    text: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "segment_index": self.segment_index,
            "severity": self.severity,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "details": self.details,
        }


@dataclass
class _BoundaryAlignmentContext:
    """Audio/text context used to validate one target segment."""

    segments: list[Segment]
    target_offset: int
    window_start: float
    window_end: float

    @property
    def alignment_text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments if segment.text.strip())

    @property
    def window_duration(self) -> float:
        return self.window_end - self.window_start


@dataclass
class _TargetWordRange:
    """Word range selected for the target segment inside aligned context."""

    start_index: int
    end_index: int
    method: str
    match_ratio: float
    selected_text: str
    target_text: str
    fallback_start_index: int
    fallback_end_index: int
    fallback_match_ratio: float


def validate_segments(
    segments: List[Segment],
    *,
    edit_points: Optional[Iterable[float]] = None,
    max_duration: float = 12.0,
    max_chars: int = 160,
) -> List[SubtitleValidationIssue]:
    """Validate basic subtitle segment constraints."""
    issues: List[SubtitleValidationIssue] = []
    edit_points = list(edit_points or [])

    for segment in segments:
        text = segment.text.strip()
        if segment.duration <= 0:
            issues.append(_issue(
                "nonpositive_duration",
                "Segment duration must be positive.",
                segment,
                severity="error",
                duration=segment.duration,
            ))

        if segment.duration > max_duration:
            issues.append(_issue(
                "duration_too_long",
                "Segment duration exceeds the configured maximum.",
                segment,
                duration=segment.duration,
                max_duration=max_duration,
            ))

        if len(text) > max_chars:
            issues.append(_issue(
                "text_too_long",
                "Segment text exceeds the configured character maximum.",
                segment,
                chars=len(text),
                max_chars=max_chars,
            ))

        stripped_repetition = trim_repeated_token_tail(text)
        if stripped_repetition != text:
            issues.append(_issue(
                "repeated_token_loop",
                "Segment appears to contain repeated-token hallucination.",
                segment,
                severity="error",
                original_chars=len(text),
                cleaned_chars=len(stripped_repetition),
            ))
        elif has_repeated_token_loop(text):
            issues.append(_issue(
                "repeated_token_loop",
                "Segment appears to contain repeated-token hallucination.",
                segment,
                severity="error",
                original_chars=len(text),
            ))

        for edit_point in edit_points:
            if segment.start_time < edit_point < segment.end_time:
                issues.append(_issue(
                    "crosses_edit_point",
                    "Segment crosses an edit point.",
                    segment,
                    edit_point=edit_point,
                ))

        if text in {"습니다", "요", "다"}:
            issues.append(_issue(
                "orphaned_korean_ending",
                "Segment contains only an orphaned Korean ending.",
                segment,
            ))

    return issues


def validate_audio_boundaries(
    audio_path: str | Path,
    segments: List[Segment],
    aligner: Any,
    *,
    scan_padding: float = 0.5,
    min_start_padding: float = 0.15,
    min_end_padding: float = 0.15,
    cutoff_tolerance: float = 0.03,
    language: str = "Korean",
) -> List[SubtitleValidationIssue]:
    """
    Check whether subtitle boundaries are too tight against speech.

    Each segment is re-aligned with neighboring subtitle text that overlaps the
    scan window. Only the aligned word range for the target segment is compared
    against the target segment boundaries.
    """
    audio_path = Path(audio_path)
    issues: List[SubtitleValidationIssue] = []

    normalized_audio_path: Optional[Path] = None
    try:
        normalized_audio_path = _extract_validation_audio(audio_path)

        for segment_position, segment in enumerate(segments):
            if not segment.text.strip() or segment.duration <= 0:
                continue

            context = _build_boundary_alignment_context(
                segments,
                segment_position,
                scan_padding=scan_padding,
            )

            try:
                words = _align_segment_window(
                    audio_path=normalized_audio_path,
                    text=context.alignment_text,
                    aligner=aligner,
                    window_start=context.window_start,
                    window_duration=context.window_duration,
                    language=language,
                )
            except Exception as error:
                issues.append(_issue(
                    "boundary_alignment_failed",
                    "Boundary alignment failed for this segment.",
                    segment,
                    severity="info",
                    error=str(error),
                ))
                continue

            if not words:
                issues.append(_issue(
                    "boundary_alignment_no_result",
                    "Boundary alignment returned no word timestamps.",
                    segment,
                    severity="info",
                ))
                continue

            target_word_range = _target_word_range(
                words,
                context.segments,
                context.target_offset,
            )
            if target_word_range is None:
                issues.append(_issue(
                    "boundary_alignment_no_target_range",
                    "Boundary alignment could not locate the target segment words.",
                    segment,
                    severity="info",
                    context_segment_indices=[item.index for item in context.segments],
                    context_window_start=context.window_start,
                    context_window_end=context.window_end,
                ))
                continue

            first_word = words[target_word_range.start_index]
            last_word = words[target_word_range.end_index]
            speech_start = context.window_start + float(first_word.start_time)
            speech_end = context.window_start + float(last_word.end_time)
            start_padding = speech_start - segment.start_time
            end_padding = segment.end_time - speech_end
            alignment_relation = _classify_alignment_relation(
                segment.start_time,
                segment.end_time,
                speech_start,
                speech_end,
            )

            common_details = {
                "segment_start": segment.start_time,
                "segment_end": segment.end_time,
                "speech_start": speech_start,
                "speech_end": speech_end,
                "start_padding": start_padding,
                "end_padding": end_padding,
                "min_start_padding": min_start_padding,
                "min_end_padding": min_end_padding,
                "scan_padding": scan_padding,
                "context_segment_indices": [item.index for item in context.segments],
                "context_window_start": context.window_start,
                "context_window_end": context.window_end,
                "target_word_start_index": target_word_range.start_index,
                "target_word_end_index": target_word_range.end_index,
                "target_word_range_method": target_word_range.method,
                "target_match_ratio": target_word_range.match_ratio,
                "target_selected_text": target_word_range.selected_text[:120],
                "target_text_normalized": target_word_range.target_text[:120],
                "fallback_word_start_index": target_word_range.fallback_start_index,
                "fallback_word_end_index": target_word_range.fallback_end_index,
                "fallback_match_ratio": target_word_range.fallback_match_ratio,
                "alignment_relation": alignment_relation,
                "first_word": getattr(first_word, "text", None),
                "last_word": getattr(last_word, "text", None),
            }

            if start_padding < -cutoff_tolerance:
                issues.append(_issue(
                    "cuts_speech_start",
                    "Segment starts after the aligned first word has already begun.",
                    segment,
                    severity="error",
                    **common_details,
                ))
            elif start_padding <= min_start_padding:
                issues.append(_issue(
                    "tight_start_boundary",
                    "Segment start is too close to the aligned first word.",
                    segment,
                    **common_details,
                ))

            if end_padding < -cutoff_tolerance:
                issues.append(_issue(
                    "cuts_speech_end",
                    "Segment ends before the aligned last word has finished.",
                    segment,
                    severity="error",
                    **common_details,
                ))
            elif end_padding <= min_end_padding:
                issues.append(_issue(
                    "tight_end_boundary",
                    "Segment end is too close to the aligned last word.",
                    segment,
                    **common_details,
                ))
    finally:
        if normalized_audio_path is not None:
            normalized_audio_path.unlink(missing_ok=True)

    return issues


def summarize_validation_diagnostics(
    segments: List[Segment],
    issues: List[SubtitleValidationIssue],
    *,
    tight_gap_seconds: float = 0.15,
) -> dict[str, Any]:
    """Build aggregate diagnostics that map validation results to likely causes."""
    issue_counts = Counter(issue.code for issue in issues)
    segment_issue_codes: dict[int, set[str]] = {}
    for issue in issues:
        segment_issue_codes.setdefault(issue.segment_index, set()).add(issue.code)

    start_cut_segments = {
        issue.segment_index for issue in issues if issue.code == "cuts_speech_start"
    }
    end_cut_segments = {
        issue.segment_index for issue in issues if issue.code == "cuts_speech_end"
    }

    prev_gaps: dict[int, float] = {}
    next_gaps: dict[int, float] = {}
    same_speaker_prev: dict[int, bool] = {}
    same_speaker_next: dict[int, bool] = {}
    for previous, current in zip(segments, segments[1:]):
        gap = current.start_time - previous.end_time
        prev_gaps[current.index] = gap
        next_gaps[previous.index] = gap
        same_speaker = previous.speaker_id == current.speaker_id
        same_speaker_prev[current.index] = same_speaker
        same_speaker_next[previous.index] = same_speaker

    boundary_issues = [
        issue
        for issue in issues
        if "speech_start" in issue.details and "speech_end" in issue.details
    ]
    relation_counts = Counter(
        issue.details.get("alignment_relation", "unknown") for issue in boundary_issues
    )
    range_method_counts = Counter(
        issue.details.get("target_word_range_method", "unknown") for issue in boundary_issues
    )
    low_match_count = sum(
        1
        for issue in boundary_issues
        if float(issue.details.get("target_match_ratio", 1.0)) < 0.7
    )
    speech_start_after_segment_end = sum(
        1
        for issue in boundary_issues
        if issue.details["speech_start"] > issue.details["segment_end"]
    )
    speech_end_before_segment_start = sum(
        1
        for issue in boundary_issues
        if issue.details["speech_end"] < issue.details["segment_start"]
    )
    edge_saturated_start_cuts = sum(
        1
        for issue in issues
        if issue.code == "cuts_speech_start"
        and abs(
            issue.details["speech_start"]
            - (issue.details["segment_start"] - issue.details["scan_padding"])
        )
        < 0.011
    )
    edge_saturated_end_cuts = sum(
        1
        for issue in issues
        if issue.code == "cuts_speech_end"
        and abs(
            issue.details["speech_end"]
            - (issue.details["segment_end"] + issue.details["scan_padding"])
        )
        < 0.011
    )

    durations = [segment.duration for segment in segments]
    return {
        "issue_counts": dict(sorted(issue_counts.items())),
        "affected_segments": len(segment_issue_codes),
        "cut_segments": sum(
            1
            for codes in segment_issue_codes.values()
            if any(code.startswith("cuts_") for code in codes)
        ),
        "tight_segments": sum(
            1
            for codes in segment_issue_codes.values()
            if any(code.startswith("tight_") for code in codes)
        ),
        "contract": {
            "nonpositive_duration": sum(segment.duration <= 0 for segment in segments),
            "long_text_over_160": sum(len(segment.text.strip()) > 160 for segment in segments),
            "repeated_token_loop": issue_counts.get("repeated_token_loop", 0),
        },
        "duration": {
            "segment_count": len(segments),
            "short_le_0_5s": sum(duration <= 0.5 for duration in durations),
            "short_le_1_0s": sum(duration <= 1.0 for duration in durations),
            "long_gt_8s": sum(duration > 8.0 for duration in durations),
        },
        "gap": {
            "total_gaps": max(len(segments) - 1, 0),
            "zero_gap": sum(abs(gap) <= 0.005 for gap in next_gaps.values()),
            "tight_gap_lt_threshold": sum(gap < tight_gap_seconds for gap in next_gaps.values()),
            "same_speaker_zero_gap": sum(
                abs(gap) <= 0.005 and same_speaker_next.get(index, False)
                for index, gap in next_gaps.items()
            ),
            "different_speaker_zero_gap": sum(
                abs(gap) <= 0.005 and not same_speaker_next.get(index, False)
                for index, gap in next_gaps.items()
            ),
            "cuts_speech_start_by_prev_gap": _gap_issue_rates(
                prev_gaps,
                start_cut_segments,
                tight_gap_seconds=tight_gap_seconds,
                same_speaker=same_speaker_prev,
            ),
            "cuts_speech_end_by_next_gap": _gap_issue_rates(
                next_gaps,
                end_cut_segments,
                tight_gap_seconds=tight_gap_seconds,
                same_speaker=same_speaker_next,
            ),
        },
        "alignment": {
            "relation_counts": dict(sorted(relation_counts.items())),
            "target_word_range_method_counts": dict(sorted(range_method_counts.items())),
            "low_target_match_ratio_lt_0_7": low_match_count,
            "speech_start_after_segment_end": speech_start_after_segment_end,
            "speech_end_before_segment_start": speech_end_before_segment_start,
            "edge_saturated_start_cuts": edge_saturated_start_cuts,
            "edge_saturated_end_cuts": edge_saturated_end_cuts,
        },
    }


def trim_repeated_token_tail(
    text: str,
    *,
    min_run: int = REPEATED_TOKEN_TAIL_MIN_RUN,
) -> str:
    """Remove an obvious repeated-token tail while preserving the useful prefix."""
    tokens = str(text).split()
    if len(tokens) < min_run:
        return text.strip()

    normalized = [_normalize_repetition_token(token) for token in tokens]
    last_token = normalized[-1]
    if not last_token:
        return text.strip()

    run_start = len(tokens) - 1
    while run_start > 0 and normalized[run_start - 1] == last_token:
        run_start -= 1

    if len(tokens) - run_start < min_run:
        return text.strip()

    return " ".join(tokens[:run_start]).strip()


def has_repeated_token_loop(
    text: str,
    *,
    min_tokens: int = REPEATED_TOKEN_LOOP_MIN_TOKENS,
    min_ratio: float = REPEATED_TOKEN_LOOP_MIN_RATIO,
) -> bool:
    """Detect repeated-token hallucination that is not limited to the tail."""
    tokens = [
        token
        for token in (_normalize_repetition_token(token) for token in str(text).split())
        if token
    ]
    if len(tokens) < min_tokens:
        return False

    _, count = Counter(tokens).most_common(1)[0]
    return count / len(tokens) >= min_ratio


def _build_boundary_alignment_context(
    segments: list[Segment],
    target_position: int,
    *,
    scan_padding: float,
) -> _BoundaryAlignmentContext:
    target = segments[target_position]
    requested_start = max(0.0, target.start_time - scan_padding)
    requested_end = target.end_time + scan_padding

    context_entries = [
        (position, segment)
        for position, segment in enumerate(segments)
        if (
            segment.text.strip()
            and segment.duration > 0
            and segment.end_time > requested_start
            and segment.start_time < requested_end
        )
    ]
    if not any(position == target_position for position, _ in context_entries):
        context_entries.append((target_position, target))
        context_entries.sort(key=lambda item: item[0])

    context_segments = [segment for _, segment in context_entries]
    target_offset = next(
        offset
        for offset, (position, _) in enumerate(context_entries)
        if position == target_position
    )

    window_start = min(requested_start, *(segment.start_time for segment in context_segments))
    window_end = max(requested_end, *(segment.end_time for segment in context_segments))

    return _BoundaryAlignmentContext(
        segments=context_segments,
        target_offset=target_offset,
        window_start=max(0.0, window_start),
        window_end=window_end,
    )


def _target_word_range(
    words: list,
    context_segments: list[Segment],
    target_offset: int,
) -> Optional[_TargetWordRange]:
    expected_lengths = [
        len(_normalize_alignment_text(segment.text))
        for segment in context_segments
    ]
    if target_offset >= len(expected_lengths) or expected_lengths[target_offset] == 0:
        return None

    target_text = _normalize_alignment_text(context_segments[target_offset].text)
    target_start_char = sum(expected_lengths[:target_offset])
    target_end_char = target_start_char + expected_lengths[target_offset]
    expected_total_chars = max(sum(expected_lengths), 1)

    char_to_word_idx: list[int] = []
    word_texts: list[str] = []
    for word_idx, word in enumerate(words):
        word_text = _normalize_alignment_text(getattr(word, "text", ""))
        word_texts.append(word_text)
        char_to_word_idx.extend([word_idx] * len(word_text))

    if not char_to_word_idx:
        return None

    start_word_idx = _char_position_to_word_index(
        char_to_word_idx,
        target_start_char,
        expected_total_chars,
    )
    end_word_idx = _char_position_to_word_index(
        char_to_word_idx,
        target_end_char - 1,
        expected_total_chars,
    )
    if start_word_idx is None or end_word_idx is None:
        return None
    if end_word_idx < start_word_idx:
        end_word_idx = start_word_idx

    fallback_selected_text = _span_text(word_texts, start_word_idx, end_word_idx)
    fallback_match_ratio = _text_similarity(target_text, fallback_selected_text)
    candidate = _best_text_match_word_range(
        word_texts,
        target_text,
        fallback_start_index=start_word_idx,
        fallback_end_index=end_word_idx,
    )

    if candidate is not None:
        candidate_start, candidate_end, candidate_score, candidate_text = candidate
        if (
            len(target_text) >= 4
            and candidate_score >= 0.75
            and candidate_score >= fallback_match_ratio + 0.05
        ):
            return _TargetWordRange(
                start_index=candidate_start,
                end_index=candidate_end,
                method="text_match",
                match_ratio=candidate_score,
                selected_text=candidate_text,
                target_text=target_text,
                fallback_start_index=start_word_idx,
                fallback_end_index=end_word_idx,
                fallback_match_ratio=fallback_match_ratio,
            )

    return _TargetWordRange(
        start_index=start_word_idx,
        end_index=end_word_idx,
        method="char_range",
        match_ratio=fallback_match_ratio,
        selected_text=fallback_selected_text,
        target_text=target_text,
        fallback_start_index=start_word_idx,
        fallback_end_index=end_word_idx,
        fallback_match_ratio=fallback_match_ratio,
    )


def _best_text_match_word_range(
    word_texts: list[str],
    target_text: str,
    *,
    fallback_start_index: int,
    fallback_end_index: int,
) -> Optional[tuple[int, int, float, str]]:
    if not target_text or len(word_texts) > 500 or len(target_text) > 400:
        return None

    best: Optional[tuple[int, int, float, str]] = None
    target_len = len(target_text)
    max_span_chars = max(target_len + 20, int(target_len * 1.5))

    for start_index in range(len(word_texts)):
        span_text = ""
        for end_index in range(start_index, len(word_texts)):
            span_text += word_texts[end_index]
            if not span_text:
                continue
            if len(span_text) > max_span_chars:
                break

            length_ratio = len(span_text) / max(target_len, 1)
            if length_ratio < 0.45:
                continue

            similarity = _text_similarity(target_text, span_text)
            length_penalty = min(abs(len(span_text) - target_len) / max(target_len, 1), 1.0)
            distance_penalty = (
                abs(start_index - fallback_start_index)
                + abs(end_index - fallback_end_index)
            ) / max(len(word_texts), 1)
            score = similarity - 0.15 * length_penalty - 0.10 * distance_penalty

            if best is None or score > best[2]:
                best = (start_index, end_index, score, span_text)

            if len(span_text) >= target_len * 1.2:
                break

    if best is None:
        return None

    start_index, end_index, _, span_text = best
    return start_index, end_index, _text_similarity(target_text, span_text), span_text


def _span_text(word_texts: list[str], start_index: int, end_index: int) -> str:
    if not word_texts:
        return ""
    start_index = max(0, min(start_index, len(word_texts) - 1))
    end_index = max(start_index, min(end_index, len(word_texts) - 1))
    return "".join(word_texts[start_index:end_index + 1])


def _text_similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _classify_alignment_relation(
    segment_start: float,
    segment_end: float,
    speech_start: float,
    speech_end: float,
) -> str:
    if speech_end < segment_start:
        return "before_segment"
    if speech_start > segment_end:
        return "after_segment"
    if speech_start < segment_start and speech_end > segment_end:
        return "covers_segment"
    if speech_start < segment_start:
        return "starts_before_segment"
    if speech_end > segment_end:
        return "ends_after_segment"
    return "inside_segment"


def _gap_issue_rates(
    gaps: dict[int, float],
    issue_segments: set[int],
    *,
    tight_gap_seconds: float,
    same_speaker: dict[int, bool],
) -> dict[str, dict[str, float | int]]:
    groups = {
        "zero": {index for index, gap in gaps.items() if abs(gap) <= 0.005},
        "tight": {index for index, gap in gaps.items() if gap < tight_gap_seconds},
        "loose": {index for index, gap in gaps.items() if gap >= tight_gap_seconds},
        "same_speaker_zero": {
            index
            for index, gap in gaps.items()
            if abs(gap) <= 0.005 and same_speaker.get(index, False)
        },
        "different_speaker_zero": {
            index
            for index, gap in gaps.items()
            if abs(gap) <= 0.005 and not same_speaker.get(index, False)
        },
    }

    return {
        label: _rate_summary(issue_segments, members)
        for label, members in groups.items()
    }


def _rate_summary(issue_segments: set[int], members: set[int]) -> dict[str, float | int]:
    affected = len(issue_segments & members)
    total = len(members)
    return {
        "affected": affected,
        "total": total,
        "rate": affected / total if total else 0.0,
    }


def _normalize_alignment_text(text: Any) -> str:
    return "".join(
        char
        for char in str(text)
        if not char.isspace() and unicodedata.category(char)[0] not in {"P", "S"}
    )


def _normalize_repetition_token(token: str) -> str:
    return "".join(
        char
        for char in str(token).casefold()
        if not char.isspace() and unicodedata.category(char)[0] not in {"P", "S"}
    )


def _char_position_to_word_index(
    char_to_word_idx: list[int],
    char_position: int,
    expected_total_chars: int,
) -> Optional[int]:
    if not char_to_word_idx:
        return None
    if char_position < len(char_to_word_idx):
        return char_to_word_idx[max(0, char_position)]

    ratio = char_position / max(expected_total_chars - 1, 1)
    aligned_char_position = round(ratio * (len(char_to_word_idx) - 1))
    return char_to_word_idx[max(0, min(aligned_char_position, len(char_to_word_idx) - 1))]


def _extract_validation_audio(audio_path: Path) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_path),
                "-map",
                "0:a:0",
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(tmp_path),
            ],
            capture_output=True,
            check=True,
        )
        return tmp_path
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _align_segment_window(
    *,
    audio_path: Path,
    text: str,
    aligner: Any,
    window_start: float,
    window_duration: float,
    language: str,
) -> list:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(window_start),
                "-i",
                str(audio_path),
                "-t",
                str(window_duration),
                "-map",
                "0:a:0",
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(tmp_path),
            ],
            capture_output=True,
            check=True,
        )

        results = aligner.align(
            audio=str(tmp_path),
            text=text,
            language=language,
        )
        if not results or len(results) == 0:
            return []
        return list(results[0] or [])
    finally:
        tmp_path.unlink(missing_ok=True)


def _issue(
    code: str,
    message: str,
    segment: Segment,
    *,
    severity: str = "warning",
    **details: Any,
) -> SubtitleValidationIssue:
    return SubtitleValidationIssue(
        code=code,
        message=message,
        segment_index=segment.index,
        severity=severity,
        start_time=segment.start_time,
        end_time=segment.end_time,
        text=segment.text,
        details=details,
    )
