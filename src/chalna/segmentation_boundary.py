"""Post-segmentation timestamp boundary rules."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chalna.models import Segment

BOUNDARY_RULE_WORD = "word_boundary"
BOUNDARY_RULE_MIDPOINT = "midpoint_gap"
BOUNDARY_RULE_LOW_ENERGY = "low_energy_gap_v1"
ALLOWED_BOUNDARY_RULES = {
    BOUNDARY_RULE_WORD,
    BOUNDARY_RULE_MIDPOINT,
    BOUNDARY_RULE_LOW_ENERGY,
}
DEFAULT_BOUNDARY_RULE = BOUNDARY_RULE_WORD


@dataclass(frozen=True)
class BoundaryRuleOptions:
    audio_search_max_gap_ms: int = 1500
    max_gap_padding_ms: int = 250
    analysis_window_ms: int = 80
    analysis_hop_ms: int = 20
    min_boundary_margin_ms: int = 40
    sample_rate: int = 16000


@dataclass
class BoundaryRuleResult:
    segments: list[Segment]
    rule: str
    effective_rule: str
    stats: dict[str, Any] = field(default_factory=dict)


class _PcmAudio:
    def __init__(self, samples: array, sample_rate: int) -> None:
        self.samples = samples
        self.sample_rate = sample_rate

    @classmethod
    def decode(cls, audio_path: Path, sample_rate: int) -> "_PcmAudio":
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".s16le", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(audio_path),
                    "-ac",
                    "1",
                    "-ar",
                    str(sample_rate),
                    "-f",
                    "s16le",
                    str(tmp_path),
                ],
                check=True,
                capture_output=True,
            )
            samples = array("h")
            samples.frombytes(tmp_path.read_bytes())
            if sys.byteorder != "little":
                samples.byteswap()
            return cls(samples=samples, sample_rate=sample_rate)
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    def quiet_boundary_seconds(
        self,
        start_seconds: float,
        end_seconds: float,
        *,
        window_ms: int,
        hop_ms: int,
        margin_ms: int,
    ) -> float | None:
        margin_seconds = max(0.0, margin_ms / 1000.0)
        search_start = start_seconds + margin_seconds
        search_end = end_seconds - margin_seconds
        if search_end <= search_start:
            return None

        start_sample = max(0, int(round(search_start * self.sample_rate)))
        end_sample = min(len(self.samples), int(round(search_end * self.sample_rate)))
        window_samples = max(1, int(round(window_ms * self.sample_rate / 1000.0)))
        hop_samples = max(1, int(round(hop_ms * self.sample_rate / 1000.0)))
        if end_sample - start_sample < window_samples:
            return None

        local = self.samples[start_sample:end_sample]
        prefix = [0]
        running = 0
        for sample in local:
            running += int(sample) * int(sample)
            prefix.append(running)

        best_offset = 0
        best_energy: float | None = None
        midpoint_offset = (len(local) - window_samples) / 2.0
        offset = 0
        while offset + window_samples <= len(local):
            energy = (prefix[offset + window_samples] - prefix[offset]) / window_samples
            if (
                best_energy is None
                or energy < best_energy
                or (
                    energy == best_energy
                    and abs(offset - midpoint_offset) < abs(best_offset - midpoint_offset)
                )
            ):
                best_energy = energy
                best_offset = offset
            offset += hop_samples

        return (start_sample + best_offset + window_samples / 2.0) / self.sample_rate


def normalize_boundary_rule(value: str | None) -> str:
    if not value:
        return DEFAULT_BOUNDARY_RULE
    if value not in ALLOWED_BOUNDARY_RULES:
        raise ValueError(
            "segmentation_boundary_rule must be one of: "
            + ", ".join(sorted(ALLOWED_BOUNDARY_RULES))
        )
    return value


def _clone_segments(segments: list[Segment]) -> list[Segment]:
    return [
        Segment(
            index=segment.index,
            start_time=segment.start_time,
            end_time=segment.end_time,
            text=segment.text,
            speaker_id=segment.speaker_id,
            confidence=segment.confidence,
        )
        for segment in segments
    ]


def _boundary_stats(rule: str, effective_rule: str, options: BoundaryRuleOptions) -> dict[str, Any]:
    return {
        "rule": rule,
        "effective_rule": effective_rule,
        "audio_search_max_gap_ms": options.audio_search_max_gap_ms,
        "max_gap_padding_ms": options.max_gap_padding_ms,
        "analysis_window_ms": options.analysis_window_ms,
        "analysis_hop_ms": options.analysis_hop_ms,
        "min_boundary_margin_ms": options.min_boundary_margin_ms,
        "pairs_checked": 0,
        "unchanged_boundaries": 0,
        "midpoint_boundaries": 0,
        "low_energy_boundaries": 0,
        "capped_gap_boundaries": 0,
        "overlap_repairs": 0,
        "fallback_boundaries": 0,
        "decode_failed": False,
        "decode_error": None,
    }


def apply_boundary_rule(
    segments: list[Segment],
    *,
    rule: str | None,
    audio_path: Path | None = None,
    options: BoundaryRuleOptions | None = None,
) -> BoundaryRuleResult:
    normalized_rule = normalize_boundary_rule(rule)
    opts = options or BoundaryRuleOptions()
    adjusted = _clone_segments(segments)

    effective_rule = normalized_rule
    stats = _boundary_stats(normalized_rule, effective_rule, opts)
    if normalized_rule == BOUNDARY_RULE_WORD or len(adjusted) <= 1:
        stats["unchanged_boundaries"] = max(0, len(adjusted) - 1)
        return BoundaryRuleResult(
            segments=adjusted,
            rule=normalized_rule,
            effective_rule=effective_rule,
            stats=stats,
        )

    pcm_audio: _PcmAudio | None = None
    if normalized_rule == BOUNDARY_RULE_LOW_ENERGY:
        if audio_path is None:
            effective_rule = BOUNDARY_RULE_MIDPOINT
            stats["effective_rule"] = effective_rule
            stats["decode_failed"] = True
            stats["decode_error"] = "audio_path_missing"
        else:
            try:
                pcm_audio = _PcmAudio.decode(Path(audio_path), opts.sample_rate)
            except Exception as exc:
                effective_rule = BOUNDARY_RULE_MIDPOINT
                stats["effective_rule"] = effective_rule
                stats["decode_failed"] = True
                stats["decode_error"] = f"{type(exc).__name__}: {exc}"

    max_search_gap = opts.audio_search_max_gap_ms / 1000.0
    max_padding = opts.max_gap_padding_ms / 1000.0

    for current, following in zip(adjusted, adjusted[1:]):
        stats["pairs_checked"] += 1
        raw_end = current.end_time
        next_start = following.start_time
        gap = next_start - raw_end

        if gap <= 0:
            if current.end_time > following.start_time:
                boundary = (current.end_time + following.start_time) / 2.0
                current.end_time = boundary
                following.start_time = boundary
                stats["overlap_repairs"] += 1
            else:
                stats["unchanged_boundaries"] += 1
            continue

        if gap > max_search_gap:
            padding = min(max_padding, gap / 2.0)
            current.end_time = raw_end + padding
            following.start_time = next_start - padding
            stats["capped_gap_boundaries"] += 1
            continue

        boundary: float | None = None
        if normalized_rule == BOUNDARY_RULE_LOW_ENERGY and pcm_audio is not None:
            boundary = pcm_audio.quiet_boundary_seconds(
                raw_end,
                next_start,
                window_ms=opts.analysis_window_ms,
                hop_ms=opts.analysis_hop_ms,
                margin_ms=opts.min_boundary_margin_ms,
            )
            if boundary is not None:
                stats["low_energy_boundaries"] += 1

        if boundary is None:
            boundary = (raw_end + next_start) / 2.0
            stats["midpoint_boundaries"] += 1
            if normalized_rule == BOUNDARY_RULE_LOW_ENERGY:
                stats["fallback_boundaries"] += 1

        current.end_time = boundary
        following.start_time = boundary

    return BoundaryRuleResult(
        segments=adjusted,
        rule=normalized_rule,
        effective_rule=effective_rule,
        stats=stats,
    )
