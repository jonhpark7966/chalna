"""
SRT subtitle utilities.
"""

from __future__ import annotations

import re
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from chalna.models import Segment


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string (e.g., "00:01:23,456")
    """
    if seconds < 0:
        seconds = 0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_srt_timestamp(ts: str) -> float:
    """
    Convert SRT timestamp (HH:MM:SS,mmm) to seconds.

    Args:
        ts: SRT timestamp string

    Returns:
        Time in seconds
    """
    match = re.match(r'(\d+):(\d+):(\d+),(\d+)', ts)
    if not match:
        return 0.0

    h, m, s, ms = map(int, match.groups())
    return h * 3600 + m * 60 + s + ms / 1000


def segments_to_srt(segments: List[Segment], include_speaker: bool = True) -> str:
    """
    Convert segments to SRT subtitle format.

    Args:
        segments: List of Segment objects
        include_speaker: Whether to include speaker labels

    Returns:
        SRT formatted string
    """
    srt_lines = []

    for seg in segments:
        # Index
        srt_lines.append(str(seg.index))

        # Timestamps
        start_ts = format_timestamp(seg.start_time)
        end_ts = format_timestamp(seg.end_time)
        srt_lines.append(f"{start_ts} --> {end_ts}")

        # Text with speaker label (always include if include_speaker is True)
        if include_speaker:
            speaker_label = seg.speaker_id if seg.speaker_id else "Speaker None"
            srt_lines.append(f"[{speaker_label}] {seg.text}")
        else:
            srt_lines.append(seg.text)

        # Blank line between entries
        srt_lines.append("")

    return "\n".join(srt_lines)


def parse_srt(srt_content: str) -> List[dict]:
    """
    Parse SRT content into a list of segment dictionaries.

    Args:
        srt_content: SRT formatted string

    Returns:
        List of segment dictionaries with keys: index, start_time, end_time, text, speaker_id
    """
    segments = []
    blocks = re.split(r'\n\n+', srt_content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        try:
            index = int(lines[0])
        except ValueError:
            continue

        # Parse timestamps
        ts_match = re.match(
            r'(\d+:\d+:\d+,\d+)\s*-->\s*(\d+:\d+:\d+,\d+)',
            lines[1]
        )
        if not ts_match:
            continue

        start_time = parse_srt_timestamp(ts_match.group(1))
        end_time = parse_srt_timestamp(ts_match.group(2))

        # Parse text and speaker
        text = '\n'.join(lines[2:])
        speaker_id = None

        speaker_match = re.match(r'\[([^\]]+)\]\s*(.+)', text, re.DOTALL)
        if speaker_match:
            speaker_id = speaker_match.group(1)
            text = speaker_match.group(2)

        segments.append({
            "index": index,
            "start_time": start_time,
            "end_time": end_time,
            "text": text.strip(),
            "speaker_id": speaker_id,
        })

    return segments


def merge_short_segments(
    segments: List[Segment],
    min_duration: float = 0.3,
    min_chars: int = 3
) -> List[Segment]:
    """
    Merge segments that are too short.

    Args:
        segments: List of segments
        min_duration: Minimum duration in seconds
        min_chars: Minimum character count

    Returns:
        List of merged segments
    """
    from chalna.models import Segment as SegmentClass

    if not segments:
        return []

    merged = []
    pending = None

    for seg in segments:
        if pending is None:
            pending = seg
            continue

        # Check if pending segment is too short
        should_merge = (
            pending.duration < min_duration or
            len(pending.text) < min_chars
        )

        # Only merge if same speaker (or no speaker info)
        same_speaker = (
            pending.speaker_id is None or
            seg.speaker_id is None or
            pending.speaker_id == seg.speaker_id
        )

        if should_merge and same_speaker:
            # Merge pending with current
            pending = SegmentClass(
                index=pending.index,
                start_time=pending.start_time,
                end_time=seg.end_time,
                text=f"{pending.text} {seg.text}".strip(),
                speaker_id=pending.speaker_id or seg.speaker_id,
                confidence=min(pending.confidence, seg.confidence),
            )
        else:
            merged.append(pending)
            pending = seg

    if pending:
        merged.append(pending)

    # Re-index
    for i, seg in enumerate(merged, start=1):
        seg.index = i

    return merged
