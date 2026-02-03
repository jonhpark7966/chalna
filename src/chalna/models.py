"""
Data models for Chalna.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Segment:
    """A single transcription segment."""

    index: int
    start_time: float  # seconds
    end_time: float  # seconds
    text: str
    speaker_id: Optional[str] = None
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "speaker_id": self.speaker_id,
            "confidence": self.confidence,
        }


@dataclass
class TranscriptionMetadata:
    """Metadata about the transcription."""

    duration: float  # total audio duration in seconds
    language: Optional[str] = None
    speakers: List[str] = field(default_factory=list)
    model_version: str = "vibevoice-asr"
    aligned: bool = True  # whether Qwen alignment was applied

    def to_dict(self) -> dict:
        return {
            "duration": self.duration,
            "language": self.language,
            "speakers": self.speakers,
            "model_version": self.model_version,
            "aligned": self.aligned,
        }


@dataclass
class TranscriptionResult:
    """Complete transcription result."""

    segments: List[Segment]
    metadata: TranscriptionMetadata

    def to_dict(self) -> dict:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "metadata": self.metadata.to_dict(),
        }

    def to_srt(self, include_speaker: bool = True) -> str:
        """Convert segments to SRT format."""
        from chalna.srt_utils import segments_to_srt
        return segments_to_srt(self.segments, include_speaker=include_speaker)

    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
