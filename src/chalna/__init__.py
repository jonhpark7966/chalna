"""
Chalna (찰나) - SRT subtitle generation service.

VibeVoice ASR + Qwen Forced Alignment for accurate timestamps and speaker diarization.
"""

__version__ = "0.1.0"

from chalna.models import Segment, TranscriptionResult
from chalna.pipeline import ChalnaPipeline

__all__ = ["ChalnaPipeline", "Segment", "TranscriptionResult", "__version__"]
