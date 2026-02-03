"""
Chalna Pipeline - VibeVoice ASR + Qwen Forced Alignment.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import torch

from chalna.models import Segment, TranscriptionMetadata, TranscriptionResult


class ChalnaPipeline:
    """
    Main pipeline for transcription with forced alignment.

    Workflow:
    1. VibeVoice ASR: Generate transcription with speaker diarization
    2. Qwen Forced Alignment: Refine timestamps for each segment
    """

    def __init__(
        self,
        device: str = "auto",
        dtype: Optional[torch.dtype] = None,
        use_alignment: bool = True,
        vibevoice_path: str = "microsoft/VibeVoice-ASR",
        aligner_path: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    ):
        """
        Initialize the pipeline.

        Args:
            device: Device to use (cuda, cpu, mps, xpu, auto)
            dtype: Data type for models (None for auto)
            use_alignment: Whether to use Qwen forced alignment
            vibevoice_path: Path or HuggingFace ID for VibeVoice model
            aligner_path: Path or HuggingFace ID for Qwen aligner model
        """
        self.device = self._resolve_device(device)
        self.dtype = dtype or self._resolve_dtype()
        self.use_alignment = use_alignment
        self.vibevoice_path = vibevoice_path
        self.aligner_path = aligner_path

        # Lazy loading
        self._vibevoice_model = None
        self._vibevoice_processor = None
        self._aligner = None

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                return "xpu"
            else:
                return "cpu"
        return device

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve dtype based on device."""
        if self.device == "cuda":
            return torch.bfloat16
        return torch.float32

    def _load_vibevoice(self):
        """Load VibeVoice model and processor."""
        if self._vibevoice_model is not None:
            return

        # Add external/VibeVoice to path
        vibevoice_path = Path(__file__).parent.parent.parent / "external" / "VibeVoice"
        if vibevoice_path.exists():
            sys.path.insert(0, str(vibevoice_path))

        from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

        print(f"Loading VibeVoice model from {self.vibevoice_path}...")

        # Try attention implementations in order: sdpa > eager
        for attn_impl in ["sdpa", "eager"]:
            try:
                self._vibevoice_model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                    self.vibevoice_path,
                    dtype=self.dtype,
                    device_map=self.device,
                    attn_implementation=attn_impl,
                    trust_remote_code=True,
                )
                print(f"Using attention implementation: {attn_impl}")
                break
            except Exception as e:
                print(f"Failed to load with {attn_impl}: {e}")
                if attn_impl == "eager":
                    raise

        self._vibevoice_processor = VibeVoiceASRProcessor.from_pretrained(
            self.vibevoice_path,
            language_model_pretrained_name="Qwen/Qwen2.5-7B",
        )

        print("VibeVoice model loaded.")

    def _load_aligner(self):
        """Load Qwen forced aligner."""
        if self._aligner is not None:
            return

        if not self.use_alignment:
            return

        try:
            from qwen_asr import Qwen3ForcedAligner

            print(f"Loading Qwen aligner from {self.aligner_path}...")

            self._aligner = Qwen3ForcedAligner.from_pretrained(
                self.aligner_path,
                dtype=self.dtype,
                device_map=self.device,
            )

            print("Qwen aligner loaded.")
        except ImportError:
            print("Warning: qwen_asr not installed. Alignment will be skipped.")
            self.use_alignment = False
        except Exception as e:
            print(f"Warning: Failed to load aligner: {e}. Alignment will be skipped.")
            self.use_alignment = False

    def transcribe(
        self,
        audio_path: str | Path,
        context: Optional[str] = None,
        language: Optional[str] = None,
        max_new_tokens: int = 32768,
    ) -> TranscriptionResult:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            context: Optional context/hotwords for better accuracy
            language: Language hint (ko, en, ja, zh, etc.)
            max_new_tokens: Maximum tokens for generation

        Returns:
            TranscriptionResult with segments and metadata
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load models
        self._load_vibevoice()
        self._load_aligner()

        # Step 1: VibeVoice ASR
        segments = self._run_vibevoice(audio_path, context, max_new_tokens)

        # Step 2: Qwen Forced Alignment (optional)
        if self.use_alignment and self._aligner and segments:
            segments = self._run_alignment(audio_path, segments)

        # Extract metadata
        speakers = list(set(s.speaker_id for s in segments if s.speaker_id))
        duration = max((s.end_time for s in segments), default=0.0)

        metadata = TranscriptionMetadata(
            duration=duration,
            language=language,
            speakers=speakers,
            model_version="vibevoice-asr",
            aligned=self.use_alignment and self._aligner is not None,
        )

        return TranscriptionResult(segments=segments, metadata=metadata)

    def _run_vibevoice(
        self,
        audio_path: Path,
        context: Optional[str],
        max_new_tokens: int,
    ) -> List[Segment]:
        """Run VibeVoice ASR."""
        # Prepare input
        inputs = self._vibevoice_processor(
            str(audio_path),
            context_info=context,
            return_tensors="pt",
        )

        # Move to device
        inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        # Generate
        with torch.no_grad():
            outputs = self._vibevoice_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )

        # Decode
        raw_output = self._vibevoice_processor.batch_decode(
            outputs,
            skip_special_tokens=True,
        )[0]

        # Parse output - try multiple methods
        parsed = self._parse_transcription(raw_output)

        # Convert to Segment objects
        segments = []
        for i, item in enumerate(parsed, start=1):
            # Handle different key naming conventions
            start_time = item.get("start_time") or item.get("Start time") or item.get("Start") or 0
            end_time = item.get("end_time") or item.get("End time") or item.get("End") or 0
            text = item.get("text") or item.get("Content") or ""

            # Handle speaker ID - check for various key names and handle 0 as valid
            speaker = None
            for key in ["speaker_id", "Speaker ID", "Speaker"]:
                if key in item and item[key] is not None:
                    speaker = item[key]
                    break

            # Skip environmental sounds and silence markers
            if text.startswith("[") and text.endswith("]"):
                continue

            segments.append(Segment(
                index=i,
                start_time=float(start_time),
                end_time=float(end_time),
                text=text,
                speaker_id=f"Speaker {speaker}" if speaker is not None else None,
            ))

        # Re-index after filtering
        for i, seg in enumerate(segments, start=1):
            seg.index = i

        return segments

    def _parse_transcription(self, raw_output: str) -> List[dict]:
        """Parse transcription output from VibeVoice."""
        import re

        # Try processor's post_process first
        try:
            parsed = self._vibevoice_processor.post_process_transcription(raw_output)
            if parsed:
                return parsed
        except Exception as e:
            print(f"post_process_transcription failed: {e}")

        # Manual parsing - find JSON array in output
        # Look for content after "assistant" tag
        if "assistant" in raw_output:
            raw_output = raw_output.split("assistant")[-1].strip()

        # Try to find JSON array
        json_match = re.search(r'\[.*\]', raw_output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")

        # Try parsing as JSONL (one object per line)
        results = []
        for line in raw_output.split('\n'):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        if results:
            return results

        print(f"Failed to parse transcription output. Raw length: {len(raw_output)}")
        return []

    def _run_alignment(self, audio_path: Path, segments: List[Segment]) -> List[Segment]:
        """Run Qwen forced alignment to refine timestamps."""
        import subprocess

        aligned_segments = []

        for seg in segments:
            if not seg.text.strip():
                aligned_segments.append(seg)
                continue

            try:
                # Extract audio segment
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name

                buffer = 1.5  # seconds before and after
                duration = seg.end_time - seg.start_time + buffer * 2
                start_offset = max(0, seg.start_time - buffer)
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", str(audio_path),
                    "-ss", str(start_offset),
                    "-t", str(duration),
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    tmp_path,
                ], capture_output=True, check=True)

                # Run alignment
                results = self._aligner.align(
                    audio=tmp_path,
                    text=seg.text,
                    language="Korean",
                )

                # Update timestamps
                if results and len(results) > 0 and len(results[0]) > 0:
                    first_item = results[0][0]
                    last_item = results[0][-1]

                    # Adjust relative to segment start
                    offset = start_offset
                    new_start = offset + first_item.start_time
                    new_end = offset + last_item.end_time

                    aligned_segments.append(Segment(
                        index=seg.index,
                        start_time=new_start,
                        end_time=new_end,
                        text=seg.text,
                        speaker_id=seg.speaker_id,
                        confidence=seg.confidence,
                    ))
                else:
                    aligned_segments.append(seg)

                # Cleanup
                Path(tmp_path).unlink(missing_ok=True)

            except Exception as e:
                print(f"Warning: Alignment failed for segment {seg.index}: {e}")
                aligned_segments.append(seg)

        return aligned_segments
