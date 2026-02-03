"""
Chalna Pipeline - VibeVoice ASR + Qwen Forced Alignment.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

import torch

from chalna.exceptions import (
    EmptyTranscriptionError,
    ModelDownloadError,
    ModelLoadError,
    OutOfMemoryError,
    TempFileError,
)
from chalna.models import Segment, TranscriptionMetadata, TranscriptionResult
from chalna.validation import (
    check_disk_space,
    estimate_temp_space_required,
    validate_audio_file,
)


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

        # Results storage
        self._last_alignment_log = []
        self._pre_alignment_segments = None

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
        """Load VibeVoice model and processor.

        Raises:
            OutOfMemoryError: If GPU/CPU memory is exhausted.
            ModelLoadError: If model loading fails.
            ModelDownloadError: If model download fails.
        """
        if self._vibevoice_model is not None:
            return

        # Add external/VibeVoice to path
        vibevoice_path = Path(__file__).parent.parent.parent / "external" / "VibeVoice"
        if vibevoice_path.exists():
            sys.path.insert(0, str(vibevoice_path))

        try:
            from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
            from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
        except ImportError as e:
            raise ModelLoadError(
                model_name="VibeVoice",
                reason="VibeVoice module not found",
                cause=e,
            )

        print(f"Loading VibeVoice model from {self.vibevoice_path}...")

        # Try attention implementations in order: sdpa > eager
        last_error = None
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
            except torch.cuda.OutOfMemoryError as e:
                raise OutOfMemoryError(memory_type="GPU", cause=e)
            except MemoryError as e:
                raise OutOfMemoryError(memory_type="CPU", cause=e)
            except OSError as e:
                # Often indicates download failure
                if "Connection" in str(e) or "HTTP" in str(e):
                    raise ModelDownloadError(model_name=self.vibevoice_path, cause=e)
                last_error = e
                if attn_impl == "eager":
                    raise ModelLoadError(
                        model_name=self.vibevoice_path,
                        reason=str(e),
                        cause=e,
                    )
            except Exception as e:
                print(f"Failed to load with {attn_impl}: {e}")
                last_error = e
                if attn_impl == "eager":
                    raise ModelLoadError(
                        model_name=self.vibevoice_path,
                        reason=str(e),
                        cause=e,
                    )

        try:
            self._vibevoice_processor = VibeVoiceASRProcessor.from_pretrained(
                self.vibevoice_path,
                language_model_pretrained_name="Qwen/Qwen2.5-7B",
            )
        except Exception as e:
            raise ModelLoadError(
                model_name="VibeVoiceASRProcessor",
                reason=str(e),
                cause=e,
            )

        print("VibeVoice model loaded.")

    def _load_aligner(self):
        """Load Qwen forced aligner.

        Raises:
            OutOfMemoryError: If GPU/CPU memory is exhausted.
        """
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
        except torch.cuda.OutOfMemoryError as e:
            raise OutOfMemoryError(memory_type="GPU", cause=e)
        except MemoryError as e:
            raise OutOfMemoryError(memory_type="CPU", cause=e)
        except Exception as e:
            print(f"Warning: Failed to load aligner: {e}. Alignment will be skipped.")
            self.use_alignment = False

    def transcribe(
        self,
        audio_path: str | Path,
        context: Optional[str] = None,
        language: Optional[str] = None,
        max_new_tokens: int = 32768,
        verbose: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            context: Optional context/hotwords for better accuracy
            language: Language hint (ko, en, ja, zh, etc.)
            max_new_tokens: Maximum tokens for generation
            verbose: Whether to print detailed alignment logs
            progress_callback: Optional callback(stage, progress) for progress updates

        Returns:
            TranscriptionResult with segments and metadata

        Raises:
            AudioTooLongError: If audio exceeds 1 hour.
            UnsupportedFormatError: If format is not supported.
            CorruptedFileError: If file is corrupted.
            OutOfMemoryError: If GPU/CPU memory is exhausted.
            EmptyTranscriptionError: If no speech is detected.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        def _progress(stage: str, value: float):
            if progress_callback:
                progress_callback(stage, value)

        # Step 0: Validate audio file (duration, format, integrity)
        _progress("validating", 0.0)
        audio_info = validate_audio_file(audio_path)
        _progress("validating", 1.0)

        # Check disk space for temp files
        required_mb = estimate_temp_space_required(audio_info)
        check_disk_space(required_mb)

        # Load models
        _progress("loading_models", 0.0)
        self._load_vibevoice()
        _progress("loading_models", 0.5)
        self._load_aligner()
        _progress("loading_models", 1.0)

        # Step 1: VibeVoice ASR
        _progress("transcribing", 0.0)
        try:
            segments = self._run_vibevoice(audio_path, context, max_new_tokens)
        except torch.cuda.OutOfMemoryError as e:
            raise OutOfMemoryError(memory_type="GPU", cause=e)
        except MemoryError as e:
            raise OutOfMemoryError(memory_type="CPU", cause=e)
        _progress("transcribing", 1.0)

        # Check for empty transcription
        if not segments:
            raise EmptyTranscriptionError(audio_duration=audio_info.duration_seconds)

        # Store pre-alignment segments (deep copy)
        self._pre_alignment_segments = [
            Segment(
                index=s.index,
                start_time=s.start_time,
                end_time=s.end_time,
                text=s.text,
                speaker_id=s.speaker_id,
                confidence=s.confidence,
            )
            for s in segments
        ]

        # Step 2: Qwen Forced Alignment (optional)
        _progress("aligning", 0.0)
        if self.use_alignment and self._aligner and segments:
            if verbose:
                print("\nForced Alignment:")
                print("  [idx]   original_start → aligned_start (delta) | original_end → aligned_end (delta)")
                print("  " + "-" * 80)
            segments = self._run_alignment(audio_path, segments, verbose=verbose)
        _progress("aligning", 1.0)

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

    def get_pre_alignment_segments(self) -> Optional[List[Segment]]:
        """Get segments before forced alignment was applied."""
        return self._pre_alignment_segments

    def get_alignment_log(self) -> List[dict]:
        """Get detailed log of alignment adjustments."""
        return self._last_alignment_log

    def align_segments(
        self,
        audio_path: str | Path,
        segments: List[Segment],
        verbose: bool = True,
    ) -> List[Segment]:
        """
        Run forced alignment on provided segments without running ASR.

        Args:
            audio_path: Path to audio file
            segments: Pre-existing segments to align
            verbose: Whether to print detailed logs

        Returns:
            Aligned segments
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load aligner only
        self._load_aligner()

        if not self._aligner:
            print("Aligner not available, returning original segments")
            return segments

        if verbose:
            print("\nForced Alignment:")
            print("  [idx]   original_start → aligned_start (delta) | original_end → aligned_end (delta)")
            print("  " + "-" * 80)

        return self._run_alignment(audio_path, segments, verbose=verbose)

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

    def _run_alignment(
        self,
        audio_path: Path,
        segments: List[Segment],
        verbose: bool = True,
    ) -> List[Segment]:
        """
        Run Qwen forced alignment to refine timestamps.

        Strategy: No buffer, tighten-only
        - Extract exact segment audio (no buffer)
        - Only apply alignment if it makes timestamps tighter:
          - aligned_start >= original_start (starts later or same)
          - aligned_end <= original_end (ends earlier or same)
        - This reduces overlap caused by VibeVoice's loose timestamps
        """
        import subprocess

        aligned_segments = []
        alignment_log = []

        for seg in segments:
            if not seg.text.strip():
                aligned_segments.append(seg)
                continue

            original_start = seg.start_time
            original_end = seg.end_time
            original_duration = original_end - original_start

            # Skip very short segments
            if original_duration < 0.1:
                aligned_segments.append(seg)
                continue

            try:
                # Extract exact segment audio (no buffer)
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp_path = tmp.name
                except OSError as e:
                    raise TempFileError(operation="create", cause=e)

                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", str(audio_path),
                    "-ss", str(original_start),
                    "-t", str(original_duration),
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

                    # Convert to absolute timestamps
                    new_start = original_start + first_item.start_time
                    new_end = original_start + last_item.end_time

                    # Check for valid alignment (positive duration)
                    aligned_duration = new_end - new_start
                    if aligned_duration < 0.1:
                        # Invalid alignment, keep original
                        aligned_segments.append(seg)
                        log_entry = {
                            "index": seg.index,
                            "status": "invalid_duration",
                            "original": {"start": original_start, "end": original_end},
                            "attempted": {"start": new_start, "end": new_end},
                            "text": seg.text[:50] + "..." if len(seg.text) > 50 else seg.text,
                        }
                        alignment_log.append(log_entry)

                        if verbose:
                            print(f"  [{seg.index:3d}] INVALID: duration={aligned_duration:.2f}s - keeping original")
                        continue

                    # Tighten-only check:
                    # - Start should stay same or move later (tighter)
                    # - End should stay same or move earlier (tighter)
                    start_tighter = new_start >= original_start - 0.01  # 10ms tolerance
                    end_tighter = new_end <= original_end + 0.01  # 10ms tolerance

                    if start_tighter and end_tighter:
                        # Good alignment - apply it
                        delta_start = new_start - original_start
                        delta_end = new_end - original_end

                        aligned_segments.append(Segment(
                            index=seg.index,
                            start_time=new_start,
                            end_time=new_end,
                            text=seg.text,
                            speaker_id=seg.speaker_id,
                            confidence=seg.confidence,
                        ))

                        log_entry = {
                            "index": seg.index,
                            "status": "aligned",
                            "original": {"start": original_start, "end": original_end},
                            "aligned": {"start": new_start, "end": new_end},
                            "delta": {"start": delta_start, "end": delta_end},
                            "text": seg.text[:50] + "..." if len(seg.text) > 50 else seg.text,
                        }
                        alignment_log.append(log_entry)

                        if verbose:
                            print(f"  [{seg.index:3d}] {original_start:7.2f}→{new_start:7.2f} ({delta_start:+.2f}s) | "
                                  f"{original_end:7.2f}→{new_end:7.2f} ({delta_end:+.2f}s)")
                    else:
                        # Alignment would expand timestamps - reject
                        aligned_segments.append(seg)
                        log_entry = {
                            "index": seg.index,
                            "status": "rejected_expand",
                            "original": {"start": original_start, "end": original_end},
                            "attempted": {"start": new_start, "end": new_end},
                            "reason": f"start_tighter={start_tighter}, end_tighter={end_tighter}",
                            "text": seg.text[:50] + "..." if len(seg.text) > 50 else seg.text,
                        }
                        alignment_log.append(log_entry)

                        if verbose:
                            delta_s = new_start - original_start
                            delta_e = new_end - original_end
                            print(f"  [{seg.index:3d}] REJECTED: would expand "
                                  f"(start {delta_s:+.2f}s, end {delta_e:+.2f}s) - keeping original")
                else:
                    aligned_segments.append(seg)
                    log_entry = {
                        "index": seg.index,
                        "status": "no_result",
                        "original": {"start": original_start, "end": original_end},
                        "text": seg.text[:50] + "..." if len(seg.text) > 50 else seg.text,
                    }
                    alignment_log.append(log_entry)

                    if verbose:
                        print(f"  [{seg.index:3d}] No alignment result - keeping original")

                # Cleanup
                Path(tmp_path).unlink(missing_ok=True)

            except Exception as e:
                aligned_segments.append(seg)
                log_entry = {
                    "index": seg.index,
                    "status": "failed",
                    "error": str(e),
                    "original": {"start": original_start, "end": original_end},
                    "text": seg.text[:50] + "..." if len(seg.text) > 50 else seg.text,
                }
                alignment_log.append(log_entry)

                if verbose:
                    print(f"  [{seg.index:3d}] FAILED: {e}")

        # Store alignment log for later access
        self._last_alignment_log = alignment_log

        return aligned_segments
