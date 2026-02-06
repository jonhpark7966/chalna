"""
Chalna Pipeline - VibeVoice API + Qwen Forced Alignment.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, List, Optional

import torch

from chalna.exceptions import (
    CodexAPIError,
    CodexRateLimitError,
    EmptyTranscriptionError,
    OutOfMemoryError,
    TempFileError,
    VibevoiceAPIError,
)
from chalna.models import IntermediateResults, Segment, TranscriptionMetadata, TranscriptionResult
from chalna.settings import settings
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
        use_llm_refinement: bool = True,
        aligner_path: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    ):
        """
        Initialize the pipeline.

        Args:
            device: Device to use for aligner (cuda, cpu, mps, xpu, auto)
            dtype: Data type for aligner model (None for auto)
            use_alignment: Whether to use Qwen forced alignment
            use_llm_refinement: Whether to use LLM-based subtitle refinement (requires Codex CLI)
            aligner_path: Path or HuggingFace ID for Qwen aligner model
        """
        self.device = self._resolve_device(device)
        self.dtype = dtype or self._resolve_dtype()
        self.use_alignment = use_alignment
        self.use_llm_refinement = use_llm_refinement
        self.aligner_path = aligner_path

        # Lazy loading (aligner only)
        self._aligner = None

        # Auto-unload settings
        self._auto_unload = False  # Keep aligner loaded for fast subsequent requests

        # Results storage - intermediate stages
        self._last_alignment_log = []
        self._pre_alignment_segments = None  # Alias for _raw_segments

        # Stage-specific intermediate results (always stored)
        self._raw_segments: Optional[List[Segment]] = None  # Stage 1: VibeVoice raw
        self._aligned_segments: Optional[List[Segment]] = None  # Stage 2: After forced alignment
        self._refined_segments: Optional[List[Segment]] = None  # Stage 3: After LLM refinement (before re-align)
        self._refinement_log: Optional[List[dict]] = None  # LLM refinement operation log

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

    def unload(self, force: bool = False) -> None:
        """Unload aligner model from GPU memory.

        Args:
            force: If True, force full cleanup (kept for API compatibility)
        """
        if self._aligner is not None:
            del self._aligner
            self._aligner = None
            print("Qwen aligner unloaded.")

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def is_loaded(self) -> bool:
        """Check if aligner model is currently loaded."""
        return self._aligner is not None

    def set_auto_unload(self, enabled: bool, keep_processor: bool = True) -> None:
        """Configure auto-unload behavior.

        Args:
            enabled: Whether to unload aligner after each request
            keep_processor: Ignored (kept for API compatibility)
        """
        self._auto_unload = enabled

    def _call_vibevoice_api(
        self,
        audio_path: Path,
        context: Optional[str] = None,
        max_new_tokens: int = 65536,
    ) -> List[Segment]:
        """Call VibeVoice API to transcribe audio.

        Args:
            audio_path: Path to audio file
            context: Optional context/hotwords
            max_new_tokens: Maximum tokens for generation

        Returns:
            List of Segment objects from API response

        Raises:
            VibevoiceAPIError: If API call fails
        """
        import httpx

        url = f"{settings.vibevoice_api_url}/transcribe"

        try:
            with open(audio_path, "rb") as f:
                files = {"file": (audio_path.name, f, "application/octet-stream")}
                data = {}
                if context:
                    data["context"] = context
                if max_new_tokens != 65536:
                    data["max_new_tokens"] = str(max_new_tokens)

                with httpx.Client(timeout=600.0) as client:
                    response = client.post(url, files=files, data=data)

        except httpx.ConnectError as e:
            raise VibevoiceAPIError(
                f"Cannot connect to VibeVoice API at {settings.vibevoice_api_url}. "
                f"Is the server running?",
                cause=e,
            )
        except httpx.TimeoutException as e:
            raise VibevoiceAPIError(
                f"VibeVoice API request timed out ({settings.vibevoice_api_url})",
                cause=e,
            )
        except Exception as e:
            raise VibevoiceAPIError(
                f"VibeVoice API request failed: {e}",
                cause=e,
            )

        if response.status_code != 200:
            raise VibevoiceAPIError(
                f"VibeVoice API returned HTTP {response.status_code}: {response.text}"
            )

        try:
            result = response.json()
        except Exception as e:
            raise VibevoiceAPIError(
                f"Failed to parse VibeVoice API response as JSON",
                cause=e,
            )

        api_segments = result.get("segments", [])
        segments = []
        for i, item in enumerate(api_segments, start=1):
            text = item.get("text", "")
            # Skip environmental sounds and silence markers
            if text.startswith("[") and text.endswith("]"):
                continue
            segments.append(Segment(
                index=i,
                start_time=float(item.get("start_time", 0)),
                end_time=float(item.get("end_time", 0)),
                text=text,
                speaker_id=item.get("speaker_id"),
            ))

        # Re-index after filtering
        for i, seg in enumerate(segments, start=1):
            seg.index = i

        return segments

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
        max_new_tokens: int = 65536,
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

        # Load aligner model
        _progress("loading_models", 0.0)
        self._load_aligner()
        _progress("loading_models", 1.0)

        # Step 1: VibeVoice ASR (via API)
        _progress("transcribing", 0.0)
        segments = self._call_vibevoice_api(audio_path, context, max_new_tokens)
        _progress("transcribing", 1.0)

        # Check for empty transcription
        if not segments:
            raise EmptyTranscriptionError(audio_duration=audio_info.duration_seconds)

        # Store raw segments (Stage 1)
        self._raw_segments = [
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
        # Keep backward compatibility
        self._pre_alignment_segments = self._raw_segments

        # Step 2: Qwen Forced Alignment (optional)
        _progress("aligning", 0.0)
        if self.use_alignment and self._aligner and segments:
            if verbose:
                print("\nForced Alignment:")
                print("  [idx]   original_start → aligned_start (delta) | original_end → aligned_end (delta)")
                print("  " + "-" * 80)
            segments = self._run_alignment(audio_path, segments, verbose=verbose)
        _progress("aligning", 1.0)

        # Store aligned segments (Stage 2)
        self._aligned_segments = [
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

        # Step 3: LLM Refinement (optional)
        if self.use_llm_refinement:
            _progress("refining", 0.0)
            try:
                segments, self._refinement_log = self._run_llm_refinement(
                    segments=segments,
                    audio_path=audio_path,
                    context=context,
                    progress_callback=progress_callback,
                    verbose=verbose,
                )
                # Store refined segments (Stage 3 - before final re-alignment)
                self._refined_segments = [
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
            except (CodexAPIError, CodexRateLimitError) as e:
                if verbose:
                    print(f"\nLLM refinement skipped: {e.message}")
                self._refinement_log = [{"status": "skipped", "error": str(e)}]
            _progress("refining", 1.0)

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

        # Build intermediate results (thread-safe, per-request)
        intermediate = IntermediateResults(
            raw_segments=self._raw_segments,
            aligned_segments=self._aligned_segments,
            refined_segments=self._refined_segments,
            alignment_log=self._last_alignment_log,
            refinement_log=self._refinement_log,
        )

        # Auto-unload aligner to free GPU memory
        if self._auto_unload:
            self.unload()

        return TranscriptionResult(
            segments=segments,
            metadata=metadata,
            intermediate=intermediate,
        )

    def get_pre_alignment_segments(self) -> Optional[List[Segment]]:
        """Get segments before forced alignment was applied (alias for get_raw_segments)."""
        return self._pre_alignment_segments

    def get_alignment_log(self) -> List[dict]:
        """Get detailed log of alignment adjustments."""
        return self._last_alignment_log

    def get_raw_segments(self) -> Optional[List[Segment]]:
        """Get raw VibeVoice segments (Stage 1, before alignment)."""
        return self._raw_segments

    def get_aligned_segments(self) -> Optional[List[Segment]]:
        """Get segments after forced alignment (Stage 2)."""
        return self._aligned_segments

    def get_refined_segments(self) -> Optional[List[Segment]]:
        """Get segments after LLM refinement (Stage 3, before final re-alignment)."""
        return self._refined_segments

    def get_refinement_log(self) -> Optional[List[dict]]:
        """Get LLM refinement operation log."""
        return self._refinement_log

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

    def _run_llm_refinement(
        self,
        segments: List[Segment],
        audio_path: Path,
        context: Optional[str],
        progress_callback: Optional[Callable[[str, float], None]],
        verbose: bool = True,
    ) -> tuple:
        """
        Run LLM refinement and re-alignment for split segments.

        Args:
            segments: List of aligned segments
            audio_path: Path to audio file
            context: Optional context/script for reference
            progress_callback: Optional callback for progress updates
            verbose: Whether to print detailed logs

        Returns:
            (refined_segments, refinement_log)
        """
        from chalna.llm_refiner import refine_segments

        if verbose:
            print("\nLLM Refinement:")
            print("  " + "-" * 80)

        # Step 1: LLM refinement
        def refine_progress(stage: str, value: float):
            if progress_callback:
                # Refining takes first 50% of this stage
                progress_callback("refining", value * 0.5)

        output = refine_segments(
            segments=segments,
            context=context,
            progress_callback=refine_progress,
        )

        refined_segments = list(output.segments)  # Make mutable copy
        log = output.log
        origin_map = output.origin_map

        if verbose:
            # Print summary
            split_count = sum(1 for entry in log if entry.get("status") == "split")
            refined_count = sum(1 for entry in log if entry.get("status") == "refined")
            error_count = sum(1 for entry in log if entry.get("status") == "error")
            parse_error_count = sum(1 for entry in log if entry.get("status") == "parse_error")
            print(f"  Split: {split_count}, Refined: {refined_count}, Errors: {error_count}, Parse errors: {parse_error_count}")

        # Step 2: Re-align segments using origin_map and log
        # Build lookup: original_index -> log entry for quick access
        log_by_original_idx = {}
        for entry in log:
            orig_idx = entry.get("original_index")
            if orig_idx is not None:
                log_by_original_idx[orig_idx] = entry

        # Identify segments needing re-alignment (by new segment index)
        single_realign_indices = []  # For non-split segments that need realignment
        split_groups = []  # For split segments: (original_start, original_end, [new_indices], [texts])

        for new_idx, orig_idx in origin_map.items():
            entry = log_by_original_idx.get(orig_idx)
            if entry is None:
                continue

            if entry.get("status") == "split":
                # Collect all new indices for this split group
                new_indices = entry.get("new_segment_indices", [])
                if new_idx == new_indices[0]:  # Only process once per split group
                    split_groups.append({
                        "original_start": entry.get("original_start"),
                        "original_end": entry.get("original_end"),
                        "new_indices": new_indices,
                        "texts": entry.get("split_texts", []),
                        "speaker_id": refined_segments[new_indices[0]].speaker_id if new_indices else None,
                        "confidence": refined_segments[new_indices[0]].confidence if new_indices else 1.0,
                    })
            elif entry.get("needs_realignment"):
                single_realign_indices.append(new_idx)

        total_realign_count = len(single_realign_indices) + len(split_groups)

        if total_realign_count > 0 and self._aligner:
            if verbose:
                print(f"  Re-aligning: {len(single_realign_indices)} single segments, {len(split_groups)} split groups")

            if progress_callback:
                progress_callback("refining", 0.5)

            realign_done = 0

            # Re-align single segments
            for new_idx in single_realign_indices:
                seg = refined_segments[new_idx]
                try:
                    aligned = self._align_single_segment(seg, audio_path)
                    if aligned:
                        refined_segments[new_idx] = aligned
                except Exception as e:
                    log.append({
                        "new_segment_index": new_idx,
                        "status": "realign_failed",
                        "error": str(e),
                    })
                    if verbose:
                        print(f"    Re-align failed for segment {seg.index}: {e}")

                realign_done += 1
                if progress_callback:
                    progress_callback("refining", 0.5 + 0.5 * realign_done / total_realign_count)

            # Re-align split groups using full original audio slice
            for group in split_groups:
                try:
                    aligned_splits = self._align_split_segments(
                        audio_path=audio_path,
                        original_start=group["original_start"],
                        original_end=group["original_end"],
                        split_texts=group["texts"],
                        speaker_id=group["speaker_id"],
                        confidence=group["confidence"],
                    )
                    if aligned_splits:
                        for i, new_idx in enumerate(group["new_indices"]):
                            if i < len(aligned_splits):
                                refined_segments[new_idx] = aligned_splits[i]
                except Exception as e:
                    log.append({
                        "new_segment_indices": group["new_indices"],
                        "status": "split_realign_failed",
                        "error": str(e),
                    })
                    if verbose:
                        print(f"    Split re-align failed: {e}")

                realign_done += 1
                if progress_callback:
                    progress_callback("refining", 0.5 + 0.5 * realign_done / total_realign_count)

        # Step 3: Fix overlapping timestamps
        refined_segments = self._fix_overlapping_timestamps(refined_segments, verbose=verbose)

        # Re-index all segments
        for i, seg in enumerate(refined_segments, start=1):
            seg.index = i

        return refined_segments, log

    def _align_split_segments(
        self,
        audio_path: Path,
        original_start: float,
        original_end: float,
        split_texts: List[str],
        speaker_id: Optional[str],
        confidence: float,
    ) -> Optional[List[Segment]]:
        """
        Align split segments using word-level alignment on the full original audio slice.

        Instead of aligning each split part on its equal-duration placeholder,
        we align the entire combined text against the original audio and derive
        boundaries from word timestamps.

        Args:
            audio_path: Path to full audio file
            original_start: Start time of original segment
            original_end: End time of original segment
            split_texts: List of text parts after splitting
            speaker_id: Speaker ID for all parts
            confidence: Confidence score for all parts

        Returns:
            List of aligned Segment objects, or None if alignment fails
        """
        import subprocess

        if not self._aligner:
            return None

        original_duration = original_end - original_start
        if original_duration < 0.1:
            return None

        try:
            # Extract original segment audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

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

            # Combine all split texts for alignment
            combined_text = " ".join(split_texts)

            # Run word-level alignment on combined text
            results = self._aligner.align(
                audio=tmp_path,
                text=combined_text,
                language="Korean",
            )

            Path(tmp_path).unlink(missing_ok=True)

            if not results or len(results) == 0 or len(results[0]) == 0:
                return None

            words = results[0]

            # Find boundaries for each split text using character positions
            aligned_segments = []
            current_char_pos = 0
            current_word_idx = 0

            for i, text in enumerate(split_texts):
                text_no_space = text.replace(" ", "")
                text_len = len(text_no_space)

                if text_len == 0:
                    continue

                # Find start word for this text part
                start_word_idx = current_word_idx

                # Find end word by counting characters
                chars_counted = 0
                end_word_idx = start_word_idx

                while end_word_idx < len(words) and chars_counted < text_len:
                    word_text = words[end_word_idx].text.replace(" ", "")
                    chars_counted += len(word_text)
                    end_word_idx += 1

                if start_word_idx >= len(words):
                    break

                # Get timestamps
                start_time = original_start + words[start_word_idx].start_time
                end_time = original_start + words[min(end_word_idx - 1, len(words) - 1)].end_time

                # Ensure valid duration
                if end_time <= start_time:
                    end_time = start_time + 0.1

                # Clamp to original bounds
                start_time = max(start_time, original_start)
                end_time = min(end_time, original_end)

                aligned_segments.append(Segment(
                    index=i + 1,  # Will be re-indexed later
                    start_time=start_time,
                    end_time=end_time,
                    text=text,
                    speaker_id=speaker_id,
                    confidence=confidence,
                ))

                current_word_idx = end_word_idx

            return aligned_segments if aligned_segments else None

        except Exception:
            return None

    def _align_single_segment(
        self,
        segment: Segment,
        audio_path: Path,
    ) -> Optional[Segment]:
        """
        Run forced alignment on a single segment.

        Args:
            segment: Segment to align
            audio_path: Path to full audio file

        Returns:
            Aligned segment or None if alignment fails
        """
        import subprocess

        if not self._aligner:
            return None

        original_start = segment.start_time
        original_end = segment.end_time
        original_duration = original_end - original_start

        if original_duration < 0.1:
            return None

        try:
            # Extract segment audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

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
                text=segment.text,
                language="Korean",
            )

            Path(tmp_path).unlink(missing_ok=True)

            if not results or len(results) == 0 or len(results[0]) == 0:
                return None

            first_item = results[0][0]
            last_item = results[0][-1]

            new_start = original_start + first_item.start_time
            new_end = original_start + last_item.end_time

            # Validate duration
            if new_end - new_start < 0.05:
                return None

            # Tighten to original bounds
            new_start = max(new_start, original_start)
            new_end = min(new_end, original_end)

            return Segment(
                index=segment.index,
                start_time=new_start,
                end_time=new_end,
                text=segment.text,
                speaker_id=segment.speaker_id,
                confidence=segment.confidence,
            )

        except Exception:
            return None

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences based on sentence-ending punctuation and commas.

        Handles Korean, English, and mixed text.

        Split criteria:
        1. Always split on sentence-ending punctuation: . ? ! 。？！
        2. Split on comma only if:
           - Text before comma >= 8 characters
           - Text after comma >= 8 characters
           - No other comma within 15 characters before (not a list)
        """
        import re

        # First, split on sentence-ending punctuation
        # Split on . ? ! 。？！ followed by space or end of string
        # But preserve the punctuation with the sentence
        pattern = r'([.?!。？！]+)\s*'

        parts = re.split(pattern, text)

        sentences = []
        current = ""

        for i, part in enumerate(parts):
            if not part:
                continue
            # Check if this part is punctuation
            if re.match(r'^[.?!。？！]+$', part):
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                current += part

        # Add remaining text
        if current.strip():
            sentences.append(current.strip())

        # Now, further split on commas with criteria
        final_sentences = []
        for sentence in sentences:
            split_result = self._split_on_comma(sentence)
            final_sentences.extend(split_result)

        # Filter out very short segments (likely noise)
        final_sentences = [s for s in final_sentences if len(s) > 1]

        return final_sentences

    def _split_on_comma(self, text: str) -> List[str]:
        """
        Split text on commas with smart criteria.

        Only split if:
        1. Text before comma >= 8 characters
        2. Text after comma >= 8 characters
        3. No other comma within 15 characters before (not a list pattern)
        """
        MIN_LENGTH = 8
        LIST_CHECK_DISTANCE = 15

        # Find all comma positions
        comma_positions = [i for i, c in enumerate(text) if c == ',']

        if not comma_positions:
            return [text]

        # Determine which commas are valid split points
        valid_splits = []

        for pos in comma_positions:
            before = text[:pos]
            after = text[pos + 1:]

            # Check minimum length criteria
            if len(before.strip()) < MIN_LENGTH or len(after.strip()) < MIN_LENGTH:
                continue

            # Check if there's another comma within LIST_CHECK_DISTANCE before this one
            is_list_pattern = False
            check_start = max(0, pos - LIST_CHECK_DISTANCE)
            preceding_text = text[check_start:pos]
            if ',' in preceding_text:
                is_list_pattern = True

            if not is_list_pattern:
                valid_splits.append(pos)

        if not valid_splits:
            return [text]

        # Split at valid positions
        result = []
        last_pos = 0

        for pos in valid_splits:
            segment = text[last_pos:pos + 1].strip()  # Include comma in previous segment
            if segment:
                result.append(segment)
            last_pos = pos + 1

        # Add remaining text
        remaining = text[last_pos:].strip()
        if remaining:
            result.append(remaining)

        return result

    def _find_sentence_boundaries(
        self,
        alignment_results: List,
        sentences: List[str],
        original_text: str,
    ) -> List[tuple]:
        """
        Find timestamp boundaries for each sentence using word-level alignment.

        Strategy: Use character positions from original text to map to aligned words.

        Args:
            alignment_results: Word-level alignment from Qwen aligner
            sentences: List of sentences to find boundaries for
            original_text: Original full text (for position tracking)

        Returns:
            List of (start_time, end_time, sentence_text) tuples
        """
        if not alignment_results or not alignment_results[0]:
            return []

        words = alignment_results[0]
        if not words:
            return []

        # Build character-to-word mapping from aligned results
        # Each word has .text, .start_time, .end_time
        char_to_word_idx = []
        for idx, word in enumerate(words):
            for _ in word.text:
                char_to_word_idx.append(idx)

        aligned_text = "".join(w.text for w in words)
        aligned_len = len(aligned_text)

        if aligned_len == 0:
            return []

        # Remove spaces from original text for alignment matching
        original_no_space = original_text.replace(" ", "")

        # Build mapping: original char position -> aligned char position
        # This handles cases where aligned text might have slight differences
        def fuzzy_char_position(orig_pos: int, search_char: str) -> int:
            """Find approximate position in aligned text for a character."""
            # Direct ratio mapping
            ratio = orig_pos / max(len(original_no_space), 1)
            approx_pos = int(ratio * aligned_len)
            return min(approx_pos, aligned_len - 1)

        boundaries = []
        current_orig_pos = 0  # Position in original_no_space

        for sentence in sentences:
            sentence_no_space = sentence.replace(" ", "")
            sentence_len = len(sentence_no_space)

            if sentence_len == 0:
                continue

            # Find sentence start position in original text
            sent_start_in_orig = original_no_space.find(sentence_no_space, current_orig_pos)
            if sent_start_in_orig == -1:
                # Try to find with first few chars
                search_len = min(5, sentence_len)
                sent_start_in_orig = original_no_space.find(sentence_no_space[:search_len], current_orig_pos)

            if sent_start_in_orig == -1:
                # Cannot find, use ratio-based estimation
                sent_start_in_orig = current_orig_pos

            sent_end_in_orig = sent_start_in_orig + sentence_len

            # Map to aligned text positions using ratio
            start_ratio = sent_start_in_orig / max(len(original_no_space), 1)
            end_ratio = sent_end_in_orig / max(len(original_no_space), 1)

            aligned_start_pos = int(start_ratio * aligned_len)
            aligned_end_pos = int(end_ratio * aligned_len)

            # Clamp to valid range
            aligned_start_pos = max(0, min(aligned_start_pos, aligned_len - 1))
            aligned_end_pos = max(aligned_start_pos + 1, min(aligned_end_pos, aligned_len))

            # Get word indices
            if aligned_start_pos < len(char_to_word_idx) and aligned_end_pos - 1 < len(char_to_word_idx):
                start_word_idx = char_to_word_idx[aligned_start_pos]
                end_word_idx = char_to_word_idx[min(aligned_end_pos - 1, len(char_to_word_idx) - 1)]

                start_time = words[start_word_idx].start_time
                end_time = words[end_word_idx].end_time

                # Ensure valid duration
                if end_time > start_time:
                    boundaries.append((start_time, end_time, sentence))

            current_orig_pos = sent_end_in_orig

        return boundaries

    def _run_alignment(
        self,
        audio_path: Path,
        segments: List[Segment],
        verbose: bool = True,
    ) -> List[Segment]:
        """
        Run Qwen forced alignment to refine timestamps and split sentences.

        Strategy:
        1. For each segment, detect sentence boundaries
        2. Run word-level forced alignment
        3. Split multi-sentence segments using word timestamps
        4. Apply tighten-only constraint for timestamp refinement
        5. If alignment fails, keep original segment
        """
        import subprocess

        aligned_segments = []
        alignment_log = []
        segment_index = 1  # Will be reassigned after processing

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

                # Check for valid alignment results
                if not results or len(results) == 0 or len(results[0]) == 0:
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
                    Path(tmp_path).unlink(missing_ok=True)
                    continue

                # Split into sentences
                sentences = self._split_into_sentences(seg.text)

                # If only one sentence or split failed, use simple alignment
                if len(sentences) <= 1:
                    # Simple single-segment alignment (original logic)
                    first_item = results[0][0]
                    last_item = results[0][-1]

                    new_start = original_start + first_item.start_time
                    new_end = original_start + last_item.end_time

                    aligned_duration = new_end - new_start
                    if aligned_duration < 0.1:
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
                    else:
                        # Tighten-only check
                        start_tighter = new_start >= original_start - 0.01
                        end_tighter = new_end <= original_end + 0.01

                        if start_tighter and end_tighter:
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
                            aligned_segments.append(seg)
                            log_entry = {
                                "index": seg.index,
                                "status": "rejected_expand",
                                "original": {"start": original_start, "end": original_end},
                                "attempted": {"start": new_start, "end": new_end},
                                "text": seg.text[:50] + "..." if len(seg.text) > 50 else seg.text,
                            }
                            alignment_log.append(log_entry)

                            if verbose:
                                delta_s = new_start - original_start
                                delta_e = new_end - original_end
                                print(f"  [{seg.index:3d}] REJECTED: would expand "
                                      f"(start {delta_s:+.2f}s, end {delta_e:+.2f}s) - keeping original")
                else:
                    # Multiple sentences - find boundaries and split
                    boundaries = self._find_sentence_boundaries(results, sentences, seg.text)

                    if not boundaries or len(boundaries) != len(sentences):
                        # Boundary detection failed, fall back to original segment
                        aligned_segments.append(seg)
                        log_entry = {
                            "index": seg.index,
                            "status": "sentence_split_failed",
                            "original": {"start": original_start, "end": original_end},
                            "sentences_detected": len(sentences),
                            "boundaries_found": len(boundaries) if boundaries else 0,
                            "text": seg.text[:50] + "..." if len(seg.text) > 50 else seg.text,
                        }
                        alignment_log.append(log_entry)

                        if verbose:
                            print(f"  [{seg.index:3d}] SPLIT_FAILED: {len(sentences)} sentences, "
                                  f"{len(boundaries) if boundaries else 0} boundaries - keeping original")
                    else:
                        # Successfully split - create new segments
                        if verbose:
                            print(f"  [{seg.index:3d}] SPLIT: {len(sentences)} sentences")

                        for i, (rel_start, rel_end, sentence_text) in enumerate(boundaries):
                            # Convert to absolute timestamps
                            abs_start = original_start + rel_start
                            abs_end = original_start + rel_end

                            # Validate duration
                            if abs_end - abs_start < 0.05:
                                continue

                            # Tighten to original bounds
                            abs_start = max(abs_start, original_start)
                            abs_end = min(abs_end, original_end)

                            new_seg = Segment(
                                index=seg.index,  # Will be re-indexed later
                                start_time=abs_start,
                                end_time=abs_end,
                                text=sentence_text,
                                speaker_id=seg.speaker_id,
                                confidence=seg.confidence,
                            )
                            aligned_segments.append(new_seg)

                            log_entry = {
                                "index": seg.index,
                                "status": "split",
                                "split_index": i + 1,
                                "split_total": len(boundaries),
                                "original": {"start": original_start, "end": original_end},
                                "aligned": {"start": abs_start, "end": abs_end},
                                "text": sentence_text[:50] + "..." if len(sentence_text) > 50 else sentence_text,
                            }
                            alignment_log.append(log_entry)

                            if verbose:
                                print(f"      [{i+1}/{len(boundaries)}] {abs_start:7.2f}→{abs_end:7.2f} | "
                                      f"{sentence_text[:40]}{'...' if len(sentence_text) > 40 else ''}")

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

        # Fix overlapping timestamps using midpoint interpolation
        aligned_segments = self._fix_overlapping_timestamps(aligned_segments, verbose=verbose)

        # Re-index all segments
        for i, seg in enumerate(aligned_segments, start=1):
            seg.index = i

        # Store alignment log for later access
        self._last_alignment_log = alignment_log

        return aligned_segments

    def _fix_overlapping_timestamps(
        self,
        segments: List[Segment],
        verbose: bool = True,
    ) -> List[Segment]:
        """
        Fix overlapping timestamps between consecutive segments using midpoint interpolation.

        When segment N's end_time > segment N+1's start_time:
        - Calculate midpoint = (end_time + start_time) / 2
        - Set segment N's end_time = midpoint
        - Set segment N+1's start_time = midpoint
        """
        if len(segments) <= 1:
            return segments

        fixed_count = 0

        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]

            if current.end_time > next_seg.start_time:
                # Overlap detected - use midpoint interpolation
                midpoint = (current.end_time + next_seg.start_time) / 2

                # Update timestamps
                current.end_time = midpoint
                next_seg.start_time = midpoint
                fixed_count += 1

        if verbose and fixed_count > 0:
            print(f"\n  Fixed {fixed_count} overlapping timestamp(s) using midpoint interpolation")

        return segments
