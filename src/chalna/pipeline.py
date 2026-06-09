"""Chalna transcription pipeline backed by ElevenLabs Scribe v2."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from chalna.exceptions import (
    CodexAPIError,
    CodexRateLimitError,
    EmptyTranscriptionError,
)
from chalna.models import (
    IntermediateResults,
    ScribeOptions,
    Segment,
    TranscriptionMetadata,
    TranscriptionResult,
)
from chalna.scribe_adapter import scribe_response_to_segments
from chalna.scribe_cache import ScribeResponseCache, build_scribe_cache_metadata
from chalna.scribe_client import ScribeClient
from chalna.settings import settings
from chalna.validation import (
    check_disk_space,
    estimate_temp_space_required,
    validate_audio_file,
)


class ChalnaPipeline:
    """Main pipeline for Scribe transcription and optional LLM subtitle refinement."""

    def __init__(
        self,
        device: str = "auto",
        dtype: object = None,
        use_alignment: bool = True,
        use_llm_refinement: bool = True,
        aligner_path: str = "",
        scribe_client: Optional[ScribeClient] = None,
        scribe_cache: Optional[ScribeResponseCache] = None,
    ):
        # Kept for constructor compatibility. Qwen alignment is intentionally disabled.
        self.device = device
        self.dtype = dtype
        self.use_alignment = False
        self.use_llm_refinement = use_llm_refinement
        self.aligner_path = aligner_path

        self.scribe_client = scribe_client or ScribeClient()
        self.scribe_cache = scribe_cache or ScribeResponseCache(settings.scribe_cache_dir)

        self._auto_unload = False
        self._last_alignment_log: list[dict] = []
        self._pre_alignment_segments: Optional[list[Segment]] = None
        self._raw_segments: Optional[list[Segment]] = None
        self._aligned_segments: Optional[list[Segment]] = None
        self._refined_segments: Optional[list[Segment]] = None
        self._refinement_log: Optional[list[dict]] = None
        self._scribe_words_by_segment_index: dict[int, list[dict]] = {}
        self._last_scribe_response: Optional[dict] = None

    def unload(self, force: bool = False) -> None:
        """No-op kept for API compatibility; Scribe does not load local GPU models."""
        return None

    def is_loaded(self) -> bool:
        """Scribe does not keep local models loaded."""
        return False

    def set_auto_unload(self, enabled: bool, keep_processor: bool = True) -> None:
        """Keep API compatibility with the previous GPU model pipeline."""
        self._auto_unload = enabled

    def transcribe(
        self,
        audio_path: str | Path,
        context: Optional[str] = None,
        language: Optional[str] = None,
        max_new_tokens: int = 65536,
        verbose: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        scribe_options: Optional[ScribeOptions] = None,
    ) -> TranscriptionResult:
        """Transcribe an audio/video file to Chalna segments and subtitles."""
        del max_new_tokens  # Kept for compatibility with the previous local ASR API.

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._reset_request_state()
        options = scribe_options or ScribeOptions()

        def _progress(stage: str, value: float, **extra):
            if progress_callback:
                progress_callback(stage, value, **extra)

        _progress("validating", 0.0)
        audio_info = validate_audio_file(audio_path)
        _progress("validating", 1.0)

        required_mb = estimate_temp_space_required(audio_info)
        check_disk_space(required_mb)

        _progress("transcribing", 0.0)
        cache_metadata = build_scribe_cache_metadata(
            audio_path=audio_path,
            audio_info=audio_info,
            model_id=self.scribe_client.model_id,
            language_code=language,
            options=options,
        )
        scribe_response = self.scribe_cache.get(cache_metadata)
        cache_hit = scribe_response is not None
        if scribe_response is None:
            scribe_response = self.scribe_client.transcribe(
                audio_path,
                language_code=language,
                options=options,
            )
            self.scribe_cache.put(cache_metadata, scribe_response)

        self._last_scribe_response = scribe_response
        adapter_result = scribe_response_to_segments(
            scribe_response,
            include_audio_events=options.tag_audio_events,
        )
        segments = adapter_result.segments
        self._scribe_words_by_segment_index = adapter_result.words_by_segment_index

        if not segments:
            raise EmptyTranscriptionError(audio_duration=audio_info.duration_seconds)

        self._raw_segments = self._clone_segments(segments)
        self._pre_alignment_segments = self._raw_segments
        _progress("transcribing", 1.0, cache_hit=cache_hit)

        if self.use_llm_refinement:
            _progress("refining", 0.0)
            try:
                segments, self._refinement_log = self._run_llm_refinement(
                    segments=segments,
                    context=context,
                    progress_callback=progress_callback,
                    verbose=verbose,
                )
                self._refined_segments = self._clone_segments(segments)
            except (CodexAPIError, CodexRateLimitError) as e:
                if verbose:
                    print(f"\nLLM refinement skipped: {e.message}")
                self._refinement_log = [{"status": "skipped", "error": str(e)}]
            _progress("refining", 1.0)

        segments = self._fix_overlapping_timestamps(segments, verbose=verbose)
        for i, segment in enumerate(segments, start=1):
            segment.index = i

        speakers = sorted({s.speaker_id for s in segments if s.speaker_id})
        result_language = adapter_result.language_code or language

        metadata = TranscriptionMetadata(
            duration=audio_info.duration_seconds,
            language=result_language,
            speakers=speakers,
            model_version=self.scribe_client.model_id,
            aligned=False,
            refined=self.use_llm_refinement and self._refined_segments is not None,
            timestamp_source=self.scribe_client.model_id,
        )

        intermediate = IntermediateResults(
            raw_segments=self._raw_segments,
            aligned_segments=None,
            refined_segments=self._refined_segments,
            chunk_raw_segments=None,
            alignment_log=[],
            refinement_log=self._refinement_log,
        )

        return TranscriptionResult(
            segments=segments,
            metadata=metadata,
            intermediate=intermediate,
        )

    def get_pre_alignment_segments(self) -> Optional[list[Segment]]:
        """Get segments before optional LLM refinement."""
        return self._pre_alignment_segments

    def get_alignment_log(self) -> list[dict]:
        """Qwen alignment is disabled; this always returns an empty log."""
        return self._last_alignment_log

    def get_raw_segments(self) -> Optional[list[Segment]]:
        """Get raw Scribe segments before optional LLM refinement."""
        return self._raw_segments

    def get_aligned_segments(self) -> Optional[list[Segment]]:
        """Qwen alignment is disabled; aligned segments are not produced."""
        return self._aligned_segments

    def get_refined_segments(self) -> Optional[list[Segment]]:
        """Get segments after LLM refinement."""
        return self._refined_segments

    def get_refinement_log(self) -> Optional[list[dict]]:
        """Get LLM refinement operation log."""
        return self._refinement_log

    def align_segments(
        self,
        audio_path: str | Path,
        segments: list[Segment],
        verbose: bool = True,
    ) -> list[Segment]:
        """Compatibility shim; Qwen forced alignment is no longer run."""
        del audio_path
        return self._fix_overlapping_timestamps(self._clone_segments(segments), verbose=verbose)

    def _reset_request_state(self) -> None:
        self._raw_segments = None
        self._aligned_segments = None
        self._refined_segments = None
        self._refinement_log = None
        self._last_alignment_log = []
        self._pre_alignment_segments = None
        self._scribe_words_by_segment_index = {}
        self._last_scribe_response = None

    def _run_llm_refinement(
        self,
        segments: list[Segment],
        context: Optional[str],
        progress_callback: Optional[Callable[[str, float], None]],
        verbose: bool = True,
    ) -> tuple[list[Segment], list[dict]]:
        """Run LLM refinement without invoking any forced aligner."""
        from chalna.llm_refiner import refine_segments

        if verbose:
            print("\nLLM Refinement:")
            print("  " + "-" * 80)

        original_segments = self._clone_segments(segments)

        def refine_progress(stage: str, value: float):
            del stage
            if progress_callback:
                progress_callback("refining", value)

        output = refine_segments(
            segments=segments,
            context=context,
            progress_callback=refine_progress,
        )

        refined_segments = list(output.segments)
        log = output.log
        self._apply_scribe_word_boundaries_for_splits(
            refined_segments=refined_segments,
            original_segments=original_segments,
            origin_map=output.origin_map,
            log=log,
        )

        if verbose:
            split_count = sum(1 for entry in log if entry.get("status") == "split")
            refined_count = sum(1 for entry in log if entry.get("status") == "refined")
            error_count = sum(1 for entry in log if entry.get("status") == "error")
            parse_error_count = sum(1 for entry in log if entry.get("status") == "parse_error")
            print(
                "  Split: "
                f"{split_count}, Refined: {refined_count}, Errors: {error_count}, "
                f"Parse errors: {parse_error_count}"
            )

        refined_segments = self._fix_overlapping_timestamps(refined_segments, verbose=verbose)
        for i, segment in enumerate(refined_segments, start=1):
            segment.index = i

        return refined_segments, log

    def _apply_scribe_word_boundaries_for_splits(
        self,
        *,
        refined_segments: list[Segment],
        original_segments: list[Segment],
        origin_map: dict[int, int],
        log: list[dict],
    ) -> None:
        """Use Scribe word timestamps to improve LLM-created split boundaries."""
        originals_by_index = {segment.index: segment for segment in original_segments}

        for entry in log:
            if entry.get("status") != "split":
                continue

            original_index = entry.get("original_index")
            new_indices = entry.get("new_segment_indices") or []
            split_texts = entry.get("split_texts") or []
            original_segment = originals_by_index.get(original_index)
            words = self._scribe_words_by_segment_index.get(original_index, [])

            if not original_segment or not new_indices or not split_texts or not words:
                continue

            boundaries = self._estimate_split_boundaries_from_words(
                words=words,
                original_segment=original_segment,
                split_texts=split_texts,
            )
            if not boundaries:
                continue

            for offset, new_idx in enumerate(new_indices):
                if offset >= len(boundaries) or new_idx >= len(refined_segments):
                    continue
                start, end = boundaries[offset]
                refined_segments[new_idx].start_time = start
                refined_segments[new_idx].end_time = end
                refined_segments[new_idx].speaker_id = original_segment.speaker_id

            entry["timestamp_source"] = "scribe_v2_words"
            for new_idx in new_indices:
                origin_map[new_idx] = original_index

    def _estimate_split_boundaries_from_words(
        self,
        *,
        words: list[dict],
        original_segment: Segment,
        split_texts: list[str],
    ) -> Optional[list[tuple[float, float]]]:
        timed_words = [
            word for word in words
            if word.get("type", "word") == "word"
            and word.get("start") is not None
            and word.get("end") is not None
        ]
        if len(timed_words) < len(split_texts):
            return None

        word_lengths = [
            max(1, len(str(word.get("text", word.get("word", ""))).strip()))
            for word in timed_words
        ]
        total_word_chars = sum(word_lengths)
        split_lengths = [max(1, len(text.replace(" ", ""))) for text in split_texts]
        total_split_chars = sum(split_lengths)
        if total_word_chars <= 0 or total_split_chars <= 0:
            return None

        boundaries: list[tuple[float, float]] = []
        start_word_idx = 0
        cumulative_split_chars = 0

        for split_idx, split_len in enumerate(split_lengths):
            cumulative_split_chars += split_len
            if split_idx == len(split_lengths) - 1:
                end_word_idx = len(timed_words) - 1
            else:
                target_chars = total_word_chars * cumulative_split_chars / total_split_chars
                running_chars = 0
                end_word_idx = start_word_idx
                for word_idx, word_len in enumerate(word_lengths):
                    running_chars += word_len
                    if running_chars >= target_chars:
                        end_word_idx = word_idx
                        break
                min_remaining = len(split_lengths) - split_idx - 1
                end_word_idx = min(end_word_idx, len(timed_words) - min_remaining - 1)
                end_word_idx = max(end_word_idx, start_word_idx)

            start = float(timed_words[start_word_idx]["start"])
            end = float(timed_words[end_word_idx]["end"])
            start = max(original_segment.start_time, start)
            end = min(original_segment.end_time, end)
            if end <= start:
                return None

            boundaries.append((start, end))
            start_word_idx = min(end_word_idx + 1, len(timed_words) - 1)

        return boundaries

    @staticmethod
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

    def _fix_overlapping_timestamps(
        self,
        segments: list[Segment],
        verbose: bool = True,
    ) -> list[Segment]:
        """Fix overlapping timestamps between consecutive segments."""
        if len(segments) <= 1:
            return segments

        fixed_count = 0
        for i in range(len(segments) - 1):
            current = segments[i]
            next_segment = segments[i + 1]
            if current.end_time > next_segment.start_time:
                midpoint = (current.end_time + next_segment.start_time) / 2
                current.end_time = midpoint
                next_segment.start_time = midpoint
                fixed_count += 1

        if verbose and fixed_count > 0:
            print(f"\n  Fixed {fixed_count} overlapping timestamp(s) using midpoint interpolation")

        return segments
