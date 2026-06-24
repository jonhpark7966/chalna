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
    LlmSegmentationOptions,
    ScribeOptions,
    Segment,
    TranscriptionMetadata,
    TranscriptionResult,
)
from chalna.scribe_adapter import scribe_response_to_segments
from chalna.scribe_cache import (
    ScribeResponseCache,
    build_scribe_cache_key,
    build_scribe_cache_metadata,
)
from chalna.scribe_client import ScribeClient
from chalna.scribe_llm_segmenter import LlmScribeSegmenter
from chalna.segment_cache import SegmentPlanCache
from chalna.segmentation_boundary import apply_boundary_rule
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
        use_llm_segmentation: bool = True,
        aligner_path: str = "",
        scribe_client: Optional[ScribeClient] = None,
        scribe_cache: Optional[ScribeResponseCache] = None,
        llm_segmenter: Optional[LlmScribeSegmenter] = None,
        segment_cache: Optional[SegmentPlanCache] = None,
    ):
        # Kept for constructor compatibility. Qwen alignment is intentionally disabled.
        self.device = device
        self.dtype = dtype
        self.use_alignment = False
        self.use_llm_refinement = use_llm_refinement
        self.use_llm_segmentation = use_llm_segmentation
        self.aligner_path = aligner_path

        self.scribe_client = scribe_client or ScribeClient()
        self.scribe_cache = scribe_cache or ScribeResponseCache(settings.scribe_cache_dir)
        self.llm_segmenter = llm_segmenter or LlmScribeSegmenter(
            cache=segment_cache or SegmentPlanCache(settings.llm_segmentation_cache_dir)
        )

        self._auto_unload = False
        self._last_alignment_log: list[dict] = []
        self._pre_alignment_segments: Optional[list[Segment]] = None
        self._raw_segments: Optional[list[Segment]] = None
        self._aligned_segments: Optional[list[Segment]] = None
        self._refined_segments: Optional[list[Segment]] = None
        self._refinement_log: Optional[list[dict]] = None
        self._segmentation_log: Optional[list[dict]] = None
        self._segmentation_boundary_stats: Optional[dict] = None
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
        llm_segmentation_options: Optional[LlmSegmentationOptions] = None,
    ) -> TranscriptionResult:
        """Transcribe an audio/video file to Chalna segments and subtitles."""
        del max_new_tokens  # Kept for compatibility with the previous local ASR API.

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._reset_request_state()
        options = scribe_options or ScribeOptions()
        segmentation_options = llm_segmentation_options or self._default_llm_segmentation_options()
        segmentation_options.enabled = segmentation_options.enabled and self.use_llm_segmentation

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
        scribe_cache_key = build_scribe_cache_key(cache_metadata)
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
        _progress("transcribing", 0.65, cache_hit=cache_hit)

        segments, words_by_segment_index, language_code, segmentation_source = (
            self._segments_from_scribe_response(
                response=scribe_response,
                scribe_cache_key=scribe_cache_key,
                language=language,
                context=context,
                scribe_options=options,
                segmentation_options=segmentation_options,
            )
        )
        boundary_result = apply_boundary_rule(
            segments,
            rule=segmentation_options.boundary_rule,
            audio_path=audio_path,
        )
        segments = boundary_result.segments
        self._segmentation_boundary_stats = boundary_result.stats
        self._scribe_words_by_segment_index = words_by_segment_index

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
        result_language = language_code or language

        metadata = TranscriptionMetadata(
            duration=audio_info.duration_seconds,
            language=result_language,
            speakers=speakers,
            model_version=self.scribe_client.model_id,
            aligned=False,
            refined=self.use_llm_refinement and self._refined_segments is not None,
            timestamp_source=self.scribe_client.model_id,
            segmentation_source=segmentation_source,
            segmentation_boundary_rule=boundary_result.rule,
            segmentation_boundary_effective_rule=boundary_result.effective_rule,
            segmentation_boundary_stats=boundary_result.stats,
        )

        intermediate = IntermediateResults(
            raw_segments=self._raw_segments,
            aligned_segments=None,
            refined_segments=self._refined_segments,
            chunk_raw_segments=None,
            alignment_log=[],
            segmentation_log=self._segmentation_log,
            refinement_log=self._refinement_log,
        )

        return TranscriptionResult(
            segments=segments,
            metadata=metadata,
            intermediate=intermediate,
        )

    def transcribe_from_scribe_response(
        self,
        audio_path: str | Path,
        scribe_response: dict,
        context: Optional[str] = None,
        language: Optional[str] = None,
        max_new_tokens: int = 65536,
        verbose: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        scribe_options: Optional[ScribeOptions] = None,
        llm_segmentation_options: Optional[LlmSegmentationOptions] = None,
    ) -> TranscriptionResult:
        """Run Chalna segmentation/refinement from a pre-existing raw Scribe response."""
        del max_new_tokens

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not isinstance(scribe_response, dict):
            raise ValueError("scribe_response must be a JSON object")

        self._reset_request_state()
        options = scribe_options or ScribeOptions()
        segmentation_options = llm_segmentation_options or self._default_llm_segmentation_options()
        segmentation_options.enabled = segmentation_options.enabled and self.use_llm_segmentation

        def _progress(stage: str, value: float, **extra):
            if progress_callback:
                progress_callback(stage, value, **extra)

        _progress("validating", 0.0)
        audio_info = validate_audio_file(audio_path)
        _progress("validating", 1.0)

        required_mb = estimate_temp_space_required(audio_info)
        check_disk_space(required_mb)

        cache_metadata = build_scribe_cache_metadata(
            audio_path=audio_path,
            audio_info=audio_info,
            model_id=self.scribe_client.model_id,
            language_code=language,
            options=options,
        )
        scribe_cache_key = build_scribe_cache_key(cache_metadata)

        _progress("transcribing", 0.0, cache_hit=True, source="provided_scribe_response")
        self._last_scribe_response = scribe_response

        segments, words_by_segment_index, language_code, segmentation_source = (
            self._segments_from_scribe_response(
                response=scribe_response,
                scribe_cache_key=scribe_cache_key,
                language=language,
                context=context,
                scribe_options=options,
                segmentation_options=segmentation_options,
            )
        )
        boundary_result = apply_boundary_rule(
            segments,
            rule=segmentation_options.boundary_rule,
            audio_path=audio_path,
        )
        segments = boundary_result.segments
        self._segmentation_boundary_stats = boundary_result.stats
        self._scribe_words_by_segment_index = words_by_segment_index

        if not segments:
            raise EmptyTranscriptionError(audio_duration=audio_info.duration_seconds)

        self._raw_segments = self._clone_segments(segments)
        self._pre_alignment_segments = self._raw_segments
        _progress("transcribing", 1.0, cache_hit=True, source="provided_scribe_response")

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
        result_language = language_code or language

        metadata = TranscriptionMetadata(
            duration=audio_info.duration_seconds,
            language=result_language,
            speakers=speakers,
            model_version=self.scribe_client.model_id,
            aligned=False,
            refined=self.use_llm_refinement and self._refined_segments is not None,
            timestamp_source=self.scribe_client.model_id,
            segmentation_source=segmentation_source,
            segmentation_boundary_rule=boundary_result.rule,
            segmentation_boundary_effective_rule=boundary_result.effective_rule,
            segmentation_boundary_stats=boundary_result.stats,
        )

        intermediate = IntermediateResults(
            raw_segments=self._raw_segments,
            aligned_segments=None,
            refined_segments=self._refined_segments,
            chunk_raw_segments=None,
            alignment_log=[],
            segmentation_log=self._segmentation_log,
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

    def get_segmentation_log(self) -> Optional[list[dict]]:
        """Get LLM word boundary planning log."""
        return self._segmentation_log

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
        self._segmentation_log = None
        self._segmentation_boundary_stats = None
        self._last_alignment_log = []
        self._pre_alignment_segments = None
        self._scribe_words_by_segment_index = {}
        self._last_scribe_response = None

    def _default_llm_segmentation_options(self) -> LlmSegmentationOptions:
        return LlmSegmentationOptions(
            enabled=self.use_llm_segmentation,
            model=settings.llm_segmentation_model,
            reasoning_effort=settings.llm_segmentation_reasoning_effort,
        )

    def _segments_from_scribe_response(
        self,
        *,
        response: dict,
        scribe_cache_key: str,
        language: Optional[str],
        context: Optional[str],
        scribe_options: ScribeOptions,
        segmentation_options: LlmSegmentationOptions,
    ) -> tuple[list[Segment], dict[int, list[dict]], Optional[str], str]:
        language_code = response.get("language_code")
        language_code = str(language_code) if language_code else language

        if segmentation_options.enabled:
            try:
                result = self.llm_segmenter.segment(
                    response,
                    scribe_cache_key=scribe_cache_key,
                    language_code=language_code,
                    context=context,
                    scribe_options=scribe_options,
                    segmentation_options=segmentation_options,
                    include_audio_events=scribe_options.tag_audio_events,
                )
                self._segmentation_log = result.log
                return (
                    result.segments,
                    result.words_by_segment_index,
                    language_code,
                    "llm",
                )
            except (CodexAPIError, CodexRateLimitError, ValueError) as e:
                self._segmentation_log = [{
                    "status": "fallback",
                    "source": "heuristic",
                    "error": str(e),
                }]

        adapter_result = scribe_response_to_segments(
            response,
            include_audio_events=scribe_options.tag_audio_events,
        )
        if self._segmentation_log is None:
            self._segmentation_log = [{
                "status": "skipped",
                "source": "heuristic",
                "reason": "llm_segmentation_disabled",
            }]
        return (
            adapter_result.segments,
            adapter_result.words_by_segment_index,
            adapter_result.language_code or language_code,
            "heuristic",
        )

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
