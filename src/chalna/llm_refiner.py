"""
LLM-based subtitle refinement using Codex CLI.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from chalna.exceptions import CodexAPIError, CodexRateLimitError
from chalna.models import Segment
from chalna.settings import settings


@dataclass
class RefinementResult:
    """Result of LLM refinement for a segment."""

    original_text: str
    refined_text: str
    split_texts: Optional[List[str]]  # Kept for legacy compatibility; refinement no longer splits.
    needs_realignment: bool
    parse_error: Optional[str] = None  # Set if parsing failed


@dataclass
class RefinementOutput:
    """Output of refine_segments function."""

    segments: List[Segment]
    log: List[dict]
    # Maps new segment index (0-based) to original segment index (1-based)
    # Refinement does not split/merge segments, but zero-duration filtering can reindex.
    origin_map: Dict[int, int] = field(default_factory=dict)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def call_codex_cli(
    prompt: str,
    model: str = "gpt-5.5",
    reasoning_effort: str = "xhigh",
    timeout: int = 120,
) -> str:
    """
    Call Codex CLI in exec (non-interactive) mode.

    Uses stdin for prompt to handle long prompts safely.

    Args:
        prompt: The prompt to send
        model: Model to use (default: gpt-5.5)
        reasoning_effort: Reasoning effort level (minimal/low/medium/high/xhigh; default: xhigh)
        timeout: Timeout in seconds

    Returns:
        The response text.

    Raises:
        CodexAPIError: If Codex CLI fails.
        CodexRateLimitError: If rate limit or quota is exceeded.
    """
    try:
        result = subprocess.run(
            [
                "codex",
                "exec",
                "--skip-git-repo-check",  # /app is not a git repo inside the container
                "-m", model,
                "-c", f"model_reasoning_effort={reasoning_effort}",
                "-",  # Read prompt from stdin
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "rate limit" in stderr.lower() or "quota" in stderr.lower():
                raise CodexRateLimitError(stderr)
            raise CodexAPIError(f"Codex CLI failed: {stderr}")

        return result.stdout.strip()

    except subprocess.TimeoutExpired as e:
        raise CodexAPIError(f"Codex CLI timeout after {timeout}s", cause=e)
    except FileNotFoundError as e:
        raise CodexAPIError("Codex CLI not found", cause=e)


def build_refinement_prompt(
    segments: List[Segment],
    context: Optional[str],
    chunk_start_idx: int,
    chunk_end_idx: int,
) -> str:
    """
    Build prompt for refining a chunk of segments.

    Includes surrounding context (2 segments before/after) for better understanding.
    """
    # Context window: include 2 segments before and after for context
    context_before = segments[max(0, chunk_start_idx - 2):chunk_start_idx]
    context_after = segments[chunk_end_idx:min(len(segments), chunk_end_idx + 2)]
    target_segments = segments[chunk_start_idx:chunk_end_idx]

    prompt = """당신은 한국어 자막 교정 전문가입니다. 음성 인식 결과의 텍스트만 교정해주세요.

## 규칙
1. 음성 인식 오류만 수정 (동음이의어, 띄어쓰기, 맞춤법)
2. 말이 끊기거나 불완전한 문장은 그대로 유지 (강제로 완성시키지 마세요)
3. 원래 의미와 뉘앙스를 절대 변경하지 마세요
4. 세그먼트를 나누거나 합치지 마세요
5. 분리 마커나 구분자를 출력하지 마세요
6. timestamp, index, speaker는 변경하지 않습니다

## 출력 형식
- 입력과 동일한 개수의 JSON 배열 반환
- index는 반드시 입력의 index와 일치해야 함
- 각 항목은 index와 text만 포함
[
  {"index": 1, "text": "교정된 텍스트"},
  {"index": 2, "text": "교정된 텍스트"}
]
"""

    if context:
        prompt += f"\n## 참고 자료 (대본/컨텍스트)\n{context}\n"

    if context_before:
        prompt += "\n## 앞 문맥 (참고용, 수정 대상 아님)\n"
        for seg in context_before:
            prompt += f"[{seg.start_time:.1f}s] {seg.text}\n"

    prompt += "\n## 교정 대상 세그먼트\n"
    for i, seg in enumerate(target_segments):
        duration = seg.end_time - seg.start_time
        prompt += f"[index={i + 1}, {duration:.1f}초] {seg.text}\n"

    if context_after:
        prompt += "\n## 뒤 문맥 (참고용, 수정 대상 아님)\n"
        for seg in context_after:
            prompt += f"[{seg.start_time:.1f}s] {seg.text}\n"

    prompt += "\n## JSON 응답:"

    return prompt


def parse_refinement_response(
    response: str,
    original_segments: List[Segment],
) -> Tuple[List[RefinementResult], Optional[str]]:
    """
    Parse Codex CLI response and extract refinement results.

    Returns:
        (results, parse_error) - parse_error is set if parsing failed
    """
    parse_error = None

    try:
        # Extract JSON from response (handle markdown code blocks)
        response_clean = response.strip()

        # Remove markdown code block if present
        if response_clean.startswith("```"):
            lines = response_clean.split("\n")
            # Find start and end of code block
            start_idx = 1 if lines[0].startswith("```") else 0
            end_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            response_clean = "\n".join(lines[start_idx:end_idx])

        # Find JSON array
        start = response_clean.find('[')
        end = response_clean.rfind(']') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON array found in response")

        data = json.loads(response_clean[start:end])

        # Validate array length
        if len(data) != len(original_segments):
            parse_error = (
                f"Array length mismatch: expected {len(original_segments)}, "
                f"got {len(data)}"
            )

        results = []
        for i, orig_seg in enumerate(original_segments):
            # Find matching item by index first
            item = None
            for d in data:
                if d.get("index") == i + 1:
                    item = d
                    break

            # Fallback to position-based matching
            if item is None and i < len(data):
                item = data[i]
                # Validate: if response has index field, warn about mismatch
                if item and item.get("index") is not None and item.get("index") != i + 1:
                    parse_error = (
                        f"Index mismatch at position {i}: expected {i + 1}, "
                        f"got {item.get('index')}"
                    )

            if item is None:
                # No matching item, keep original
                results.append(RefinementResult(
                    original_text=orig_seg.text,
                    refined_text=orig_seg.text,
                    split_texts=None,
                    needs_realignment=False,
                    parse_error=f"No matching item for index {i + 1}",
                ))
                continue

            text_value = item.get("text", orig_seg.text)
            text = str(text_value) if text_value is not None else orig_seg.text
            text = " ".join(text.replace("|SPLIT|", " ").split())

            # Sanity check: if refined text is drastically different in length, warn
            orig_len = len(orig_seg.text)
            new_len = len(text)
            if orig_len > 20 and (new_len < orig_len * 0.3 or new_len > orig_len * 3):
                # Likely a misalignment - keep original
                results.append(RefinementResult(
                    original_text=orig_seg.text,
                    refined_text=orig_seg.text,
                    split_texts=None,
                    needs_realignment=False,
                    parse_error=(
                        f"Length mismatch: orig={orig_len}, new={new_len}, "
                        "keeping original"
                    ),
                ))
                continue

            # Text changed and segment is > 3 seconds -> needs re-alignment
            duration = orig_seg.end_time - orig_seg.start_time
            text_changed = text.strip() != orig_seg.text.strip()
            results.append(RefinementResult(
                original_text=orig_seg.text,
                refined_text=text.strip(),
                split_texts=None,
                needs_realignment=text_changed and duration > 3,
            ))

        return results, parse_error

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        # Fallback: keep original with error info
        parse_error = f"JSON parse failed: {type(e).__name__}: {e}"
        return [
            RefinementResult(
                original_text=seg.text,
                refined_text=seg.text,
                split_texts=None,
                needs_realignment=False,
                parse_error=parse_error,
            )
            for seg in original_segments
        ], parse_error


def _process_full_context(
    segments: List[Segment],
    context: Optional[str],
) -> Tuple[List[RefinementResult], Optional[str], dict]:
    """
    Process all segments through one LLM refinement call so the model sees full context.
    """
    prompt = build_refinement_prompt(
        segments=segments,
        context=context,
        chunk_start_idx=0,
        chunk_end_idx=len(segments),
    )
    model = "gpt-5.5"
    reasoning_effort = "xhigh"
    io_log = {
        "status": "llm_io",
        "stage": "refine",
        "provider": "codex",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "cache_hit": False,
        "input": {
            "prompt": prompt,
            "prompt_sha256": _hash_text(prompt),
            "segment_count": len(segments),
            "has_context": bool(context),
        },
    }
    try:
        response = call_codex_cli(
            prompt,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout=settings.llm_refinement_timeout,
        )
    except Exception as exc:
        io_log.update({
            "llm_io_status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise

    results, parse_error = parse_refinement_response(response, segments)
    io_log.update({
        "llm_io_status": "ok",
        "output": {
            "response": response,
            "response_sha256": _hash_text(response),
            "parse_error": parse_error,
            "result_count": len(results),
        },
    })
    return results, parse_error, io_log


def _build_segments_from_chunk_result(
    chunk_segments: List[Segment],
    results: Optional[List[RefinementResult]],
    parse_error: Optional[str],
    error: Optional[Exception],
    chunk_idx: int,
    refined_segments: List[Segment],
    refinement_log: List[dict],
    origin_map: Dict[int, int],
) -> None:
    """Build refined segments from a single chunk's results. Mutates the output lists in-place."""
    if error is not None:
        # Error on this chunk — keep original segments
        for seg in chunk_segments:
            new_idx = len(refined_segments)
            origin_map[new_idx] = seg.index
            refined_segments.append(Segment(
                index=new_idx + 1,
                start_time=seg.start_time,
                end_time=seg.end_time,
                text=seg.text,
                speaker_id=seg.speaker_id,
                confidence=seg.confidence,
            ))
            refinement_log.append({
                "original_index": seg.index,
                "new_segment_index": new_idx,
                "status": "error",
                "error": str(error),
                "text": seg.text,
            })
        return

    for orig_seg, result in zip(chunk_segments, results):
        new_idx = len(refined_segments)
        origin_map[new_idx] = orig_seg.index
        refined_segments.append(Segment(
            index=new_idx + 1,
            start_time=orig_seg.start_time,
            end_time=orig_seg.end_time,
            text=result.refined_text,
            speaker_id=orig_seg.speaker_id,
            confidence=orig_seg.confidence,
        ))

        log_entry = {
            "original_index": orig_seg.index,
            "new_segment_index": new_idx,
        }

        if result.parse_error:
            log_entry["status"] = "parse_error"
            log_entry["parse_error"] = result.parse_error
            log_entry["text"] = orig_seg.text
        elif result.refined_text != orig_seg.text:
            log_entry["status"] = "refined"
            log_entry["original_text"] = orig_seg.text
            log_entry["refined_text"] = result.refined_text
            log_entry["needs_realignment"] = result.needs_realignment
        else:
            log_entry["status"] = "unchanged"
            log_entry["text"] = orig_seg.text

        refinement_log.append(log_entry)

    if parse_error:
        refinement_log.append({
            "chunk_idx": chunk_idx,
            "status": "chunk_parse_warning",
            "warning": parse_error,
        })


def refine_segments(
    segments: List[Segment],
    context: Optional[str] = None,
    chunk_size: int = 30,
    max_workers: int = 5,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> RefinementOutput:
    """
    Refine all segments using LLM.

    All segments are processed in one LLM call so the model can use full context.
    chunk_size and max_workers are retained for API compatibility.

    Args:
        segments: List of aligned segments
        context: Optional context/script for reference
        chunk_size: Deprecated; retained for callers that still pass it
        max_workers: Deprecated; retained for callers that still pass it
        progress_callback: Optional callback(stage, value) for progress

    Returns:
        RefinementOutput with segments, log, and origin_map

    Raises:
        CodexAPIError: If Codex CLI is not available
        CodexRateLimitError: If rate limit exceeded
    """
    _ = (chunk_size, max_workers)

    if not segments:
        return RefinementOutput(segments=[], log=[], origin_map={})

    if progress_callback:
        progress_callback("refining", 0.0)

    results, parse_error, io_log = _process_full_context(segments, context)

    refined_segments: List[Segment] = []
    refinement_log: List[dict] = [{
        "status": "refinement_mode",
        "mode": "full_context_single_call",
        "segment_count": len(segments),
    }]
    origin_map: Dict[int, int] = {}

    _build_segments_from_chunk_result(
        segments, results, parse_error, None, 0,
        refined_segments, refinement_log, origin_map,
    )
    refinement_log.append(io_log)

    if progress_callback:
        progress_callback("refining", 1.0)

    # Filter out zero-duration segments
    filtered_segments: List[Segment] = []
    filtered_origin_map: Dict[int, int] = {}
    removed_count = 0

    for old_idx, seg in enumerate(refined_segments):
        duration = seg.end_time - seg.start_time
        if duration <= 0:
            # Log removal
            refinement_log.append({
                "original_index": origin_map.get(old_idx),
                "status": "removed_zero_duration",
                "text": seg.text,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
            })
            removed_count += 1
        else:
            new_idx = len(filtered_segments)
            filtered_origin_map[new_idx] = origin_map.get(old_idx, seg.index)
            # Re-index the segment
            filtered_segments.append(Segment(
                index=new_idx + 1,
                start_time=seg.start_time,
                end_time=seg.end_time,
                text=seg.text,
                speaker_id=seg.speaker_id,
                confidence=seg.confidence,
            ))

    if removed_count > 0:
        refinement_log.append({
            "status": "zero_duration_removal_summary",
            "removed_count": removed_count,
        })

    return RefinementOutput(
        segments=filtered_segments,
        log=refinement_log,
        origin_map=filtered_origin_map,
    )
