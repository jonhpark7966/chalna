"""
LLM-based subtitle refinement using Codex CLI.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from chalna.exceptions import CodexAPIError, CodexRateLimitError
from chalna.models import Segment


@dataclass
class RefinementResult:
    """Result of LLM refinement for a segment."""

    original_text: str
    refined_text: str
    split_texts: Optional[List[str]]  # None if not split, list if split
    needs_realignment: bool
    parse_error: Optional[str] = None  # Set if parsing failed


@dataclass
class RefinementOutput:
    """Output of refine_segments function."""

    segments: List[Segment]
    log: List[dict]
    # Maps new segment index (0-based) to original segment index (1-based)
    # For split segments, multiple new indices map to the same original index
    origin_map: Dict[int, int] = field(default_factory=dict)


def call_codex_cli(
    prompt: str,
    model: str = "gpt-5.2",
    reasoning_effort: str = "medium",
    timeout: int = 120,
) -> str:
    """
    Call Codex CLI in exec (non-interactive) mode.

    Uses stdin for prompt to handle long prompts safely.

    Args:
        prompt: The prompt to send
        model: Model to use (default: gpt-5.2)
        reasoning_effort: Reasoning effort level (minimal/low/medium/high/xhigh)
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

    prompt = """당신은 한국어 자막 교정 전문가입니다. 음성 인식 결과를 교정해주세요.

## 규칙
1. 음성 인식 오류만 수정 (동음이의어, 띄어쓰기, 맞춤법)
2. 말이 끊기거나 불완전한 문장은 그대로 유지 (강제로 완성시키지 마세요)
3. 원래 의미와 뉘앙스를 절대 변경하지 마세요

## 중요: 긴 세그먼트 분리 (필수)
- 5초 이상이면서 여러 문장이 포함된 세그먼트는 반드시 |SPLIT| 마커로 분리하세요
- 문장 경계(마침표, 물음표, 느낌표, 의미 단위)에서 분리하세요
- 예: "첫 번째 문장입니다. 두 번째 문장입니다." → "첫 번째 문장입니다. |SPLIT| 두 번째 문장입니다."

## 출력 형식
- 입력과 동일한 개수의 JSON 배열 반환
- index는 반드시 입력의 index와 일치해야 함
[
  {"index": 1, "text": "교정된 텍스트"},
  {"index": 2, "text": "첫 문장 |SPLIT| 두 번째 문장"}
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
        # Mark long segments that need splitting
        split_marker = " ⚠️분리필요" if duration >= 5.0 else ""
        prompt += f"[index={i + 1}, {duration:.1f}초{split_marker}] {seg.text}\n"

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
            parse_error = f"Array length mismatch: expected {len(original_segments)}, got {len(data)}"

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
                    parse_error = f"Index mismatch at position {i}: expected {i + 1}, got {item.get('index')}"

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

            text = item.get("text", orig_seg.text)

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
                    parse_error=f"Length mismatch: orig={orig_len}, new={new_len}, keeping original",
                ))
                continue

            if "|SPLIT|" in text:
                split_texts = [t.strip() for t in text.split("|SPLIT|") if t.strip()]
                results.append(RefinementResult(
                    original_text=orig_seg.text,
                    refined_text=text.replace("|SPLIT|", " ").strip(),
                    split_texts=split_texts,
                    needs_realignment=True,
                ))
            else:
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


def refine_segments(
    segments: List[Segment],
    context: Optional[str] = None,
    chunk_size: int = 30,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> RefinementOutput:
    """
    Refine all segments using LLM.

    Args:
        segments: List of aligned segments
        context: Optional context/script for reference
        chunk_size: Number of segments per LLM call (~30 segments = 2-3min audio)
        progress_callback: Optional callback(stage, value) for progress

    Returns:
        RefinementOutput with segments, log, and origin_map

    Raises:
        CodexAPIError: If Codex CLI is not available (on first chunk)
        CodexRateLimitError: If rate limit exceeded (on first chunk)
    """
    refined_segments: List[Segment] = []
    refinement_log: List[dict] = []
    origin_map: Dict[int, int] = {}  # new_idx (0-based) -> original_index (1-based)

    total_chunks = (len(segments) + chunk_size - 1) // chunk_size
    first_chunk_error: Optional[Exception] = None

    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(segments))
        chunk_segments = segments[start_idx:end_idx]

        if progress_callback:
            progress_callback("refining", chunk_idx / total_chunks)

        try:
            prompt = build_refinement_prompt(
                segments=segments,
                context=context,
                chunk_start_idx=start_idx,
                chunk_end_idx=end_idx,
            )

            response = call_codex_cli(prompt)
            results, parse_error = parse_refinement_response(response, chunk_segments)

            for orig_seg, result in zip(chunk_segments, results):
                if result.split_texts:
                    # Segment needs splitting - create placeholder segments
                    # Actual timestamps will be refined by re-alignment
                    time_per_part = (orig_seg.end_time - orig_seg.start_time) / len(result.split_texts)
                    split_start_idx = len(refined_segments)

                    for i, text in enumerate(result.split_texts):
                        new_idx = len(refined_segments)
                        origin_map[new_idx] = orig_seg.index  # Track origin
                        refined_segments.append(Segment(
                            index=new_idx + 1,
                            start_time=orig_seg.start_time + i * time_per_part,
                            end_time=orig_seg.start_time + (i + 1) * time_per_part,
                            text=text,
                            speaker_id=orig_seg.speaker_id,
                            confidence=orig_seg.confidence * 0.9,
                        ))

                    refinement_log.append({
                        "original_index": orig_seg.index,
                        "status": "split",
                        "original_text": orig_seg.text,
                        "split_count": len(result.split_texts),
                        "split_texts": result.split_texts,
                        "new_segment_indices": list(range(split_start_idx, len(refined_segments))),
                        "original_start": orig_seg.start_time,
                        "original_end": orig_seg.end_time,
                    })
                else:
                    new_idx = len(refined_segments)
                    origin_map[new_idx] = orig_seg.index  # Track origin
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

            # Log chunk-level parse error if any
            if parse_error:
                refinement_log.append({
                    "chunk_idx": chunk_idx,
                    "status": "chunk_parse_warning",
                    "warning": parse_error,
                })

        except (CodexAPIError, CodexRateLimitError) as e:
            # On first chunk, raise immediately (fatal error like CLI not found)
            if chunk_idx == 0:
                raise

            # On subsequent chunks, log error and keep original segments
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
                    "error": str(e),
                    "text": seg.text,
                })

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
