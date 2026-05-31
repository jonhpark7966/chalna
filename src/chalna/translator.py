"""Target-language translation via Codex CLI (length-aware for dubbing).

Reuses the same Codex CLI path as LLM refinement. Translations are prompted to
keep roughly the same spoken length as the source so they fit the original timing
when dubbed downstream (e.g. by jeoneum).
"""
from __future__ import annotations

import json
from typing import List, Optional

from chalna.llm_refiner import call_codex_cli


def _extract_json_array(text: str) -> list:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON array in translation output: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def build_translation_prompt(
    segments: List[dict], source_language: Optional[str], target_language: str
) -> str:
    src = source_language or "the source language"
    payload = json.dumps(
        [{"index": s["index"], "text": s["text"]} for s in segments], ensure_ascii=False
    )
    return (
        f"You are a professional subtitle translator for DUBBING. "
        f"Translate each segment from {src} to {target_language}.\n"
        "Rules:\n"
        "1. Preserve meaning, tone, and speaker intent.\n"
        "2. Keep each translation roughly the SAME SPOKEN LENGTH as the source "
        "(similar syllable count / duration) so it fits the original timing when "
        "dubbed. Prefer concise phrasing over literal expansion.\n"
        "3. Do NOT merge or split segments. Return EXACTLY the same number of items "
        "with the SAME index values.\n"
        'Output ONLY a JSON array, no prose: [{"index": 1, "text": "..."}, ...]\n\n'
        f"Segments:\n{payload}\n"
    )


def translate_segments(
    segments: List[dict],
    source_language: Optional[str],
    target_language: str,
    model: str = "gpt-5.5",
    reasoning_effort: str = "xhigh",
    timeout: int = 300,
) -> List[dict]:
    """Translate [{index, text}, ...] -> [{index, text}, ...] in the same order."""
    if not segments:
        return []
    prompt = build_translation_prompt(segments, source_language, target_language)
    raw = call_codex_cli(prompt, model=model, reasoning_effort=reasoning_effort, timeout=timeout)
    items = _extract_json_array(raw)
    by_index = {int(it["index"]): str(it["text"]) for it in items}
    return [{"index": s["index"], "text": by_index.get(s["index"], "")} for s in segments]
