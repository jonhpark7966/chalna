"""Build compact transcription context from curated SRT assets."""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}[,.]\d{3}")
SPEAKER_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9_-]{0,15})\]")
ENGLISH_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]*(?:[-+./][A-Za-z0-9]+)*|[A-Z]{2,})"
    r"(?:\s+(?:[A-Z][A-Za-z0-9]*(?:[-+./][A-Za-z0-9]+)*|[A-Z]{2,})){0,3}\b"
)
KOREAN_DOMAIN_RE = re.compile(
    r"[가-힣A-Za-z0-9+-]*(?:"
    r"월드모델|모델|에이전트|로보틱스|어플리케이션|컨디션|스테이트|액션|"
    r"폴리시|익스플로레이션|인바이런먼트|옵저베이션|리워드|펑션|"
    r"시뮬레이터|비전|랭귀지|클로드|코드|랄프톤|랄프|링크데인|"
    r"딥러닝|강화학습|컨텍스트"
    r")[가-힣A-Za-z0-9+-]*"
)

COMMON_ENGLISH = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "with",
    "Can",
    "Do",
    "Does",
    "How",
    "I",
    "No",
    "Tell",
    "That",
    "There",
    "They",
    "This",
    "What",
    "When",
    "Where",
    "Why",
    "You",
}

KOREAN_SUFFIXES = (
    "이라는",
    "라는",
    "이다",
    "입니다",
    "들이",
    "들을",
    "으로",
    "에서",
    "에게",
    "까지",
    "부터",
    "하고",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "로",
    "도",
    "만",
    "의",
)


def _default_assets_dir() -> Path:
    return Path(os.environ.get("CHALNA_CONTEXT_ASSETS_DIR", Path(__file__).resolve().parents[2] / "assets"))


def _iter_srt_texts(assets_dir: Path) -> list[str]:
    if not assets_dir.exists():
        return []

    texts: list[str] = []
    for path in sorted(assets_dir.glob("*.srt")):
        try:
            texts.append(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            texts.append(path.read_text(encoding="utf-8-sig", errors="ignore"))
    return texts


def _srt_to_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.isdigit() or TIMESTAMP_RE.match(line):
            continue
        lines.append(line)
    return lines


def _is_useful_english_term(term: str) -> bool:
    if term in COMMON_ENGLISH or term.lower() in COMMON_ENGLISH:
        return False
    if " " in term:
        return True
    if "-" in term or "+" in term or "/" in term or "." in term:
        return True
    if term.isupper() and len(term) >= 2:
        return True
    return any(ch.isupper() for ch in term[1:])


def _normalize_korean_term(term: str) -> str:
    normalized = term.strip()
    for suffix in KOREAN_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) - len(suffix) >= 2:
            return normalized[: -len(suffix)]
    return normalized


def _extract_asset_terms(texts: list[str]) -> tuple[list[str], list[str], list[str]]:
    speakers: Counter[str] = Counter()
    english_terms: Counter[str] = Counter()
    korean_terms: Counter[str] = Counter()

    for text in texts:
        for line in _srt_to_lines(text):
            for speaker in SPEAKER_RE.findall(line):
                speakers[speaker] += 1

            for token in ENGLISH_PHRASE_RE.findall(line):
                normalized = " ".join(token.strip().split())
                if not _is_useful_english_term(normalized):
                    continue
                english_terms[normalized] += 1

            cleaned = SPEAKER_RE.sub(" ", line)
            for term in KOREAN_DOMAIN_RE.findall(cleaned):
                term = _normalize_korean_term(term)
                if len(term) >= 2:
                    korean_terms[term] += 1

    return (
        [term for term, _ in speakers.most_common(12)],
        [term for term, _ in english_terms.most_common(40)],
        [term for term, _ in korean_terms.most_common(40)],
    )


def build_transcription_context(
    user_context: str | None,
    *,
    assets_dir: Path | None = None,
    max_chars: int = 4000,
) -> str | None:
    """Merge user-provided context with compact hints extracted from SRT assets.

    The assets directory is scanned at request time, so adding corrected SRT files
    improves future transcription prompts without changing application code.
    """
    assets_dir = assets_dir or _default_assets_dir()
    texts = _iter_srt_texts(assets_dir)
    user_context = (user_context or "").strip()

    sections: list[str] = []
    if user_context:
        sections.append(f"## 요청 컨텍스트\n{user_context}")

    if texts:
        speakers, english_terms, korean_terms = _extract_asset_terms(texts)
        hints: list[str] = [
            "## 누적 SRT 기반 전사 힌트",
            "- 아래 용어들은 검수된 SRT에서 자주 나온 표현입니다. 비슷하게 들리는 발화는 이 표기를 우선 고려하세요.",
            "- 기술 용어, 제품명, 인명, 영어 약어, 한국어식 영어 발음을 임의로 일반 단어로 바꾸지 마세요.",
        ]
        if speakers:
            hints.append(f"- 자주 등장하는 speaker label: {', '.join(speakers)}")
        if english_terms:
            hints.append(f"- 보존할 영어/약어 후보: {', '.join(english_terms)}")
        if korean_terms:
            hints.append(f"- 도메인/한국어식 표기 후보: {', '.join(korean_terms)}")
        sections.append("\n".join(hints))

    if not sections:
        return None

    merged = "\n\n".join(sections).strip()
    if len(merged) <= max_chars:
        return merged
    return merged[: max_chars - 80].rstrip() + "\n\n[context truncated to fit prompt budget]"
