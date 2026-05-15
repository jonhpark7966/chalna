from pathlib import Path

from chalna.context_builder import build_transcription_context


def test_returns_user_context_when_no_assets(tmp_path: Path):
    context = build_transcription_context("출연자: JB, JC", assets_dir=tmp_path)

    assert context is not None
    assert "## 요청 컨텍스트" in context
    assert "출연자: JB, JC" in context
    assert "누적 SRT 기반" not in context


def test_extracts_compact_hints_from_srt_assets(tmp_path: Path):
    (tmp_path / "domain.srt").write_text(
        "\n".join(
            [
                "1",
                "00:00:00,000 --> 00:00:02,000",
                "[JC] Action-Conditioned 월드모델과 World Labs를 얘기합니다.",
                "",
                "2",
                "00:00:02,000 --> 00:00:04,000",
                "[JB] Vision-Language-Action 모델과 에이전트가 중요합니다.",
            ]
        ),
        encoding="utf-8",
    )

    context = build_transcription_context("제목: 월드모델", assets_dir=tmp_path)

    assert context is not None
    assert "제목: 월드모델" in context
    assert "누적 SRT 기반 전사 힌트" in context
    assert "JC" in context
    assert "Action-Conditioned" in context
    assert "Vision-Language-Action" in context
    assert "월드모델" in context
    assert "에이전트" in context


def test_limits_context_size(tmp_path: Path):
    (tmp_path / "large.srt").write_text(
        "\n".join(
            [
                f"{i}\n00:00:{i % 60:02d},000 --> 00:00:{(i + 1) % 60:02d},000\n[JC] World Labs 월드모델 에이전트"
                for i in range(200)
            ]
        ),
        encoding="utf-8",
    )

    context = build_transcription_context("x" * 5000, assets_dir=tmp_path, max_chars=1000)

    assert context is not None
    assert len(context) <= 1000
    assert "context truncated" in context
