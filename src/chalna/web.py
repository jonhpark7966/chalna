"""
Chalna Gradio Web Interface.
"""

import subprocess
import tempfile
import time
from pathlib import Path

import gradio as gr

from chalna.exceptions import ChalnaError
from chalna.pipeline import ChalnaPipeline


def get_audio_duration(file_path: str) -> float | None:
    """Get audio/video duration using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def format_time(seconds: float) -> str:
    """Format seconds to MM:SS or HH:MM:SS."""
    if seconds < 0:
        return "00:00"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def create_app(device: str = "auto") -> gr.Blocks:
    """Create Gradio app."""

    # Pipeline (lazy loaded)
    pipeline = None

    def on_file_upload(file):
        """Handle file upload - show duration and estimated time."""
        if file is None:
            return ""

        duration = get_audio_duration(file)
        if duration is None:
            return "파일 정보를 읽을 수 없습니다."

        # Estimate processing time (conservative)
        # VibeVoice: ~3-4x realtime, Alignment: ~0.5-1x realtime
        # First run includes model loading (~1-2min)
        estimated_transcribe = duration * 3.5
        estimated_align = duration * 0.8
        estimated_total = estimated_transcribe + estimated_align

        info = f"오디오 길이: {format_time(duration)} ({duration:.1f}초)\n"
        info += f"예상 처리 시간: {format_time(estimated_total)} ~ {format_time(estimated_total * 1.5)}\n"
        info += "(첫 실행 시 모델 로딩으로 1-2분 추가 소요)"

        return info

    def transcribe(
        audio_file,
        context: str,
        progress=gr.Progress(track_tqdm=True),
    ):
        nonlocal pipeline

        if audio_file is None:
            return "파일을 업로드해주세요.", None, None, None, None, ""

        start_time = time.time()

        def elapsed() -> str:
            return format_time(time.time() - start_time)

        try:
            # Get audio duration for progress estimation
            duration = get_audio_duration(audio_file) or 0

            # Lazy load pipeline
            if pipeline is None:
                progress(0, desc=f"[{elapsed()}] 모델 로딩 중... (첫 실행 시 1-2분 소요)")
                pipeline = ChalnaPipeline(device=device, use_alignment=True, use_llm_refinement=True)

            progress(0.05, desc=f"[{elapsed()}] 오디오 검증 중...")

            # Progress callback for pipeline
            stage_weights = {
                "validating": (0.05, 0.10),
                "loading_models": (0.10, 0.15),
                "transcribing": (0.15, 0.55),
                "aligning": (0.55, 0.75),
                "refining": (0.75, 0.95),
            }

            def progress_callback(stage: str, value: float):
                if stage in stage_weights:
                    start, end = stage_weights[stage]
                    actual_progress = start + (end - start) * value

                    stage_names = {
                        "validating": "오디오 검증 중",
                        "loading_models": "모델 준비 중",
                        "transcribing": "자막 생성 중 (VibeVoice)",
                        "aligning": "타임스탬프 정렬 중 (Qwen)",
                        "refining": "자막 다듬기 중 (LLM)",
                    }
                    stage_name = stage_names.get(stage, stage)

                    # Show duration info during transcription
                    if stage == "transcribing" and duration > 0:
                        progress(
                            actual_progress,
                            desc=f"[{elapsed()}] {stage_name}... (오디오: {format_time(duration)})",
                        )
                    else:
                        progress(actual_progress, desc=f"[{elapsed()}] {stage_name}...")

            # Run transcription with progress callback
            result = pipeline.transcribe(
                audio_path=audio_file,
                context=context if context.strip() else None,
                progress_callback=progress_callback,
            )

            progress(0.95, desc=f"[{elapsed()}] SRT 생성 중...")

            # Generate SRT
            srt_content = result.to_srt(include_speaker=True)

            # Save final SRT to temp file for download
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".srt",
                delete=False,
                encoding="utf-8",
            ) as f:
                f.write(srt_content)
                srt_path = f.name

            # Generate intermediate result files (use result.intermediate for thread safety)
            from chalna.srt_utils import segments_to_srt

            raw_path = None
            aligned_path = None
            refined_path = None

            intermediate = result.intermediate

            # Stage 1: Raw segments
            raw_segs = intermediate.raw_segments if intermediate else None
            if raw_segs:
                raw_srt = segments_to_srt(raw_segs, include_speaker=True)
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix="_1_raw.srt", delete=False, encoding="utf-8"
                ) as f:
                    f.write(raw_srt)
                    raw_path = f.name

            # Stage 2: Aligned segments
            aligned_segs = intermediate.aligned_segments if intermediate else None
            if aligned_segs:
                aligned_srt = segments_to_srt(aligned_segs, include_speaker=True)
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix="_2_aligned.srt", delete=False, encoding="utf-8"
                ) as f:
                    f.write(aligned_srt)
                    aligned_path = f.name

            # Stage 3: Refined segments
            refined_segs = intermediate.refined_segments if intermediate else None
            if refined_segs:
                refined_srt = segments_to_srt(refined_segs, include_speaker=True)
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix="_3_refined.srt", delete=False, encoding="utf-8"
                ) as f:
                    f.write(refined_srt)
                    refined_path = f.name

            progress(1.0, desc=f"[{elapsed()}] 완료!")

            total_time = time.time() - start_time
            status = f"완료! 총 소요시간: {format_time(total_time)} ({total_time:.1f}초)\n"
            status += f"세그먼트: {len(result.segments)}개"
            if result.metadata.speakers:
                status += f" | 화자: {len(result.metadata.speakers)}명"

            return srt_content, srt_path, raw_path, aligned_path, refined_path, status

        except ChalnaError as e:
            total_time = time.time() - start_time
            return f"오류: {e.message}", None, None, None, None, f"실패 (경과: {format_time(total_time)})"
        except Exception as e:
            total_time = time.time() - start_time
            return f"오류: {str(e)}", None, None, None, None, f"실패 (경과: {format_time(total_time)})"

    # Build UI
    with gr.Blocks(title="찰나 (Chalna)") as app:
        gr.Markdown("# 찰나 (Chalna)")
        gr.Markdown("음성/영상 파일을 SRT 자막으로 변환합니다.")

        with gr.Row():
            with gr.Column():
                audio_input = gr.File(
                    label="파일 업로드",
                    file_types=[
                        ".mp3", ".wav", ".m4a", ".flac", ".ogg",
                        ".mp4", ".mov", ".webm", ".mkv", ".avi",
                    ],
                )
                file_info = gr.Textbox(
                    label="파일 정보",
                    interactive=False,
                    lines=3,
                )
                context_input = gr.Textbox(
                    label="Context (선택)",
                    placeholder="예: 참석자: 철수, 영희",
                    lines=2,
                )
                submit_btn = gr.Button("변환", variant="primary")

            with gr.Column():
                status_output = gr.Textbox(
                    label="진행 상태",
                    interactive=False,
                    lines=2,
                )
                srt_output = gr.Textbox(
                    label="SRT 출력",
                    lines=12,
                )
                download_output = gr.File(label="최종 SRT 다운로드")

                with gr.Accordion("중간 결과물", open=False):
                    raw_download = gr.File(label="1. Raw (VibeVoice)")
                    aligned_download = gr.File(label="2. Aligned (Qwen)")
                    refined_download = gr.File(label="3. Refined (LLM)")

        # Event handlers
        audio_input.change(
            fn=on_file_upload,
            inputs=[audio_input],
            outputs=[file_info],
        )

        submit_btn.click(
            fn=transcribe,
            inputs=[audio_input, context_input],
            outputs=[srt_output, download_output, raw_download, aligned_download, refined_download, status_output],
        )

    return app


def launch(
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
    device: str = "auto",
):
    """Launch Gradio server."""
    app = create_app(device=device)
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
    )
