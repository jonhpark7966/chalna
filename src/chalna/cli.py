"""
Chalna CLI - Command-line interface for SRT subtitle generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from chalna.exceptions import (
    AudioTooLongError,
    ChalnaError,
    CodexAPIError,
    CodexRateLimitError,
    CorruptedFileError,
    DiskSpaceError,
    ElevenLabsAPIError,
    EmptyTranscriptionError,
    FFmpegNotFoundError,
    FilePermissionError,
    FileTooLargeError,
    ModelDownloadError,
    ModelLoadError,
    OutOfMemoryError,
    TempFileError,
    UnsupportedFormatError,
)
from chalna.models import LlmSegmentationOptions, ScribeOptions

app = typer.Typer(
    name="chalna",
    help="찰나 (Chalna) - SRT subtitle generation with speaker diarization",
    add_completion=False,
)
console = Console()


@app.command()
def transcribe(
    input_file: Path = typer.Argument(
        ...,
        help="Input audio/video file path",
        exists=True,
        readable=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "-o", "--output",
        help="Output file path (default: input name with .srt extension)",
    ),
    context: Optional[str] = typer.Option(
        None,
        "-c", "--context",
        help="Context/hotwords for better accuracy (e.g., '참석자: 철수, 영희')",
    ),
    language: Optional[str] = typer.Option(
        None,
        "-l", "--language",
        help="Language hint (ko, en, ja, zh, etc.)",
    ),
    no_speaker: bool = typer.Option(
        False,
        "--no-speaker",
        help="Exclude speaker labels from output",
    ),
    no_align: bool = typer.Option(
        False,
        "--no-align",
        help="Deprecated; ignored because Qwen forced alignment is disabled",
    ),
    llm_refine: bool = typer.Option(
        True,
        "--llm-refine/--no-llm-refine",
        help="Use LLM to refine Scribe subtitles (requires Codex CLI)",
    ),
    llm_segmentation: bool = typer.Option(
        True,
        "--llm-segmentation/--no-llm-segmentation",
        help="Use LLM to plan Scribe word-to-subtitle segment boundaries",
    ),
    segmentation_boundary_rule: str = typer.Option(
        "word_boundary",
        "--segmentation-boundary-rule",
        help="Timestamp boundary rule: word_boundary, midpoint_gap, low_energy_gap_v1",
    ),
    diarize: bool = typer.Option(
        True,
        "--diarize/--no-diarize",
        help="Enable Scribe speaker diarization",
    ),
    tag_audio_events: bool = typer.Option(
        True,
        "--tag-audio-events/--no-tag-audio-events",
        help="Include Scribe non-speech audio event tags",
    ),
    num_speakers: Optional[int] = typer.Option(
        None,
        "--num-speakers",
        help="Expected speaker count for Scribe diarization (1-32)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output JSON instead of SRT",
    ),
    save_intermediate: bool = typer.Option(
        False,
        "--save-intermediate",
        help="Save raw Scribe/refinement intermediate results",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device to use (cuda, cpu, mps, xpu, auto)",
    ),
    verbose: bool = typer.Option(
        False,
        "-v", "--verbose",
        help="Show detailed logs",
    ),
):
    """
    Transcribe audio/video file to SRT subtitles.

    Examples:

        chalna transcribe meeting.mp3 -o meeting.srt

        chalna transcribe lecture.wav -c "강사: 김교수" --no-speaker

        chalna transcribe interview.m4a --json -o interview.json
    """
    # Determine output path
    if output is None:
        suffix = ".json" if json_output else ".srt"
        output = input_file.with_suffix(suffix)

    console.print("[bold blue]Chalna (찰나)[/bold blue] - SRT Subtitle Generator")
    console.print()
    console.print(f"  Input:  {input_file}")
    console.print(f"  Output: {output}")
    if context:
        console.print(f"  Context: {context}")
    console.print()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Transcribing with Scribe v2...", total=None)

            from chalna.pipeline import ChalnaPipeline

            pipeline = ChalnaPipeline(
                device=device,
                use_alignment=False,
                use_llm_refinement=llm_refine,
                use_llm_segmentation=llm_segmentation,
            )

            progress.update(task, description="Transcribing audio...")

            # Run transcription
            result = pipeline.transcribe(
                audio_path=input_file,
                context=context,
                language=language,
                verbose=verbose,
                scribe_options=ScribeOptions(
                    diarize=diarize,
                    tag_audio_events=tag_audio_events,
                    num_speakers=num_speakers,
                ),
                llm_segmentation_options=LlmSegmentationOptions(
                    enabled=llm_segmentation,
                    boundary_rule=segmentation_boundary_rule,
                ),
            )

            progress.update(task, description="Writing output...")

            # Save intermediate results (always save stage SRTs)
            import json as json_module

            from chalna.srt_utils import segments_to_srt

            base_path = output.parent / output.stem

            # Get intermediate results from result (thread-safe)
            intermediate = result.intermediate

            # Stage 1: Raw segments (Scribe)
            raw_segments = intermediate.raw_segments if intermediate else None
            if raw_segments:
                raw_srt_path = base_path.parent / f"{base_path.name}_1_raw.srt"
                raw_srt_content = segments_to_srt(raw_segments, include_speaker=True)
                raw_srt_path.write_text(raw_srt_content, encoding="utf-8")
                console.print(f"  Raw SRT: {raw_srt_path}")

            # Stage 2: Refined segments (LLM)
            refined_segments = intermediate.refined_segments if intermediate else None
            if refined_segments:
                refined_srt_path = base_path.parent / f"{base_path.name}_2_refined.srt"
                refined_srt_content = segments_to_srt(refined_segments, include_speaker=True)
                refined_srt_path.write_text(refined_srt_content, encoding="utf-8")
                console.print(f"  Refined SRT: {refined_srt_path}")

            # Save detailed logs only if --save-intermediate is set
            if save_intermediate:
                from chalna.models import TranscriptionMetadata, TranscriptionResult

                # Save raw Scribe JSON
                if raw_segments:
                    pre_json_path = base_path.parent / f"{base_path.name}_1_raw.json"
                    pre_result = TranscriptionResult(
                        segments=raw_segments,
                        metadata=TranscriptionMetadata(
                            duration=max((s.end_time for s in raw_segments), default=0.0),
                            language=language,
                            speakers=list(set(s.speaker_id for s in raw_segments if s.speaker_id)),
                            model_version="scribe_v2",
                            aligned=False,
                            timestamp_source="scribe_v2",
                        )
                    )
                    pre_json_path.write_text(pre_result.to_json(), encoding="utf-8")
                    console.print(f"  Raw JSON: {pre_json_path}")

                # Save alignment log
                alignment_log = intermediate.alignment_log if intermediate else None
                if alignment_log:
                    log_path = base_path.parent / f"{base_path.name}_alignment.log"
                    log_lines = []
                    for entry in alignment_log:
                        status = entry["status"]
                        idx = entry.get("index", 0)
                        text = entry.get("text", "")
                        orig = entry.get("original", {})

                        if status == "aligned":
                            log_lines.append(
                                f"[{idx:3d}] ALIGNED: "
                                f"{orig['start']:.2f}→{entry['aligned']['start']:.2f} "
                                f"({entry['delta']['start']:+.2f}s) | "
                                f"{orig['end']:.2f}→{entry['aligned']['end']:.2f} "
                                f"({entry['delta']['end']:+.2f}s) | {text}"
                            )
                        elif status == "rejected_expand":
                            log_lines.append(
                                f"[{idx:3d}] REJECTED: would expand - kept original "
                                f"{orig['start']:.2f}→{orig['end']:.2f} | {text}"
                            )
                        elif status == "invalid_duration":
                            log_lines.append(
                                f"[{idx:3d}] INVALID: duration too short - kept original "
                                f"{orig['start']:.2f}→{orig['end']:.2f} | {text}"
                            )
                        elif status == "no_result":
                            log_lines.append(
                                f"[{idx:3d}] NO_RESULT: kept original "
                                f"{orig['start']:.2f}→{orig['end']:.2f} | {text}"
                            )
                        elif status == "failed":
                            log_lines.append(
                                f"[{idx:3d}] FAILED: {entry.get('error', 'unknown')} | {text}"
                            )
                        else:
                            log_lines.append(
                                f"[{idx:3d}] {status.upper()}: kept original "
                                f"{orig.get('start', 0):.2f}→{orig.get('end', 0):.2f} | {text}"
                            )
                    log_path.write_text("\n".join(log_lines), encoding="utf-8")
                    console.print(f"  Alignment log: {log_path}")

                # Save refinement log
                refinement_log = intermediate.refinement_log if intermediate else None
                if refinement_log:
                    refine_log_path = base_path.parent / f"{base_path.name}_refinement.json"
                    refine_log_path.write_text(
                        json_module.dumps(refinement_log, indent=2, ensure_ascii=False),
                        encoding="utf-8"
                    )
                    console.print(f"  Refinement log: {refine_log_path}")

        # Generate output
        if json_output:
            content = result.to_json()
        else:
            content = result.to_srt(include_speaker=not no_speaker)

        # Write output
        output.write_text(content, encoding="utf-8")

        # Summary
        console.print()
        console.print("[bold green]Done![/bold green]")
        console.print(f"  Segments: {len(result.segments)}")
        console.print(f"  Duration: {result.metadata.duration:.1f}s")
        if result.metadata.speakers:
            console.print(f"  Speakers: {', '.join(result.metadata.speakers)}")
        console.print(f"  Aligned:  {'Yes' if result.metadata.aligned else 'No'}")
        console.print()
        console.print(f"  Output saved to: {output}")

    except AudioTooLongError as e:
        console.print("[bold red]Error:[/bold red] Audio too long")
        console.print(f"  Duration: {e.details['duration_seconds']:.1f}s")
        console.print(f"  Maximum: {e.details['max_duration_seconds']:.0f}s (10 hours)")
        console.print()
        console.print("[dim]Tip: Split long audio files using ffmpeg:[/dim]")
        console.print("[dim]  ffmpeg -i input.mp3 -ss 0 -t 3600 part1.mp3[/dim]")
        raise typer.Exit(code=1)

    except FileTooLargeError as e:
        console.print("[bold red]Error:[/bold red] File too large")
        console.print(f"  File size: {e.details['file_size_mb']:.1f} MB")
        console.print(
            f"  Maximum: {e.details['max_size_mb']:.0f} MB "
            f"({e.details['max_size_mb'] / 1024:.1f} GB)"
        )
        console.print()
        console.print("[dim]Tip: Compress or split the file:[/dim]")
        console.print("[dim]  ffmpeg -i input.mp4 -b:a 128k compressed.mp4[/dim]")
        raise typer.Exit(code=1)

    except UnsupportedFormatError as e:
        console.print(f"[bold red]Error:[/bold red] Unsupported format: {e.details['format']}")
        console.print()
        console.print("[dim]Supported formats:[/dim]")
        console.print("[dim]  Audio: mp3, wav, flac, aac, ogg, opus, m4a, wma[/dim]")
        console.print("[dim]  Video: mp4, mov, webm, mkv, avi[/dim]")
        raise typer.Exit(code=1)

    except CorruptedFileError as e:
        console.print("[bold red]Error:[/bold red] Corrupted or unreadable file")
        if e.details.get("reason"):
            console.print(f"  Reason: {e.details['reason']}")
        console.print()
        console.print("[dim]Tip: Verify the file with ffprobe:[/dim]")
        console.print(f"[dim]  ffprobe \"{input_file}\"[/dim]")
        raise typer.Exit(code=1)

    except FilePermissionError as e:
        console.print("[bold red]Error:[/bold red] Permission denied")
        console.print(f"  Cannot read: {e.details['file_path']}")
        console.print()
        console.print("[dim]Tip: Check file permissions[/dim]")
        raise typer.Exit(code=1)

    except OutOfMemoryError as e:
        console.print(f"[bold red]Error:[/bold red] {e.details['memory_type']} memory exhausted")
        console.print()
        console.print("[dim]Suggestions:[/dim]")
        console.print("  - Use shorter audio files")
        if e.details['memory_type'] == "GPU":
            console.print("  - Use --device cpu to run on CPU instead")
            console.print("  - Close other GPU-intensive applications")
        else:
            console.print("  - Free up system memory")
        raise typer.Exit(code=1)

    except EmptyTranscriptionError as e:
        console.print("[bold yellow]Warning:[/bold yellow] No speech detected in audio")
        if e.details.get("audio_duration"):
            console.print(f"  Audio duration: {e.details['audio_duration']:.1f}s")
        console.print()
        console.print("[dim]Possible causes:[/dim]")
        console.print("  - Audio contains no speech")
        console.print("  - Audio is too quiet")
        console.print("  - Audio is heavily distorted")
        raise typer.Exit(code=0)  # Exit 0 since this isn't an error per se

    except ModelLoadError as e:
        console.print(
            f"[bold red]Error:[/bold red] Failed to load model: "
            f"{e.details['model_name']}"
        )
        if e.details.get("reason"):
            console.print(f"  Reason: {e.details['reason']}")
        console.print()
        console.print("[dim]Suggestions:[/dim]")
        console.print("  - Ensure model files are not corrupted")
        console.print("  - Try clearing HuggingFace cache: rm -rf ~/.cache/huggingface")
        raise typer.Exit(code=1)

    except ModelDownloadError as e:
        console.print(
            f"[bold red]Error:[/bold red] Failed to download model: "
            f"{e.details['model_name']}"
        )
        console.print()
        console.print("[dim]Suggestions:[/dim]")
        console.print("  - Check your internet connection")
        console.print("  - Try again later")
        console.print("  - Set HF_HUB_OFFLINE=1 to use cached models")
        raise typer.Exit(code=1)

    except ElevenLabsAPIError as e:
        console.print("[bold red]Error:[/bold red] ElevenLabs Scribe error")
        console.print(f"  {e.message}")
        console.print()
        console.print("[dim]Suggestions:[/dim]")
        console.print("  - Set ELEVENLABS_API_KEY in your environment or .env")
        console.print("  - Check your ElevenLabs quota and network connectivity")
        raise typer.Exit(code=1)

    except CodexAPIError as e:
        console.print("[bold red]Error:[/bold red] Codex CLI failed")
        console.print(f"  Reason: {e.details['reason']}")
        console.print()
        console.print("[dim]Suggestions:[/dim]")
        console.print("  - Check that Codex CLI is installed: npm install -g @openai/codex")
        console.print("  - Run with --no-llm-refine to skip LLM refinement")
        raise typer.Exit(code=1)

    except CodexRateLimitError as e:
        console.print("[bold red]Error:[/bold red] Codex API rate limit exceeded")
        console.print(f"  Reason: {e.details['reason']}")
        console.print()
        console.print("[dim]Suggestions:[/dim]")
        console.print("  - Wait and try again later")
        console.print("  - Run with --no-llm-refine to skip LLM refinement")
        raise typer.Exit(code=1)

    except DiskSpaceError as e:
        console.print("[bold red]Error:[/bold red] Insufficient disk space")
        console.print(f"  Available: {e.details['available_mb']:.1f} MB")
        console.print(f"  Required: {e.details['required_mb']:.1f} MB")
        console.print()
        console.print("[dim]Tip: Free up disk space and try again[/dim]")
        raise typer.Exit(code=1)

    except TempFileError as e:
        console.print(
            f"[bold red]Error:[/bold red] Temporary file operation failed: "
            f"{e.details['operation']}"
        )
        console.print()
        console.print("[dim]Suggestions:[/dim]")
        console.print("  - Check temp directory permissions")
        console.print("  - Ensure sufficient disk space")
        raise typer.Exit(code=1)

    except FFmpegNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e.details['tool']} not found")
        console.print()
        console.print("[dim]Installation instructions:[/dim]")
        console.print("  Ubuntu/Debian: sudo apt install ffmpeg")
        console.print("  macOS: brew install ffmpeg")
        console.print("  Windows: choco install ffmpeg")
        raise typer.Exit(code=1)

    except ChalnaError as e:
        # Catch-all for any other Chalna errors
        console.print(f"[bold red]Error [{e.error_code.value}]:[/bold red] {e.message}")
        if verbose and e.details:
            console.print(f"  Details: {e.details}")
        raise typer.Exit(code=1)

    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


@app.command("validate-boundaries")
def validate_boundaries(
    audio_file: Path = typer.Argument(
        ...,
        help="Source audio/video file path",
        exists=True,
        readable=True,
    ),
    subtitles: Path = typer.Argument(
        ...,
        help="SRT file to validate against the source audio",
        exists=True,
        readable=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        help="Optional JSON report output path",
    ),
    scan_padding: float = typer.Option(
        0.5,
        "--scan-padding",
        help="Seconds of audio context to align before and after each segment",
    ),
    min_start_padding: float = typer.Option(
        0.15,
        "--min-start-padding",
        help="Minimum acceptable seconds before the first aligned word",
    ),
    min_end_padding: float = typer.Option(
        0.15,
        "--min-end-padding",
        help="Minimum acceptable seconds after the last aligned word",
    ),
    cutoff_tolerance: float = typer.Option(
        0.03,
        "--cutoff-tolerance",
        help="Seconds of tolerance before reporting speech cutoff instead of tightness",
    ),
    language: str = typer.Option(
        "Korean",
        "--alignment-language",
        help="Deprecated; ignored because Qwen forced alignment is disabled",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device to use (cuda, cpu, mps, xpu, auto)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print the full JSON report to stdout",
    ),
    fail_on_issue: bool = typer.Option(
        False,
        "--fail-on-issue",
        help="Exit with code 2 when warning/error issues are found",
    ),
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Show detailed exception logs",
    ),
):
    """
    Validate whether SRT segment boundaries are too tight for edit use.

    Qwen-based audio boundary realignment is disabled in the Scribe v2 pipeline,
    so this command currently reports SRT contract issues only.
    """
    import json as json_module

    from chalna.models import Segment
    from chalna.srt_utils import parse_srt
    from chalna.subtitle_validator import summarize_validation_diagnostics, validate_segments

    parsed_segments = parse_srt(subtitles.read_text(encoding="utf-8"))
    segments = [
        Segment(
            index=item["index"],
            start_time=item["start_time"],
            end_time=item["end_time"],
            text=item["text"],
            speaker_id=item.get("speaker_id"),
        )
        for item in parsed_segments
    ]

    if not segments:
        console.print(f"[bold red]Error:[/bold red] No SRT segments found in {subtitles}")
        raise typer.Exit(code=1)

    contract_issues = validate_segments(segments)

    if verbose:
        console.print(
            "[yellow]Audio boundary validation is disabled in the Scribe v2 pipeline.[/yellow]"
        )
    issues = contract_issues

    report = {
        "audio_file": str(audio_file),
        "subtitles": str(subtitles),
        "segments_checked": len(segments),
        "passed": not any(issue.severity in {"warning", "error"} for issue in issues),
        "thresholds": {
            "scan_padding": scan_padding,
            "min_start_padding": min_start_padding,
            "min_end_padding": min_end_padding,
            "cutoff_tolerance": cutoff_tolerance,
        },
        "diagnostics": summarize_validation_diagnostics(segments, issues),
        "issues": [issue.to_dict() for issue in issues],
    }
    report_json = json_module.dumps(report, ensure_ascii=False, indent=2)

    if output:
        output.write_text(report_json, encoding="utf-8")

    if json_output:
        console.print(report_json)
    else:
        console.print()
        console.print("[bold blue]Boundary validation[/bold blue]")
        console.print(f"  Segments checked: {len(segments)}")
        console.print(f"  Issues: {len(issues)}")
        diagnostics = report["diagnostics"]
        console.print(
            "  Diagnostics: "
            f"zero-gap={diagnostics['gap']['zero_gap']}, "
            f"same-speaker zero-gap={diagnostics['gap']['same_speaker_zero_gap']}, "
            f"low-match={diagnostics['alignment']['low_target_match_ratio_lt_0_7']}"
        )
        if output:
            console.print(f"  Report: {output}")

        for issue in issues[:20]:
            console.print(
                f"  [{issue.severity}] #{issue.segment_index} {issue.code}: "
                f"{issue.start_time:.2f}s→{issue.end_time:.2f}s"
            )

        if len(issues) > 20:
            console.print(f"  ... {len(issues) - 20} more issue(s)")

    if fail_on_issue and not report["passed"]:
        raise typer.Exit(code=2)


@app.command()
def serve(
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="Host to bind to",
    ),
    port: int = typer.Option(
        7861,
        "--port", "-p",
        help="Port to bind to",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable auto-reload for development",
    ),
    workers: int = typer.Option(
        1,
        "--workers", "-w",
        help="Number of worker processes",
    ),
):
    """
    Start the Chalna REST API server.

    Examples:

        chalna serve --port 7861

        chalna serve --host 127.0.0.1 --port 9000 --reload
    """
    import uvicorn

    console.print("[bold blue]Chalna (찰나)[/bold blue] - REST API Server")
    console.print()
    console.print(f"  Host: {host}")
    console.print(f"  Port: {port}")
    display_host = "localhost" if host == "0.0.0.0" else host
    console.print(f"  Web UI:   http://{display_host}:{port}/")
    console.print(f"  API Docs: http://{display_host}:{port}/docs")
    console.print()

    uvicorn.run(
        "chalna.server:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
    )


@app.command()
def version():
    """Show version information."""
    from chalna import __version__
    console.print(f"Chalna (찰나) v{__version__}")


if __name__ == "__main__":
    app()
