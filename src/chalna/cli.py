"""
Chalna CLI - Command-line interface for SRT subtitle generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

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
        help="Skip Qwen forced alignment (faster, less accurate timestamps)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output JSON instead of SRT",
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

    console.print(f"[bold blue]Chalna (찰나)[/bold blue] - SRT Subtitle Generator")
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
            # Load models
            task = progress.add_task("Loading models...", total=None)

            from chalna.pipeline import ChalnaPipeline

            pipeline = ChalnaPipeline(
                device=device,
                use_alignment=not no_align,
            )

            progress.update(task, description="Transcribing audio...")

            # Run transcription
            result = pipeline.transcribe(
                audio_path=input_file,
                context=context,
                language=language,
            )

            progress.update(task, description="Writing output...")

        # Generate output
        if json_output:
            content = result.to_json()
        else:
            content = result.to_srt(include_speaker=not no_speaker)

        # Write output
        output.write_text(content, encoding="utf-8")

        # Summary
        console.print()
        console.print(f"[bold green]Done![/bold green]")
        console.print(f"  Segments: {len(result.segments)}")
        console.print(f"  Duration: {result.metadata.duration:.1f}s")
        if result.metadata.speakers:
            console.print(f"  Speakers: {', '.join(result.metadata.speakers)}")
        console.print(f"  Aligned:  {'Yes' if result.metadata.aligned else 'No'}")
        console.print()
        console.print(f"  Output saved to: {output}")

    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="Host to bind to",
    ),
    port: int = typer.Option(
        8000,
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

        chalna serve --port 8000

        chalna serve --host 127.0.0.1 --port 9000 --reload
    """
    import uvicorn

    console.print(f"[bold blue]Chalna (찰나)[/bold blue] - REST API Server")
    console.print()
    console.print(f"  Host: {host}")
    console.print(f"  Port: {port}")
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
