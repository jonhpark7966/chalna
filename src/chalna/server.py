"""
Chalna REST API Server.
"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field

from chalna import __version__
from chalna.exceptions import ChalnaError
from chalna.validation import validate_audio_file


# =============================================================================
# App Setup
# =============================================================================

app = FastAPI(
    title="Chalna (찰나)",
    description="SRT subtitle generation service with speaker diarization",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Exception Handlers
# =============================================================================


@app.exception_handler(ChalnaError)
async def chalna_error_handler(request: Request, exc: ChalnaError):
    """Handle all Chalna-specific exceptions with structured error response."""
    return JSONResponse(
        status_code=exc.http_status,
        content=exc.to_dict(),
    )


# Global pipeline instance (lazy loaded)
_pipeline = None

# Job storage (in-memory, for demo)
_jobs: Dict[str, "Job"] = {}


def get_pipeline():
    """Get or create pipeline instance."""
    global _pipeline
    if _pipeline is None:
        from chalna.pipeline import ChalnaPipeline
        _pipeline = ChalnaPipeline()
    return _pipeline


# =============================================================================
# Models
# =============================================================================

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0.0
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    output_format: str = "srt"
    alignment_log: Optional[List[dict]] = None
    progress_history: List[Dict[str, Any]] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Structured error response."""
    error: bool = True
    error_code: str
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None


class TranscriptionProgress(BaseModel):
    """Progress update during transcription."""
    stage: str  # "validating", "loading_models", "transcribing", "aligning"
    progress: float  # 0.0 ~ 1.0


class TranscribeResponse(BaseModel):
    job_id: str
    status: JobStatus
    estimated_time: Optional[int] = None  # seconds


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float
    result: Optional[str] = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    alignment_log: Optional[List[dict]] = None
    progress_history: List[Dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    version: str
    models: Dict[str, str]
    gpu: Optional[str] = None


class SegmentModel(BaseModel):
    index: int
    start_time: float
    end_time: float
    text: str
    speaker_id: Optional[str] = None
    confidence: float = 1.0


class MetadataModel(BaseModel):
    duration: float
    language: Optional[str] = None
    speakers: list[str] = Field(default_factory=list)
    model_version: str = "vibevoice-asr"
    aligned: bool = True


class TranscriptionResultModel(BaseModel):
    segments: list[SegmentModel]
    metadata: MetadataModel


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Check server health and model status.
    """
    import torch

    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)

    # Check model status
    models = {
        "vibevoice": "not_loaded",
        "qwen_aligner": "not_loaded",
    }

    if _pipeline is not None:
        if _pipeline._vibevoice_model is not None:
            models["vibevoice"] = "loaded"
        if _pipeline._aligner is not None:
            models["qwen_aligner"] = "loaded"

    return HealthResponse(
        status="ok",
        version=__version__,
        models=models,
        gpu=gpu,
    )


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="Audio or video file"),
    context: Optional[str] = Form(None, description="Context/hotwords for better accuracy"),
    language: Optional[str] = Form(None, description="Language hint (ko, en, ja, zh)"),
    include_speaker: bool = Form(True, description="Include speaker labels"),
    use_alignment: bool = Form(True, description="Use Qwen forced alignment"),
    output_format: Literal["srt", "json"] = Form("srt", description="Output format"),
    include_logs: bool = Form(False, description="Include alignment logs in JSON output"),
):
    """
    Transcribe audio/video file to subtitles (synchronous).

    Returns SRT or JSON based on output_format parameter.

    Error responses follow the structure:
    {
        "error": true,
        "error_code": "E1001",
        "error_type": "AudioTooLongError",
        "message": "Audio duration exceeds...",
        "details": {...}
    }
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Save uploaded file
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Validate audio file first (duration, format, integrity)
        # This will raise ChalnaError exceptions handled by the exception handler
        validate_audio_file(tmp_path)

        # Get pipeline
        pipeline = get_pipeline()
        pipeline.use_alignment = use_alignment

        # Run transcription
        result = pipeline.transcribe(
            audio_path=tmp_path,
            context=context,
            language=language,
        )

        # Format output
        if output_format == "json":
            response_data = result.to_dict()

            # Include alignment logs if requested
            if include_logs:
                response_data["alignment_log"] = pipeline.get_alignment_log()

            return JSONResponse(
                content=response_data,
                media_type="application/json",
            )
        else:
            return PlainTextResponse(
                content=result.to_srt(include_speaker=include_speaker),
                media_type="text/plain; charset=utf-8",
            )

    except ChalnaError:
        # Re-raise ChalnaError to be handled by the exception handler
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup
        tmp_path.unlink(missing_ok=True)


@app.post("/transcribe/async", response_model=TranscribeResponse)
async def transcribe_async(
    file: UploadFile = File(..., description="Audio or video file"),
    context: Optional[str] = Form(None, description="Context/hotwords for better accuracy"),
    language: Optional[str] = Form(None, description="Language hint (ko, en, ja, zh)"),
    include_speaker: bool = Form(True, description="Include speaker labels"),
    use_alignment: bool = Form(True, description="Use Qwen forced alignment"),
    output_format: Literal["srt", "json"] = Form("srt", description="Output format"),
    include_logs: bool = Form(False, description="Include alignment logs in result"),
):
    """
    Transcribe audio/video file asynchronously (for long files).

    Returns job_id immediately. Poll /jobs/{job_id} for status.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Save uploaded file
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    # Validate audio file upfront (fail fast)
    try:
        validate_audio_file(tmp_path)
    except ChalnaError:
        # Cleanup and re-raise for exception handler
        tmp_path.unlink(missing_ok=True)
        raise

    # Create job
    job_id = str(uuid.uuid4())
    job = Job(
        job_id=job_id,
        status=JobStatus.QUEUED,
        created_at=datetime.utcnow(),
        output_format=output_format,
    )
    _jobs[job_id] = job

    # Start background task
    asyncio.create_task(
        _process_job(
            job_id=job_id,
            audio_path=tmp_path,
            context=context,
            language=language,
            include_speaker=include_speaker,
            use_alignment=use_alignment,
            output_format=output_format,
            include_logs=include_logs,
        )
    )

    # Estimate time (rough: 1 min audio = 10 sec processing)
    file_size_mb = len(content) / (1024 * 1024)
    estimated_time = int(file_size_mb * 20)  # rough estimate

    return TranscribeResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        estimated_time=estimated_time,
    )


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """
    Get status of an async transcription job.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        result=job.result,
        error=job.error,
        error_details=job.error_details,
        alignment_log=job.alignment_log,
        progress_history=job.progress_history,
    )


# =============================================================================
# Background Tasks
# =============================================================================

async def _process_job(
    job_id: str,
    audio_path: Path,
    context: Optional[str],
    language: Optional[str],
    include_speaker: bool,
    use_alignment: bool,
    output_format: str,
    include_logs: bool = False,
):
    """Process transcription job in background."""
    job = _jobs[job_id]

    def progress_callback(stage: str, progress: float):
        """Update job progress from pipeline."""
        entry = {
            "stage": stage,
            "progress": progress,
            "timestamp": datetime.utcnow().isoformat(),
        }
        job.progress_history.append(entry)

        # Map stage progress to overall job progress
        stage_weights = {
            "validating": (0.0, 0.05),
            "loading_models": (0.05, 0.3),
            "transcribing": (0.3, 0.8),
            "aligning": (0.8, 1.0),
        }
        if stage in stage_weights:
            start, end = stage_weights[stage]
            job.progress = start + (end - start) * progress

    try:
        job.status = JobStatus.PROCESSING
        job.progress = 0.0

        # Get pipeline
        pipeline = get_pipeline()
        pipeline.use_alignment = use_alignment

        # Run transcription (in thread pool to not block)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.transcribe(
                audio_path=audio_path,
                context=context,
                language=language,
                progress_callback=progress_callback,
            )
        )

        job.progress = 0.95

        # Include alignment logs if requested
        if include_logs:
            job.alignment_log = pipeline.get_alignment_log()

        # Format output
        if output_format == "json":
            result_data = result.to_dict()
            if include_logs:
                result_data["alignment_log"] = job.alignment_log
            import json
            job.result = json.dumps(result_data, ensure_ascii=False, indent=2)
        else:
            job.result = result.to_srt(include_speaker=include_speaker)

        job.status = JobStatus.COMPLETED
        job.progress = 1.0
        job.completed_at = datetime.utcnow()

    except ChalnaError as e:
        job.status = JobStatus.FAILED
        job.error = e.message
        job.error_details = e.to_dict()

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)

    finally:
        # Cleanup
        audio_path.unlink(missing_ok=True)


# =============================================================================
# Startup/Shutdown
# =============================================================================

@app.on_event("startup")
async def startup():
    """Preload models on startup."""
    # Optionally preload models here
    # get_pipeline()
    pass


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    # Cleanup jobs
    _jobs.clear()
