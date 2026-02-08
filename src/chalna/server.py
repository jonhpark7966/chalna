"""
Chalna REST API Server.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

# Results storage directory
RESULTS_DIR = Path(os.environ.get("CHALNA_RESULTS_DIR", "/home/jonhpark/workspace/chalna/results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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

# Job queue infrastructure (FIFO single worker)
_job_queue: asyncio.Queue = asyncio.Queue()
_queue_worker_task: Optional[asyncio.Task] = None
_job_events: Dict[str, asyncio.Event] = {}  # sync endpoint wait
_job_params: Dict[str, Dict[str, Any]] = {}  # queued job parameters


def get_pipeline():
    """Get or create pipeline instance."""
    global _pipeline
    if _pipeline is None:
        from chalna.pipeline import ChalnaPipeline
        _pipeline = ChalnaPipeline()

        # Configure auto-unload from environment
        auto_unload = os.environ.get("CHALNA_AUTO_UNLOAD", "false").lower() == "true"
        _pipeline.set_auto_unload(auto_unload)

    return _pipeline


def _compute_queue_position(job_id: str) -> Optional[int]:
    """Compute queue position for a job.

    Returns:
        None if completed/failed, 0 if processing, 1+ if queued (1 = next).
    """
    job = _jobs.get(job_id)
    if job is None:
        return None
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
        return None
    if job.status == JobStatus.PROCESSING:
        return 0
    # QUEUED: count how many QUEUED jobs were created before this one
    position = 1
    for other in _jobs.values():
        if other.job_id == job_id:
            continue
        if other.status == JobStatus.QUEUED and other.created_at < job.created_at:
            position += 1
    return position


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
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    output_format: str = "srt"
    alignment_log: Optional[List[dict]] = None
    refinement_log: Optional[List[dict]] = None
    progress_history: List[Dict[str, Any]] = Field(default_factory=list)
    # Intermediate results
    raw_srt: Optional[str] = None  # Stage 1: VibeVoice raw
    aligned_srt: Optional[str] = None  # Stage 2: After forced alignment
    refined_srt: Optional[str] = None  # Stage 3: After LLM refinement
    # Chunk observability
    chunks_completed: int = 0
    total_chunks: int = 0
    chunk_raw_srts: Dict[int, str] = Field(default_factory=dict)
    # Refinement status
    refined: Optional[bool] = None


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
    refinement_log: Optional[List[dict]] = None
    progress_history: List[Dict[str, Any]] = Field(default_factory=list)
    # Intermediate results
    raw_srt: Optional[str] = None
    aligned_srt: Optional[str] = None
    refined_srt: Optional[str] = None
    # Chunk observability
    chunks_completed: int = 0
    total_chunks: int = 0
    # Refinement status
    refined: Optional[bool] = None
    # Queue info
    queue_position: Optional[int] = None
    started_at: Optional[datetime] = None


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
    refined: bool = True


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
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass

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


@app.post("/unload")
async def unload_models(force: bool = False):
    """
    Manually unload models from GPU memory.

    Args:
        force: If True, also unload processor (normally kept for fast reload)

    Returns:
        Status message
    """
    global _pipeline

    if _pipeline is None:
        return {"status": "ok", "message": "No models loaded"}

    was_loaded = _pipeline.is_loaded()
    _pipeline.unload(force=force)

    if was_loaded:
        return {"status": "ok", "message": "Models unloaded successfully"}
    else:
        return {"status": "ok", "message": "No models were loaded"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="Audio or video file"),
    context: Optional[str] = Form(None, description="Context/hotwords for better accuracy"),
    language: Optional[str] = Form(None, description="Language hint (ko, en, ja, zh)"),
    include_speaker: bool = Form(True, description="Include speaker labels"),
    use_alignment: bool = Form(True, description="Use Qwen forced alignment"),
    use_llm_refinement: bool = Form(True, description="Use LLM to refine subtitles"),
    output_format: Literal["srt", "json"] = Form("srt", description="Output format"),
    include_logs: bool = Form(False, description="Include alignment/refinement logs in JSON output"),
    include_intermediate: bool = Form(False, description="Include intermediate stage results"),
):
    """
    Transcribe audio/video file to subtitles (synchronous).

    Routes through the job queue to prevent concurrent GPU access.
    Blocks until the job completes.

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

    # Validate audio file upfront (fail fast before queuing)
    try:
        validate_audio_file(tmp_path)
    except ChalnaError:
        tmp_path.unlink(missing_ok=True)
        raise

    # Create internal job and enqueue
    job_id = str(uuid.uuid4())
    job = Job(
        job_id=job_id,
        status=JobStatus.QUEUED,
        created_at=datetime.utcnow(),
        output_format=output_format,
    )
    _jobs[job_id] = job

    _job_params[job_id] = dict(
        audio_path=tmp_path,
        context=context,
        language=language,
        include_speaker=include_speaker,
        use_alignment=use_alignment,
        use_llm_refinement=use_llm_refinement,
        output_format=output_format,
        include_logs=include_logs,
        include_intermediate=include_intermediate,
    )

    event = asyncio.Event()
    _job_events[job_id] = event
    await _job_queue.put(job_id)

    # Wait for job to complete
    await event.wait()

    # Build response from completed job
    try:
        if job.status == JobStatus.FAILED:
            if job.error_details and "http_status" in job.error_details:
                return JSONResponse(
                    status_code=job.error_details["http_status"],
                    content={k: v for k, v in job.error_details.items() if k != "http_status"},
                )
            raise HTTPException(status_code=500, detail=job.error or "Unknown error")

        # COMPLETED — return result based on output_format
        if output_format == "json":
            result_data = json.loads(job.result) if job.result else {}
            return JSONResponse(content=result_data, media_type="application/json")
        else:
            return PlainTextResponse(
                content=job.result or "",
                media_type="text/plain; charset=utf-8",
            )
    finally:
        # Cleanup sync job from storage (no need to poll)
        _jobs.pop(job_id, None)


@app.post("/transcribe/async", response_model=TranscribeResponse)
async def transcribe_async(
    file: UploadFile = File(..., description="Audio or video file"),
    context: Optional[str] = Form(None, description="Context/hotwords for better accuracy"),
    language: Optional[str] = Form(None, description="Language hint (ko, en, ja, zh)"),
    include_speaker: bool = Form(True, description="Include speaker labels"),
    use_alignment: bool = Form(True, description="Use Qwen forced alignment"),
    use_llm_refinement: bool = Form(True, description="Use LLM to refine subtitles"),
    output_format: Literal["srt", "json"] = Form("srt", description="Output format"),
    include_logs: bool = Form(False, description="Include alignment/refinement logs in result"),
    include_intermediate: bool = Form(False, description="Include intermediate stage results"),
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

    # Enqueue for processing
    _job_params[job_id] = dict(
        audio_path=tmp_path,
        context=context,
        language=language,
        include_speaker=include_speaker,
        use_alignment=use_alignment,
        use_llm_refinement=use_llm_refinement,
        output_format=output_format,
        include_logs=include_logs,
        include_intermediate=include_intermediate,
    )
    await _job_queue.put(job_id)

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
        refinement_log=job.refinement_log,
        progress_history=job.progress_history,
        raw_srt=job.raw_srt,
        aligned_srt=job.aligned_srt,
        refined_srt=job.refined_srt,
        chunks_completed=job.chunks_completed,
        total_chunks=job.total_chunks,
        refined=job.refined,
        queue_position=_compute_queue_position(job_id),
        started_at=job.started_at,
    )


@app.get("/jobs/{job_id}/chunks/{chunk_index}")
async def get_chunk_result(job_id: str, chunk_index: int):
    """
    Get raw ASR SRT result for a specific chunk.

    Returns the per-chunk VibeVoice output before merging/alignment.
    Available once the chunk has been processed (check chunks_completed in job status).
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]

    if chunk_index not in job.chunk_raw_srts:
        if chunk_index < 0 or chunk_index >= job.total_chunks:
            raise HTTPException(
                status_code=404,
                detail=f"Chunk index {chunk_index} out of range (0-{job.total_chunks - 1})",
            )
        raise HTTPException(
            status_code=404,
            detail=f"Chunk {chunk_index} not yet available (completed: {job.chunks_completed}/{job.total_chunks})",
        )

    return PlainTextResponse(
        content=job.chunk_raw_srts[chunk_index],
        media_type="text/plain; charset=utf-8",
    )


# =============================================================================
# Background Tasks
# =============================================================================

async def _queue_worker():
    """Single FIFO worker that processes jobs sequentially."""
    while True:
        job_id = await _job_queue.get()
        try:
            job = _jobs.get(job_id)
            if job is None or job.status != JobStatus.QUEUED:
                continue
            params = _job_params.pop(job_id, None)
            if params is None:
                continue
            await _process_job(job_id=job_id, **params)
        except Exception as e:
            job = _jobs.get(job_id)
            if job and job.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
                job.status = JobStatus.FAILED
                job.error = f"Queue worker error: {e}"
        finally:
            _job_queue.task_done()
            event = _job_events.pop(job_id, None)
            if event:
                event.set()


async def _process_job(
    job_id: str,
    audio_path: Path,
    context: Optional[str],
    language: Optional[str],
    include_speaker: bool,
    use_alignment: bool,
    use_llm_refinement: bool,
    output_format: str,
    include_logs: bool = False,
    include_intermediate: bool = False,
):
    """Process transcription job in background."""
    job = _jobs[job_id]

    def progress_callback(stage: str, progress: float, **kwargs):
        """Update job progress from pipeline."""
        entry = {
            "stage": stage,
            "progress": progress,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs,
        }
        job.progress_history.append(entry)

        # Track chunk progress
        if "chunk" in kwargs:
            job.chunks_completed = kwargs["chunk"]
            job.total_chunks = kwargs.get("total_chunks", 0)

        # Map stage progress to overall job progress
        stage_weights = {
            "validating": (0.0, 0.05),
            "loading_models": (0.05, 0.25),
            "transcribing": (0.25, 0.55),
            "aligning": (0.55, 0.75),
            "refining": (0.75, 0.95),
        }
        if stage in stage_weights:
            start, end = stage_weights[stage]
            job.progress = start + (end - start) * progress

    try:
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.utcnow()
        job.progress = 0.0

        # Get pipeline
        pipeline = get_pipeline()
        pipeline.use_alignment = use_alignment
        pipeline.use_llm_refinement = use_llm_refinement

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

        # Include logs if requested (use result.intermediate for thread safety)
        if include_logs and result.intermediate:
            job.alignment_log = result.intermediate.alignment_log
            job.refinement_log = result.intermediate.refinement_log

        # Include intermediate results if requested
        if include_intermediate and result.intermediate:
            from chalna.srt_utils import segments_to_srt

            if result.intermediate.raw_segments:
                job.raw_srt = segments_to_srt(
                    result.intermediate.raw_segments, include_speaker=True
                )

            if result.intermediate.aligned_segments:
                job.aligned_srt = segments_to_srt(
                    result.intermediate.aligned_segments, include_speaker=True
                )

            if result.intermediate.refined_segments:
                job.refined_srt = segments_to_srt(
                    result.intermediate.refined_segments, include_speaker=True
                )

            # Store per-chunk raw SRTs for observability
            if result.intermediate.chunk_raw_segments:
                for i, chunk_segs in enumerate(result.intermediate.chunk_raw_segments):
                    job.chunk_raw_srts[i] = segments_to_srt(
                        chunk_segs, include_speaker=True
                    )

        # Format output
        if output_format == "json":
            result_data = result.to_dict()
            if include_logs:
                result_data["alignment_log"] = job.alignment_log
                result_data["refinement_log"] = job.refinement_log
            if include_intermediate:
                result_data["raw_srt"] = job.raw_srt
                result_data["aligned_srt"] = job.aligned_srt
                result_data["refined_srt"] = job.refined_srt
            import json
            job.result = json.dumps(result_data, ensure_ascii=False, indent=2)
        else:
            job.result = result.to_srt(include_speaker=include_speaker)

        job.refined = result.metadata.refined
        job.status = JobStatus.COMPLETED
        job.progress = 1.0
        job.completed_at = datetime.utcnow()

        # Save results to files
        _save_job_results(job, result, include_speaker)

    except ChalnaError as e:
        job.status = JobStatus.FAILED
        job.error = e.message
        job.error_details = {**e.to_dict(), "http_status": e.http_status}
        _save_job_error(job)

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        _save_job_error(job)

    finally:
        # Cleanup
        audio_path.unlink(missing_ok=True)


# =============================================================================
# Startup/Shutdown
# =============================================================================

def _save_job_results(job: Job, result, include_speaker: bool) -> None:
    """Save job results to files for persistence.

    Creates:
    - {job_id}.srt - Final SRT output
    - {job_id}.json - Full result with metadata
    - {job_id}_raw.srt - Raw VibeVoice output (if available)
    - {job_id}_aligned.srt - After alignment (if available)
    - {job_id}_refined.srt - After LLM refinement (if available)
    """
    try:
        job_dir = RESULTS_DIR / job.job_id[:8]  # Use first 8 chars for subdirectory
        job_dir.mkdir(parents=True, exist_ok=True)

        timestamp = job.created_at.strftime("%Y%m%d_%H%M%S")
        base_name = f"{timestamp}_{job.job_id[:8]}"

        # Save final SRT
        srt_content = result.to_srt(include_speaker=include_speaker)
        srt_path = job_dir / f"{base_name}.srt"
        srt_path.write_text(srt_content, encoding="utf-8")

        # Save JSON with full result
        json_data = {
            "job_id": job.job_id,
            "created_at": job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "status": job.status.value,
            "result": result.to_dict(),
            "progress_history": job.progress_history,
        }

        # Add intermediate results if available
        if result.intermediate:
            json_data["intermediate"] = {
                "alignment_log": result.intermediate.alignment_log,
                "refinement_log": result.intermediate.refinement_log,
            }

        json_path = job_dir / f"{base_name}.json"
        json_path.write_text(
            json.dumps(json_data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        # Save intermediate SRT files (generate from result.intermediate if not in job)
        from chalna.srt_utils import segments_to_srt

        if result.intermediate:
            if result.intermediate.raw_segments:
                raw_srt = job.raw_srt or segments_to_srt(result.intermediate.raw_segments, include_speaker=True)
                (job_dir / f"{base_name}_1_raw.srt").write_text(raw_srt, encoding="utf-8")

            if result.intermediate.aligned_segments:
                aligned_srt = job.aligned_srt or segments_to_srt(result.intermediate.aligned_segments, include_speaker=True)
                (job_dir / f"{base_name}_2_aligned.srt").write_text(aligned_srt, encoding="utf-8")

            if result.intermediate.refined_segments:
                refined_srt = job.refined_srt or segments_to_srt(result.intermediate.refined_segments, include_speaker=True)
                (job_dir / f"{base_name}_3_refined.srt").write_text(refined_srt, encoding="utf-8")

        print(f"Results saved to: {job_dir}")

    except Exception as e:
        print(f"Failed to save job results: {e}")


def _save_job_error(job: Job) -> None:
    """Save job error information for debugging."""
    try:
        job_dir = RESULTS_DIR / job.job_id[:8]
        job_dir.mkdir(parents=True, exist_ok=True)

        timestamp = job.created_at.strftime("%Y%m%d_%H%M%S")
        base_name = f"{timestamp}_{job.job_id[:8]}"

        error_data = {
            "job_id": job.job_id,
            "created_at": job.created_at.isoformat(),
            "status": job.status.value,
            "error": job.error,
            "error_details": job.error_details,
            "progress_history": job.progress_history,
        }

        error_path = job_dir / f"{base_name}_error.json"
        error_path.write_text(
            json.dumps(error_data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        print(f"Error saved to: {error_path}")

    except Exception as e:
        print(f"Failed to save job error: {e}")


@app.on_event("startup")
async def startup():
    """Start queue worker and preload models."""
    global _queue_worker_task
    _queue_worker_task = asyncio.create_task(_queue_worker())
    print(f"Job queue worker started")
    print(f"Results will be saved to: {RESULTS_DIR}")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global _queue_worker_task

    # Cancel queue worker
    if _queue_worker_task is not None:
        _queue_worker_task.cancel()
        try:
            await _queue_worker_task
        except asyncio.CancelledError:
            pass
        _queue_worker_task = None

    # Cleanup temp files from pending jobs
    for job_id, params in _job_params.items():
        audio_path = params.get("audio_path")
        if audio_path and hasattr(audio_path, "unlink"):
            audio_path.unlink(missing_ok=True)
    _job_params.clear()

    # Wake up any waiting sync events
    for event in _job_events.values():
        event.set()
    _job_events.clear()

    _jobs.clear()
