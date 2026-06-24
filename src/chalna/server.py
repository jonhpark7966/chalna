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
from datetime import timedelta as _timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from chalna import __version__
from chalna.db import count_jobs as db_count_jobs
from chalna.db import get_job as db_get_job
from chalna.db import init_db, migrate_from_results_dir
from chalna.db import list_jobs as db_list_jobs
from chalna.db import save_job as db_save_job
from chalna.exceptions import ChalnaError
from chalna.models import LlmSegmentationOptions, ScribeOptions
from chalna.monitoring import capture_job_exception, init_sentry
from chalna.segmentation_boundary import DEFAULT_BOUNDARY_RULE, normalize_boundary_rule
from chalna.settings import settings
from chalna.validation import validate_audio_file

# Results storage directory
RESULTS_DIR = Path(os.environ.get("CHALNA_RESULTS_DIR", "/home/jonhpark/workspace/chalna/results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# App Setup
# =============================================================================

init_sentry(
    dsn=settings.sentry_dsn,
    environment=settings.sentry_environment,
    release=settings.sentry_release,
    traces_sample_rate=settings.sentry_traces_sample_rate,
)

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
# Static Files (Web UI)
# =============================================================================

_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """Serve the built-in Web UI."""
    index_path = _STATIC_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>Chalna API</h1><p><a href='/docs'>API Docs</a></p>")


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

# Performance stats for ETA estimation (seconds of processing per second of audio)
# Updated with exponential moving average as jobs complete
_performance_stats: Dict[str, float] = {
    "asr_rate": 0.133,       # Scribe API latency per second of audio, adjusted by jobs
    "refine_rate": 0.067,    # ~4s per min of audio (with parallel refinement)
    "overhead": 10.0,        # flat overhead (validation, upload, formatting)
    "completed_count": 0,
}


def _make_scribe_options(
    *,
    diarize: bool,
    tag_audio_events: bool,
    num_speakers: Optional[int],
) -> ScribeOptions:
    try:
        return ScribeOptions(
            diarize=diarize,
            tag_audio_events=tag_audio_events,
            num_speakers=num_speakers,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _make_llm_segmentation_options(
    *,
    use_llm_segmentation: bool,
    bypass_llm_segmentation_cache: bool = False,
    segmentation_boundary_rule: str = DEFAULT_BOUNDARY_RULE,
) -> LlmSegmentationOptions:
    return LlmSegmentationOptions(
        enabled=use_llm_segmentation,
        model=settings.llm_segmentation_model,
        reasoning_effort=settings.llm_segmentation_reasoning_effort,
        bypass_cache=bypass_llm_segmentation_cache,
        boundary_rule=normalize_boundary_rule(segmentation_boundary_rule),
    )


def _validate_segmentation_boundary_rule(value: str) -> str:
    try:
        return normalize_boundary_rule(value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


async def _read_overlap_intervals_upload(
    overlap_intervals: Optional[UploadFile],
) -> dict[str, Any] | list[Any] | None:
    if overlap_intervals is None or not overlap_intervals.filename:
        return None
    try:
        payload = json.loads((await overlap_intervals.read()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid overlap intervals JSON") from exc
    if not isinstance(payload, (dict, list)):
        raise HTTPException(
            status_code=400,
            detail="Overlap intervals JSON must be an object or list",
        )
    return payload


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


def _estimate_job_duration(job: "Job") -> float:
    """Estimate total processing time for a job in seconds."""
    if job.audio_duration is None:
        return 60.0  # fallback

    duration = job.audio_duration
    rate = _performance_stats["asr_rate"]
    if job.use_llm_segmentation:
        rate += _performance_stats["refine_rate"]
    if job.use_llm_refinement:
        rate += _performance_stats["refine_rate"]

    return duration * rate + _performance_stats["overhead"]


def _estimate_remaining(job: "Job") -> float:
    """Estimate remaining processing time for a job that's already running."""
    estimated_total = _estimate_job_duration(job)

    if job.status == JobStatus.PROCESSING and job.started_at:
        elapsed = (datetime.utcnow() - job.started_at).total_seconds()
        return max(0.0, estimated_total - elapsed)

    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
        return 0.0

    return estimated_total


def _compute_eta(job_id: str) -> Dict[str, Any]:
    """Compute ETA for a job including queue wait time.

    Returns dict with:
        wait_seconds: time until job starts processing
        processing_seconds: estimated processing time
        completion: estimated completion datetime
    """
    job = _jobs.get(job_id)
    if job is None:
        return {}

    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
        return {}

    processing_seconds = _estimate_job_duration(job)

    if job.status == JobStatus.PROCESSING:
        remaining = _estimate_remaining(job)
        return {
            "wait_seconds": 0.0,
            "processing_seconds": remaining,
            "completion": datetime.utcnow() + _timedelta(seconds=remaining),
        }

    # QUEUED: sum up remaining time for all jobs ahead
    wait = 0.0
    for other in _jobs.values():
        if other.job_id == job_id:
            continue
        if other.status == JobStatus.PROCESSING:
            wait += _estimate_remaining(other)
        elif other.status == JobStatus.QUEUED and other.created_at < job.created_at:
            wait += _estimate_job_duration(other)

    completion = datetime.utcnow() + _timedelta(seconds=wait + processing_seconds)

    return {
        "wait_seconds": round(wait, 1),
        "processing_seconds": round(processing_seconds, 1),
        "completion": completion,
    }


def _update_performance_stats(job: "Job") -> None:
    """Update rolling average performance stats when a job completes."""
    if job.audio_duration is None or job.audio_duration <= 0:
        return
    if not job.started_at or not job.completed_at:
        return

    # Parse stage durations from progress_history
    stage_times: Dict[str, List[str]] = {}
    for entry in job.progress_history:
        stage = entry.get("stage", "")
        ts = entry.get("timestamp", "")
        if stage and ts:
            stage_times.setdefault(stage, []).append(ts)

    def _ts_range(timestamps: List[str]) -> float:
        if len(timestamps) < 2:
            return 0.0
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                t0 = datetime.strptime(timestamps[0], fmt).timestamp()
                t1 = datetime.strptime(timestamps[-1], fmt).timestamp()
                return max(0.0, t1 - t0)
            except ValueError:
                continue
        return 0.0

    dur = job.audio_duration
    total_elapsed = (job.completed_at - job.started_at).total_seconds()

    asr_time = _ts_range(stage_times.get("transcribing", []))
    refine_time = _ts_range(stage_times.get("refining", []))
    overhead = max(0.0, total_elapsed - asr_time - refine_time)

    # Exponential moving average (alpha=0.3 gives recent data more weight)
    alpha = 0.3
    n = _performance_stats["completed_count"]

    if n == 0:
        # First job: replace defaults entirely
        alpha = 1.0

    _performance_stats["asr_rate"] = (
        alpha * (asr_time / dur) + (1 - alpha) * _performance_stats["asr_rate"]
    )
    if job.use_llm_refinement and refine_time > 0:
        _performance_stats["refine_rate"] = (
            alpha * (refine_time / dur) + (1 - alpha) * _performance_stats["refine_rate"]
        )
    _performance_stats["overhead"] = (
        alpha * overhead + (1 - alpha) * _performance_stats["overhead"]
    )
    _performance_stats["completed_count"] = n + 1


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
    segmentation_log: Optional[List[dict]] = None
    refinement_log: Optional[List[dict]] = None
    progress_history: List[Dict[str, Any]] = Field(default_factory=list)
    # Intermediate results
    raw_srt: Optional[str] = None  # Stage 1: Scribe raw
    aligned_srt: Optional[str] = None  # Deprecated: Qwen alignment is disabled
    refined_srt: Optional[str] = None  # Stage 2: After LLM refinement
    raw_scribe_response: Optional[Dict[str, Any]] = None
    # Chunk observability
    chunks_completed: int = 0
    total_chunks: int = 0
    chunk_raw_srts: Dict[int, str] = Field(default_factory=dict)
    # Refinement status
    refined: Optional[bool] = None
    # ETA estimation
    audio_duration: Optional[float] = None  # seconds
    use_alignment: bool = False
    use_llm_segmentation: bool = True
    bypass_llm_segmentation_cache: bool = False
    use_llm_refinement: bool = True


class ErrorResponse(BaseModel):
    """Structured error response."""
    error: bool = True
    error_code: str
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None


class TranscriptionProgress(BaseModel):
    """Progress update during transcription."""
    stage: str  # "validating", "transcribing", "refining"
    progress: float  # 0.0 ~ 1.0


class TranscribeResponse(BaseModel):
    job_id: str
    status: JobStatus
    estimated_wait_seconds: Optional[float] = None
    estimated_processing_seconds: Optional[float] = None
    estimated_completion: Optional[datetime] = None


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float
    result: Optional[str] = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    alignment_log: Optional[List[dict]] = None
    segmentation_log: Optional[List[dict]] = None
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
    # Audio info
    audio_duration: Optional[float] = None
    # Stage tracking
    current_stage: Optional[str] = None
    stage_timestamps: Dict[str, str] = Field(default_factory=dict)
    # ETA estimation
    estimated_wait_seconds: Optional[float] = None
    estimated_processing_seconds: Optional[float] = None
    estimated_completion: Optional[datetime] = None


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
    model_version: str = "scribe_v2"
    aligned: bool = False
    refined: bool = True
    timestamp_source: Optional[str] = None
    segmentation_source: Optional[str] = None
    segmentation_boundary_rule: Optional[str] = None
    segmentation_boundary_effective_rule: Optional[str] = None
    segmentation_boundary_stats: Optional[dict[str, Any]] = None


class TranscriptionResultModel(BaseModel):
    segments: list[SegmentModel]
    metadata: MetadataModel


class DoctorCheck(BaseModel):
    name: str
    ok: bool
    critical: bool = True
    detail: Optional[str] = None


class DoctorResponse(BaseModel):
    status: str  # "ok" | "degraded" | "error"
    version: str
    checks: List[DoctorCheck]


class TranslateSegmentIO(BaseModel):
    index: int
    text: str


class TranslateRequest(BaseModel):
    segments: List[TranslateSegmentIO]
    target_language: str
    source_language: Optional[str] = None


class TranslateResponse(BaseModel):
    translations: List[TranslateSegmentIO]
    target_language: str


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Check server health and model status.
    """
    models = {
        "scribe_v2": "configured" if settings.elevenlabs_api_key else "missing_api_key",
    }

    return HealthResponse(
        status="ok",
        version=__version__,
        models=models,
        gpu=None,
    )


def _run_doctor_checks() -> List[DoctorCheck]:
    """Inspect external tools and Scribe runtime configuration.

    Critical checks failing -> status "error"; only non-critical failing ->
    "degraded". `codex` is non-critical because transcription works without LLM
    refinement or translation.
    """
    import shutil
    import subprocess as _sp

    checks: List[DoctorCheck] = []

    # codex CLI — LLM refinement (and target-language translation)
    codex_path = shutil.which("codex")
    if codex_path:
        ver = None
        try:
            r = _sp.run(["codex", "--version"], capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                ver = (r.stdout or r.stderr).strip().splitlines()[0]
        except Exception:
            ver = None
        checks.append(DoctorCheck(name="codex", ok=True, critical=False,
                                  detail=f"{codex_path} ({ver})" if ver else codex_path))
    else:
        checks.append(DoctorCheck(
            name="codex",
            ok=False,
            critical=False,
            detail="codex CLI not found on PATH - LLM refinement/translation disabled",
        ))

    # ffmpeg — audio/video decoding
    ff = shutil.which("ffmpeg")
    checks.append(DoctorCheck(name="ffmpeg", ok=bool(ff), critical=True,
                              detail=ff or "ffmpeg not found on PATH"))

    checks.append(DoctorCheck(
        name="elevenlabs_api_key",
        ok=bool(settings.elevenlabs_api_key),
        critical=True,
        detail="configured" if settings.elevenlabs_api_key else "ELEVENLABS_API_KEY is not set",
    ))
    checks.append(DoctorCheck(
        name="scribe_model",
        ok=bool(settings.scribe_model_id),
        critical=True,
        detail=settings.scribe_model_id,
    ))
    checks.append(DoctorCheck(
        name="scribe_cache",
        ok=True,
        critical=False,
        detail=settings.scribe_cache_dir,
    ))

    return checks


@app.get("/doctor", response_model=DoctorResponse)
async def doctor():
    """
    Diagnose the runtime setup: external tools (codex, ffmpeg), Scribe config,
    and cache configuration. Surfaces environment issues that /health does not
    (e.g. a missing codex CLI that silently disables LLM refinement/translation).
    """
    checks = _run_doctor_checks()
    has_critical_failure = any(c.critical and not c.ok for c in checks)
    has_optional_failure = any((not c.critical) and not c.ok for c in checks)
    status = "error" if has_critical_failure else ("degraded" if has_optional_failure else "ok")
    return DoctorResponse(status=status, version=__version__, checks=checks)


@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    """
    Translate segments to a target language (length-aware, for dubbing) via Codex CLI.

    Requires the codex CLI to be available (see /doctor). Returns one translation
    per input segment, preserving indices.
    """
    from chalna.translator import translate_segments

    try:
        out = translate_segments(
            [s.model_dump() for s in req.segments],
            source_language=req.source_language,
            target_language=req.target_language,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"translation failed: {e}")
    return TranslateResponse(
        translations=[TranslateSegmentIO(**t) for t in out],
        target_language=req.target_language,
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
    use_alignment: bool = Form(True, description="Deprecated; ignored"),
    use_llm_segmentation: bool = Form(
        True,
        description="Plan Scribe word-to-segment boundaries with LLM",
    ),
    bypass_llm_segmentation_cache: bool = Form(
        False,
        description="Bypass cached LLM segment plans and recompute segmentation",
    ),
    use_llm_refinement: bool = Form(True, description="Refine Scribe output with LLM"),
    segmentation_boundary_rule: str = Form(
        DEFAULT_BOUNDARY_RULE,
        description="Post-segmentation timestamp boundary rule",
    ),
    diarize: bool = Form(True, description="Enable Scribe speaker diarization"),
    tag_audio_events: bool = Form(True, description="Tag non-speech audio events"),
    num_speakers: Optional[int] = Form(None, description="Expected speaker count (1-32)"),
    output_format: Literal["srt", "json"] = Form("srt", description="Output format"),
    include_logs: bool = Form(False, description="Include refinement logs in JSON output"),
    include_intermediate: bool = Form(False, description="Include intermediate stage results"),
    overlap_intervals: Optional[UploadFile] = File(
        None,
        description="Optional overlap protection intervals JSON",
    ),
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
    _make_scribe_options(
        diarize=diarize,
        tag_audio_events=tag_audio_events,
        num_speakers=num_speakers,
    )
    segmentation_boundary_rule = _validate_segmentation_boundary_rule(
        segmentation_boundary_rule
    )
    overlap_protection = await _read_overlap_intervals_upload(overlap_intervals)

    # Save uploaded file
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    # Validate audio file upfront (fail fast before queuing)
    try:
        audio_info = validate_audio_file(tmp_path)
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
        audio_duration=audio_info.duration_seconds,
        use_alignment=False,
        use_llm_segmentation=use_llm_segmentation,
        bypass_llm_segmentation_cache=bypass_llm_segmentation_cache,
        use_llm_refinement=use_llm_refinement,
    )
    _jobs[job_id] = job

    _job_params[job_id] = dict(
        audio_path=tmp_path,
        context=context,
        language=language,
        include_speaker=include_speaker,
        use_alignment=use_alignment,
        use_llm_segmentation=use_llm_segmentation,
        bypass_llm_segmentation_cache=bypass_llm_segmentation_cache,
        use_llm_refinement=use_llm_refinement,
        segmentation_boundary_rule=segmentation_boundary_rule,
        diarize=diarize,
        tag_audio_events=tag_audio_events,
        num_speakers=num_speakers,
        output_format=output_format,
        include_logs=include_logs,
        include_intermediate=include_intermediate,
        overlap_protection=overlap_protection,
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
    use_alignment: bool = Form(True, description="Deprecated; ignored"),
    use_llm_segmentation: bool = Form(
        True,
        description="Plan Scribe word-to-segment boundaries with LLM",
    ),
    bypass_llm_segmentation_cache: bool = Form(
        False,
        description="Bypass cached LLM segment plans and recompute segmentation",
    ),
    use_llm_refinement: bool = Form(True, description="Refine Scribe output with LLM"),
    segmentation_boundary_rule: str = Form(
        DEFAULT_BOUNDARY_RULE,
        description="Post-segmentation timestamp boundary rule",
    ),
    diarize: bool = Form(True, description="Enable Scribe speaker diarization"),
    tag_audio_events: bool = Form(True, description="Tag non-speech audio events"),
    num_speakers: Optional[int] = Form(None, description="Expected speaker count (1-32)"),
    output_format: Literal["srt", "json"] = Form("srt", description="Output format"),
    include_logs: bool = Form(False, description="Include refinement logs in result"),
    include_intermediate: bool = Form(False, description="Include intermediate stage results"),
    overlap_intervals: Optional[UploadFile] = File(
        None,
        description="Optional overlap protection intervals JSON",
    ),
):
    """
    Transcribe audio/video file asynchronously (for long files).

    Returns job_id immediately. Poll /jobs/{job_id} for status.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    _make_scribe_options(
        diarize=diarize,
        tag_audio_events=tag_audio_events,
        num_speakers=num_speakers,
    )
    segmentation_boundary_rule = _validate_segmentation_boundary_rule(
        segmentation_boundary_rule
    )
    overlap_protection = await _read_overlap_intervals_upload(overlap_intervals)

    # Save uploaded file
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    # Validate audio file upfront (fail fast)
    try:
        audio_info = validate_audio_file(tmp_path)
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
        audio_duration=audio_info.duration_seconds,
        use_alignment=False,
        use_llm_segmentation=use_llm_segmentation,
        bypass_llm_segmentation_cache=bypass_llm_segmentation_cache,
        use_llm_refinement=use_llm_refinement,
    )
    _jobs[job_id] = job

    # Enqueue for processing
    _job_params[job_id] = dict(
        audio_path=tmp_path,
        context=context,
        language=language,
        include_speaker=include_speaker,
        use_alignment=use_alignment,
        use_llm_segmentation=use_llm_segmentation,
        bypass_llm_segmentation_cache=bypass_llm_segmentation_cache,
        use_llm_refinement=use_llm_refinement,
        segmentation_boundary_rule=segmentation_boundary_rule,
        diarize=diarize,
        tag_audio_events=tag_audio_events,
        num_speakers=num_speakers,
        output_format=output_format,
        include_logs=include_logs,
        include_intermediate=include_intermediate,
        overlap_protection=overlap_protection,
    )
    await _job_queue.put(job_id)

    # Compute ETA
    eta = _compute_eta(job_id)

    return TranscribeResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        estimated_wait_seconds=eta.get("wait_seconds"),
        estimated_processing_seconds=eta.get("processing_seconds"),
        estimated_completion=eta.get("completion"),
    )


@app.post("/transcribe/from-scribe/async", response_model=TranscribeResponse)
async def transcribe_from_scribe_async(
    file: UploadFile = File(..., description="Audio or video file"),
    scribe_response: UploadFile = File(..., description="Raw Scribe V2 JSON response"),
    context: Optional[str] = Form(None, description="Context for segmentation/refinement only"),
    language: Optional[str] = Form(None, description="Language hint (ko, en, ja, zh)"),
    include_speaker: bool = Form(True, description="Include speaker labels"),
    use_alignment: bool = Form(True, description="Deprecated; ignored"),
    use_llm_segmentation: bool = Form(
        True,
        description="Plan Scribe word-to-segment boundaries with LLM",
    ),
    bypass_llm_segmentation_cache: bool = Form(
        False,
        description="Bypass cached LLM segment plans and recompute segmentation",
    ),
    use_llm_refinement: bool = Form(True, description="Refine Scribe output with LLM"),
    segmentation_boundary_rule: str = Form(
        DEFAULT_BOUNDARY_RULE,
        description="Post-segmentation timestamp boundary rule",
    ),
    diarize: bool = Form(True, description="Enable Scribe speaker diarization"),
    tag_audio_events: bool = Form(True, description="Tag non-speech audio events"),
    num_speakers: Optional[int] = Form(None, description="Expected speaker count (1-32)"),
    output_format: Literal["srt", "json"] = Form("srt", description="Output format"),
    include_logs: bool = Form(False, description="Include refinement logs in result"),
    include_intermediate: bool = Form(False, description="Include intermediate stage results"),
    overlap_intervals: Optional[UploadFile] = File(
        None,
        description="Optional overlap protection intervals JSON",
    ),
):
    """Run Chalna segmentation/refinement from an already cached raw Scribe response."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    if not scribe_response.filename:
        raise HTTPException(status_code=400, detail="No Scribe response provided")
    _make_scribe_options(
        diarize=diarize,
        tag_audio_events=tag_audio_events,
        num_speakers=num_speakers,
    )
    segmentation_boundary_rule = _validate_segmentation_boundary_rule(
        segmentation_boundary_rule
    )
    overlap_protection = await _read_overlap_intervals_upload(overlap_intervals)

    try:
        raw_payload = json.loads((await scribe_response.read()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid Scribe response JSON") from exc
    if not isinstance(raw_payload, dict):
        raise HTTPException(status_code=400, detail="Scribe response must be a JSON object")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        audio_info = validate_audio_file(tmp_path)
    except ChalnaError:
        tmp_path.unlink(missing_ok=True)
        raise

    job_id = str(uuid.uuid4())
    job = Job(
        job_id=job_id,
        status=JobStatus.QUEUED,
        created_at=datetime.utcnow(),
        output_format=output_format,
        audio_duration=audio_info.duration_seconds,
        use_alignment=False,
        use_llm_segmentation=use_llm_segmentation,
        bypass_llm_segmentation_cache=bypass_llm_segmentation_cache,
        use_llm_refinement=use_llm_refinement,
        raw_scribe_response=raw_payload,
    )
    _jobs[job_id] = job

    _job_params[job_id] = dict(
        audio_path=tmp_path,
        context=context,
        language=language,
        include_speaker=include_speaker,
        use_alignment=use_alignment,
        use_llm_segmentation=use_llm_segmentation,
        bypass_llm_segmentation_cache=bypass_llm_segmentation_cache,
        use_llm_refinement=use_llm_refinement,
        segmentation_boundary_rule=segmentation_boundary_rule,
        diarize=diarize,
        tag_audio_events=tag_audio_events,
        num_speakers=num_speakers,
        output_format=output_format,
        include_logs=include_logs,
        include_intermediate=include_intermediate,
        scribe_response_override=raw_payload,
        overlap_protection=overlap_protection,
    )
    await _job_queue.put(job_id)

    eta = _compute_eta(job_id)

    return TranscribeResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        estimated_wait_seconds=eta.get("wait_seconds"),
        estimated_processing_seconds=eta.get("processing_seconds"),
        estimated_completion=eta.get("completion"),
    )


@app.get("/jobs")
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """List completed/failed jobs from DB (history)."""
    jobs = db_list_jobs(status=status, limit=limit, offset=offset)
    total = db_count_jobs(status=status)
    return {"jobs": jobs, "total": total, "limit": limit, "offset": offset}


@app.get("/jobs/active")
async def get_active_jobs():
    """Get currently active (processing/queued) jobs for live monitoring."""
    active = []
    for job in _jobs.values():
        if job.status in (JobStatus.PROCESSING, JobStatus.QUEUED):
            eta = _compute_eta(job.job_id)
            # Extract current stage and stage timestamps from progress_history
            current_stage = None
            stage_timestamps = {}
            for entry in job.progress_history:
                stage = entry.get("stage")
                ts = entry.get("timestamp")
                if stage and ts:
                    if stage not in stage_timestamps:
                        stage_timestamps[stage] = ts
                    current_stage = stage

            active.append({
                "job_id": job.job_id,
                "status": job.status.value,
                "progress": job.progress,
                "current_stage": current_stage,
                "stage_timestamps": stage_timestamps,
                "progress_history": job.progress_history,
                "raw_srt": job.raw_srt,
                "aligned_srt": job.aligned_srt,
                "refined_srt": job.refined_srt,
                "chunks_completed": job.chunks_completed,
                "total_chunks": job.total_chunks,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "created_at": job.created_at.isoformat(),
                "audio_duration": job.audio_duration,
                "queue_position": _compute_queue_position(job.job_id),
                "estimated_wait_seconds": eta.get("wait_seconds"),
                "estimated_processing_seconds": eta.get("processing_seconds"),
                "estimated_completion": (
                    eta.get("completion").isoformat() if eta.get("completion") else None
                ),
            })
    return {"jobs": active}


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """
    Get status of an async transcription job.

    Checks in-memory jobs first (active), then falls back to DB (history).
    """
    # Active in-memory job (queued/processing/just completed)
    if job_id in _jobs:
        job = _jobs[job_id]
        eta = _compute_eta(job_id)

        # Extract current stage and stage timestamps from progress_history
        current_stage = None
        stage_timestamps = {}
        for entry in job.progress_history:
            stage = entry.get("stage")
            ts = entry.get("timestamp")
            if stage and ts:
                if stage not in stage_timestamps:
                    stage_timestamps[stage] = ts
                current_stage = stage

        return JobResponse(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            result=job.result,
            error=job.error,
            error_details=job.error_details,
            alignment_log=job.alignment_log,
            segmentation_log=job.segmentation_log,
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
            audio_duration=job.audio_duration,
            current_stage=current_stage,
            stage_timestamps=stage_timestamps,
            estimated_wait_seconds=eta.get("wait_seconds"),
            estimated_processing_seconds=eta.get("processing_seconds"),
            estimated_completion=eta.get("completion"),
        )

    # Fallback: DB historical job
    db_job = db_get_job(job_id)
    if db_job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Try to load SRT result from disk
    result_srt = None
    if db_job.get("has_result_files") and db_job.get("results_dir"):
        result_srt = _load_srt_from_results(db_job["results_dir"])

    return JobResponse(
        job_id=db_job["job_id"],
        status=db_job["status"],
        progress=1.0 if db_job["status"] == "completed" else 0.0,
        result=result_srt,
        error=db_job.get("error"),
        refined=db_job.get("refined"),
        started_at=db_job.get("started_at"),
        audio_duration=db_job.get("audio_duration"),
    )


@app.get("/jobs/{job_id}/chunks/{chunk_index}")
async def get_chunk_result(job_id: str, chunk_index: int):
    """Get raw ASR SRT result for a specific chunk, if chunked output exists."""
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
            detail=(
                f"Chunk {chunk_index} not yet available "
                f"(completed: {job.chunks_completed}/{job.total_chunks})"
            ),
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
            capture_job_exception(e, job_id=job_id, context={"stage": "queue_worker"})
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
    use_llm_segmentation: bool,
    bypass_llm_segmentation_cache: bool,
    use_llm_refinement: bool,
    segmentation_boundary_rule: str,
    diarize: bool,
    tag_audio_events: bool,
    num_speakers: Optional[int],
    output_format: str,
    include_logs: bool = False,
    include_intermediate: bool = False,
    scribe_response_override: Optional[dict] = None,
    overlap_protection: Optional[dict[str, Any] | list[Any]] = None,
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
        # Weights match actual pipeline order:
        # validating → transcribing → optional refining
        transcribing_end = 0.80 if use_llm_refinement else 0.95
        stage_weights = {
            "validating": (0.0, 0.05),
            "transcribing": (0.05, transcribing_end),
            "refining": (0.80, 0.95),
        }
        if stage in stage_weights:
            start, end = stage_weights[stage]
            job.progress = start + (end - start) * progress

        # Store intermediate results as stages complete
        if progress >= 1.0:
            try:
                from chalna.srt_utils import segments_to_srt
                p = get_pipeline()
                if stage == "transcribing" and p._raw_segments:
                    job.raw_srt = segments_to_srt(p._raw_segments, include_speaker=True)
                elif stage == "refining" and p._refined_segments:
                    job.refined_srt = segments_to_srt(p._refined_segments, include_speaker=True)
            except Exception:
                pass  # Don't fail job for monitoring side-effects

    try:
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.utcnow()
        job.progress = 0.0

        # Get pipeline
        pipeline = get_pipeline()
        pipeline.use_alignment = False
        pipeline.use_llm_segmentation = use_llm_segmentation
        pipeline.use_llm_refinement = use_llm_refinement
        scribe_options = _make_scribe_options(
            diarize=diarize,
            tag_audio_events=tag_audio_events,
            num_speakers=num_speakers,
        )
        llm_segmentation_options = _make_llm_segmentation_options(
            use_llm_segmentation=use_llm_segmentation,
            bypass_llm_segmentation_cache=bypass_llm_segmentation_cache,
            segmentation_boundary_rule=segmentation_boundary_rule,
        )

        # Run transcription (in thread pool to not block)
        loop = asyncio.get_event_loop()
        if scribe_response_override is not None:
            job.raw_scribe_response = scribe_response_override
            result = await loop.run_in_executor(
                None,
                lambda: pipeline.transcribe_from_scribe_response(
                    audio_path=audio_path,
                    scribe_response=scribe_response_override,
                    context=context,
                    language=language,
                    progress_callback=progress_callback,
                    scribe_options=scribe_options,
                    llm_segmentation_options=llm_segmentation_options,
                    overlap_protection=overlap_protection,
                )
            )
        else:
            result = await loop.run_in_executor(
                None,
                lambda: pipeline.transcribe(
                    audio_path=audio_path,
                    context=context,
                    language=language,
                    progress_callback=progress_callback,
                    scribe_options=scribe_options,
                    llm_segmentation_options=llm_segmentation_options,
                    overlap_protection=overlap_protection,
                )
            )
            job.raw_scribe_response = pipeline._last_scribe_response

        job.progress = 0.95

        # Include logs if requested (use result.intermediate for thread safety)
        if include_logs and result.intermediate:
            job.alignment_log = result.intermediate.alignment_log
            job.segmentation_log = result.intermediate.segmentation_log
            job.refinement_log = result.intermediate.refinement_log

        # Include intermediate results if requested
        if include_intermediate and result.intermediate:
            from chalna.srt_utils import segments_to_srt

            if result.intermediate.raw_segments:
                job.raw_srt = segments_to_srt(
                    result.intermediate.raw_segments, include_speaker=True
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
                result_data["segmentation_log"] = job.segmentation_log
                result_data["refinement_log"] = job.refinement_log
            if include_intermediate:
                result_data["raw_srt"] = job.raw_srt
                result_data["refined_srt"] = job.refined_srt
                result_data["scribe_response"] = job.raw_scribe_response
            import json
            job.result = json.dumps(result_data, ensure_ascii=False, indent=2)
        else:
            job.result = result.to_srt(include_speaker=include_speaker)

        job.refined = result.metadata.refined
        job.status = JobStatus.COMPLETED
        job.progress = 1.0
        job.completed_at = datetime.utcnow()

        # Update performance stats for future ETA estimates
        _update_performance_stats(job)

        # Save results to files
        _save_job_results(job, result, include_speaker)
        _save_job_to_db(job)

    except ChalnaError as e:
        if e.http_status >= 500:
            capture_job_exception(
                e,
                job_id=job_id,
                context={
                    "error_code": e.error_code.value,
                    "error_type": e.__class__.__name__,
                    "http_status": e.http_status,
                    "audio_path": str(audio_path),
                    "language": language,
                    "use_alignment": use_alignment,
                    "use_llm_segmentation": use_llm_segmentation,
                    "use_llm_refinement": use_llm_refinement,
                },
            )
        job.status = JobStatus.FAILED
        job.error = e.message
        job.error_details = {**e.to_dict(), "http_status": e.http_status}
        _save_job_error(job)
        _save_job_to_db(job)

    except Exception as e:
        capture_job_exception(
            e,
            job_id=job_id,
            context={
                "error_type": e.__class__.__name__,
                "audio_path": str(audio_path),
                "language": language,
                "use_alignment": use_alignment,
                "use_llm_refinement": use_llm_refinement,
            },
        )
        job.status = JobStatus.FAILED
        job.error = str(e)
        _save_job_error(job)
        _save_job_to_db(job)

    finally:
        # Cleanup
        audio_path.unlink(missing_ok=True)


# =============================================================================
# Startup/Shutdown
# =============================================================================

def _load_srt_from_results(results_dir_name: str) -> Optional[str]:
    """Load the final SRT file from a results subdirectory."""
    job_dir = RESULTS_DIR / results_dir_name
    if not job_dir.is_dir():
        return None
    # Find .srt files that aren't intermediate (_raw, _aligned, _refined)
    for srt_file in sorted(job_dir.glob("*.srt")):
        is_intermediate = any(
            marker in srt_file.name for marker in ("_raw", "_aligned", "_refined")
        )
        if not is_intermediate:
            try:
                return srt_file.read_text(encoding="utf-8")
            except OSError:
                return None
    return None


def _save_job_to_db(job: Job) -> None:
    """Persist job metadata to SQLite."""
    try:
        db_save_job({
            "job_id": job.job_id,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "audio_duration": job.audio_duration,
            "error": job.error,
            "refined": job.refined,
            "results_dir": job.job_id[:8],
            "has_result_files": True,
        })
    except Exception as e:
        print(f"Failed to save job to DB: {e}")


def _save_job_results(job: Job, result, include_speaker: bool) -> None:
    """Save job results to files for persistence.

    Creates:
    - {job_id}.srt - Final SRT output
    - {job_id}.json - Full result with metadata
    - {job_id}_raw.srt - Raw Scribe output (if available)
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
                "segmentation_log": result.intermediate.segmentation_log,
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
                raw_srt = job.raw_srt or segments_to_srt(
                    result.intermediate.raw_segments,
                    include_speaker=True,
                )
                (job_dir / f"{base_name}_1_raw.srt").write_text(raw_srt, encoding="utf-8")

            if result.intermediate.refined_segments:
                refined_srt = job.refined_srt or segments_to_srt(
                    result.intermediate.refined_segments,
                    include_speaker=True,
                )
                (job_dir / f"{base_name}_2_refined.srt").write_text(
                    refined_srt,
                    encoding="utf-8",
                )

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
    """Initialize DB, migrate existing results, and start queue worker."""
    global _queue_worker_task

    # Initialize SQLite DB
    init_db(RESULTS_DIR)
    migrated = migrate_from_results_dir(RESULTS_DIR)
    if migrated > 0:
        print(f"Migrated {migrated} existing jobs to DB")
    total = db_count_jobs()
    print(f"Job DB initialized: {total} jobs in history")

    _queue_worker_task = asyncio.create_task(_queue_worker())
    print("Job queue worker started")
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
