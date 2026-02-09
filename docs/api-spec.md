# Chalna API Spec

## 1. Overview

| Item | Value |
|------|-------|
| Title | Chalna (찰나) |
| Description | SRT subtitle generation service with speaker diarization |
| Version | `0.1.0` |
| Default Port | `7861` |
| CORS | All origins allowed (`*`) |
| Base URL | `http://localhost:7861` |

## 2. Common Error Response

All `ChalnaError` exceptions are returned as structured JSON:

```json
{
  "error": true,
  "error_code": "E1001",
  "error_type": "AudioTooLongError",
  "message": "Audio duration (40000.0s) exceeds maximum allowed (36000s / 10 hours)",
  "details": {
    "duration_seconds": 40000.0,
    "max_duration_seconds": 36000
  }
}
```

**Schema — `ErrorResponse`**

| Field | Type | Description |
|-------|------|-------------|
| `error` | `bool` | Always `true` |
| `error_code` | `string` | Error code (e.g. `"E1001"`) |
| `error_type` | `string` | Exception class name |
| `message` | `string` | Human-readable message |
| `details` | `object \| null` | Additional context |

---

## 3. Endpoints

### 3.1 `GET /health`

Check server health and model status.

**Request**

No parameters.

**Response `200`** — `HealthResponse`

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string` | `"ok"` |
| `version` | `string` | Server version (e.g. `"0.1.0"`) |
| `models` | `object` | Model load status |
| `models.vibevoice` | `string` | `"loaded"` or `"not_loaded"` |
| `models.qwen_aligner` | `string` | `"loaded"` or `"not_loaded"` |
| `gpu` | `string \| null` | GPU device name, or `null` if unavailable |

```json
{
  "status": "ok",
  "version": "0.1.0",
  "models": {
    "vibevoice": "not_loaded",
    "qwen_aligner": "not_loaded"
  },
  "gpu": "NVIDIA GeForce RTX 5090"
}
```

---

### 3.2 `POST /unload`

Manually unload models from GPU memory.

**Request — Query Parameters**

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `force` | `bool` | `false` | No | If `true`, also unload processor (normally kept for fast reload) |

**Response `200`**

```json
{
  "status": "ok",
  "message": "Models unloaded successfully"
}
```

Possible `message` values:
- `"Models unloaded successfully"` — models were loaded and are now unloaded
- `"No models were loaded"` — pipeline existed but no models were loaded
- `"No models loaded"` — pipeline was not initialized

---

### 3.3 `POST /transcribe`

Synchronous transcription. Uploads a file and blocks until the result is ready.

**Request — `multipart/form-data`**

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `file` | `UploadFile` | — | **Yes** | Audio or video file |
| `context` | `string` | `null` | No | Context/hotwords for better accuracy |
| `language` | `string` | `null` | No | Language hint (`ko`, `en`, `ja`, `zh`) |
| `include_speaker` | `bool` | `true` | No | Include speaker labels in SRT output |
| `use_alignment` | `bool` | `true` | No | Use Qwen forced alignment |
| `use_llm_refinement` | `bool` | `true` | No | Use LLM to refine subtitles |
| `output_format` | `string` | `"srt"` | No | `"srt"` or `"json"` |
| `include_logs` | `bool` | `false` | No | Include alignment/refinement logs (JSON only) |
| `include_intermediate` | `bool` | `false` | No | Include intermediate stage results (JSON only) |

**Response `200` — SRT format** (`output_format="srt"`)

Content-Type: `text/plain; charset=utf-8`

```
1
00:00:00,000 --> 00:00:03,500
[Speaker 1] 안녕하세요.

2
00:00:03,500 --> 00:00:07,200
[Speaker 2] 네, 반갑습니다.
```

**Response `200` — JSON format** (`output_format="json"`)

Content-Type: `application/json`

```json
{
  "segments": [
    {
      "index": 1,
      "start_time": 0.0,
      "end_time": 3.5,
      "text": "안녕하세요.",
      "speaker_id": "Speaker 1",
      "confidence": 1.0
    }
  ],
  "metadata": {
    "duration": 120.5,
    "language": "ko",
    "speakers": ["Speaker 1", "Speaker 2"],
    "model_version": "vibevoice-asr",
    "aligned": true,
    "refined": true
  }
}
```

When `include_logs=true`, additional fields are added:

| Field | Type | Description |
|-------|------|-------------|
| `alignment_log` | `list[object] \| null` | Per-segment alignment details |
| `refinement_log` | `list[object] \| null` | Per-segment refinement details |

When `include_intermediate=true`, additional fields are added:

| Field | Type | Description |
|-------|------|-------------|
| `raw_srt` | `string \| null` | Stage 1: Raw VibeVoice output (SRT) |
| `aligned_srt` | `string \| null` | Stage 2: After forced alignment (SRT) |
| `refined_srt` | `string \| null` | Stage 3: After LLM refinement (SRT) |

**Errors**

| Code | Type | HTTP | Condition |
|------|------|------|-----------|
| — | `HTTPException` | 400 | No file provided (empty filename) |
| E1001 | `AudioTooLongError` | 400 | Duration > 36000s |
| E1002 | `UnsupportedFormatError` | 400 | Unsupported audio/video format |
| E1003 | `CorruptedFileError` | 400 | File is corrupted or unreadable |
| E1004 | `FileTooLargeError` | 400 | File size > 2GB |
| E1005 | `FilePermissionError` | 400 | Cannot read file |
| E2001 | `OutOfMemoryError` | 503 | GPU/CPU memory exhausted |
| E2002 | `EmptyTranscriptionError` | 200 | No speech detected |
| E4003 | `FFmpegNotFoundError` | 500 | ffmpeg not installed |
| — | `HTTPException` | 500 | Unexpected error |

---

### 3.4 `POST /transcribe/async`

Asynchronous transcription. Returns a `job_id` immediately; poll `/jobs/{job_id}` for status.

**Request — `multipart/form-data`**

Same parameters as `POST /transcribe`.

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `file` | `UploadFile` | — | **Yes** | Audio or video file |
| `context` | `string` | `null` | No | Context/hotwords for better accuracy |
| `language` | `string` | `null` | No | Language hint (`ko`, `en`, `ja`, `zh`) |
| `include_speaker` | `bool` | `true` | No | Include speaker labels in SRT output |
| `use_alignment` | `bool` | `true` | No | Use Qwen forced alignment |
| `use_llm_refinement` | `bool` | `true` | No | Use LLM to refine subtitles |
| `output_format` | `string` | `"srt"` | No | `"srt"` or `"json"` |
| `include_logs` | `bool` | `false` | No | Include alignment/refinement logs in result |
| `include_intermediate` | `bool` | `false` | No | Include intermediate stage results |

**Response `200`** — `TranscribeResponse`

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | `string` | UUID for the job |
| `status` | `string` | `"queued"` |
| `estimated_wait_seconds` | `float \| null` | Seconds until processing starts (queue wait) |
| `estimated_processing_seconds` | `float \| null` | Estimated processing time for this job |
| `estimated_completion` | `datetime \| null` | Estimated completion time (ISO 8601) |

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "queued",
  "estimated_wait_seconds": 120.5,
  "estimated_processing_seconds": 45.2,
  "estimated_completion": "2026-02-09T15:30:45"
}
```

**Errors**

Same validation errors as `POST /transcribe` (validation runs upfront before queuing).

---

### 3.5 `GET /jobs/{job_id}`

Get status of an async transcription job.

**Request — Path Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `job_id` | `string` | **Yes** | Job UUID from `/transcribe/async` |

**Response `200`** — `JobResponse`

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | `string` | Job UUID |
| `status` | `string` | `"queued"`, `"processing"`, `"completed"`, or `"failed"` |
| `progress` | `float` | 0.0 ~ 1.0 |
| `result` | `string \| null` | SRT text or JSON string (when completed) |
| `error` | `string \| null` | Error message (when failed) |
| `error_details` | `object \| null` | Structured error (when failed, ChalnaError) |
| `alignment_log` | `list[object] \| null` | Alignment log (if `include_logs` was set) |
| `refinement_log` | `list[object] \| null` | Refinement log (if `include_logs` was set) |
| `progress_history` | `list[object]` | Timeline of progress updates |
| `raw_srt` | `string \| null` | Stage 1 raw output (if `include_intermediate` was set) |
| `aligned_srt` | `string \| null` | Stage 2 aligned output (if `include_intermediate` was set) |
| `refined_srt` | `string \| null` | Stage 3 refined output (if `include_intermediate` was set) |
| `chunks_completed` | `int` | Number of chunks processed so far |
| `total_chunks` | `int` | Total number of chunks (0 if not chunked) |

**Example — Processing**

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "processing",
  "progress": 0.42,
  "result": null,
  "error": null,
  "error_details": null,
  "alignment_log": null,
  "refinement_log": null,
  "progress_history": [
    {"stage": "loading_models", "progress": 1.0, "timestamp": "2025-01-15T10:30:00"}
  ],
  "raw_srt": null,
  "aligned_srt": null,
  "refined_srt": null,
  "chunks_completed": 3,
  "total_chunks": 12,
  "refined": null,
  "queue_position": 0,
  "started_at": "2025-01-15T10:29:55",
  "estimated_wait_seconds": 0.0,
  "estimated_processing_seconds": 85.3,
  "estimated_completion": "2025-01-15T10:31:20"
}
```

**Example — Completed**

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "completed",
  "progress": 1.0,
  "result": "1\n00:00:00,000 --> 00:00:03,500\n[Speaker 1] 안녕하세요.\n\n",
  "error": null,
  "error_details": null,
  "alignment_log": null,
  "refinement_log": null,
  "progress_history": [],
  "raw_srt": null,
  "aligned_srt": null,
  "refined_srt": null,
  "chunks_completed": 12,
  "total_chunks": 12,
  "refined": true,
  "queue_position": null,
  "started_at": "2025-01-15T10:29:55",
  "estimated_wait_seconds": null,
  "estimated_processing_seconds": null,
  "estimated_completion": null
}
```

**Example — Failed**

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "failed",
  "progress": 0.25,
  "result": null,
  "error": "GPU memory exhausted during processing",
  "error_details": {
    "error": true,
    "error_code": "E2001",
    "error_type": "OutOfMemoryError",
    "message": "GPU memory exhausted during processing",
    "details": {"memory_type": "GPU"},
    "http_status": 503
  },
  "alignment_log": null,
  "refinement_log": null,
  "progress_history": [],
  "raw_srt": null,
  "aligned_srt": null,
  "refined_srt": null,
  "chunks_completed": 3,
  "total_chunks": 12,
  "refined": null,
  "queue_position": null,
  "started_at": "2025-01-15T10:29:55",
  "estimated_wait_seconds": null,
  "estimated_processing_seconds": null,
  "estimated_completion": null
}
```

**Errors**

| HTTP | Condition |
|------|-----------|
| 404 | Job not found |

---

### 3.6 `GET /jobs/{job_id}/chunks/{chunk_index}`

Get raw ASR SRT result for a specific chunk. Returns the per-chunk VibeVoice output before merging/alignment.

**Request — Path Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `job_id` | `string` | **Yes** | Job UUID |
| `chunk_index` | `int` | **Yes** | 0-based chunk index |

**Response `200`**

Content-Type: `text/plain; charset=utf-8`

Returns raw SRT text for the specified chunk.

```
1
00:00:00,000 --> 00:00:05,200
[Speaker 1] 이 부분은 첫 번째 청크입니다.

2
00:00:05,200 --> 00:00:10,000
[Speaker 1] 아직 정렬 전 결과입니다.
```

**Errors**

| HTTP | Condition |
|------|-----------|
| 404 | Job not found |
| 404 | Chunk index out of range (`0` to `total_chunks - 1`) |
| 404 | Chunk not yet available (processing in progress) |

---

## 4. Data Models

### Segment

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `index` | `int` | — | 1-based segment number |
| `start_time` | `float` | — | Start time in seconds |
| `end_time` | `float` | — | End time in seconds |
| `text` | `string` | — | Transcribed text |
| `speaker_id` | `string \| null` | `null` | Speaker label (e.g. `"Speaker 1"`) |
| `confidence` | `float` | `1.0` | Confidence score (0.0 ~ 1.0) |

### TranscriptionMetadata

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `duration` | `float` | — | Total audio duration in seconds |
| `language` | `string \| null` | `null` | Detected/specified language |
| `speakers` | `list[string]` | `[]` | List of speaker IDs |
| `model_version` | `string` | `"vibevoice-asr"` | Model identifier |
| `aligned` | `bool` | `true` | Whether Qwen alignment was applied |
| `refined` | `bool` | `true` | Whether LLM refinement was applied |

### JobStatus (enum)

| Value | Description |
|-------|-------------|
| `queued` | Job created, waiting to start |
| `processing` | Transcription in progress |
| `completed` | Transcription finished successfully |
| `failed` | Transcription failed |

### TranscribeResponse

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | `string` | UUID |
| `status` | `JobStatus` | Always `"queued"` on creation |
| `estimated_wait_seconds` | `float \| null` | Seconds until processing starts (queue wait) |
| `estimated_processing_seconds` | `float \| null` | Estimated processing time for this job |
| `estimated_completion` | `datetime \| null` | Estimated completion time (ISO 8601) |

### JobResponse

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | `string` | UUID |
| `status` | `JobStatus` | Current job status |
| `progress` | `float` | 0.0 ~ 1.0 |
| `result` | `string \| null` | Final SRT or JSON string |
| `error` | `string \| null` | Error message |
| `error_details` | `object \| null` | Structured error dict |
| `alignment_log` | `list[object] \| null` | Alignment details |
| `refinement_log` | `list[object] \| null` | Refinement details |
| `progress_history` | `list[object]` | Progress timeline |
| `raw_srt` | `string \| null` | Stage 1 output |
| `aligned_srt` | `string \| null` | Stage 2 output |
| `refined_srt` | `string \| null` | Stage 3 output |
| `chunks_completed` | `int` | Chunks processed |
| `total_chunks` | `int` | Total chunks |
| `refined` | `bool \| null` | Whether LLM refinement was applied (`null` while pending/processing) |
| `queue_position` | `int \| null` | Queue position: `null`=completed/failed, `0`=processing, `1`+=waiting |
| `started_at` | `datetime \| null` | When processing started |
| `estimated_wait_seconds` | `float \| null` | Seconds until processing starts (`null` when completed/failed) |
| `estimated_processing_seconds` | `float \| null` | Estimated remaining processing time (`null` when completed/failed) |
| `estimated_completion` | `datetime \| null` | Estimated completion time (`null` when completed/failed) |

### HealthResponse

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string` | `"ok"` |
| `version` | `string` | Server version |
| `models` | `object` | `{vibevoice: string, qwen_aligner: string}` |
| `gpu` | `string \| null` | GPU name |

---

## 5. Error Code Reference

### Validation Errors (1xxx) — HTTP 400

| Code | Error Class | Description |
|------|-------------|-------------|
| E1001 | `AudioTooLongError` | Audio duration exceeds 10 hours (36000s) |
| E1002 | `UnsupportedFormatError` | Audio/video format not supported |
| E1003 | `CorruptedFileError` | File is corrupted or unreadable |
| E1004 | `FileTooLargeError` | File size exceeds 2GB |
| E1005 | `FilePermissionError` | Cannot read file (permission denied) |

### Runtime Errors (2xxx)

| Code | Error Class | HTTP | Description |
|------|-------------|------|-------------|
| E2001 | `OutOfMemoryError` | 503 | GPU or CPU memory exhausted |
| E2002 | `EmptyTranscriptionError` | 200 | No speech detected in audio |
| E2004 | `ModelLoadError` | 500 | Model failed to load |

### Network Errors (3xxx)

| Code | Error Class | HTTP | Description |
|------|-------------|------|-------------|
| E3001 | `ModelDownloadError` | 503 | Model download failed |
| E3002 | `CodexAPIError` | 503 | Codex CLI API call failed |
| E3003 | `CodexRateLimitError` | 429 | Codex API rate limit exceeded |
| E3004 | `VibevoiceAPIError` | 503 | VibeVoice API call failed |

### System Errors (4xxx)

| Code | Error Class | HTTP | Description |
|------|-------------|------|-------------|
| E4001 | `DiskSpaceError` | 503 | Insufficient disk space |
| E4002 | `TempFileError` | 500 | Temporary file operation failed |
| E4003 | `FFmpegNotFoundError` | 500 | ffmpeg/ffprobe not installed |
