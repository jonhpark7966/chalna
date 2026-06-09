# Chalna API Spec

## Overview

| Item | Value |
|------|-------|
| Title | Chalna (찰나) |
| Description | SRT subtitle generation service with ElevenLabs Scribe v2 and optional LLM segmentation/refinement |
| Version | `0.1.0` |
| Default Port | `7861` |
| Base URL | `http://localhost:7861` |

## Pipeline

```text
validating
 -> scribe_v2 transcribe
 -> optional LLM segment plan
 -> optional LLM refine
 -> timestamp overlap cleanup only
 -> SRT/JSON 저장
```

Qwen forced aligner is not executed. `use_alignment` remains accepted for backward compatibility but is ignored. LLM segmentation plans word index ranges only; timestamps always come from Scribe word timestamps.

## Common Error Response

```json
{
  "error": true,
  "error_code": "E1001",
  "error_type": "AudioTooLongError",
  "message": "Audio duration exceeds...",
  "details": {}
}
```

## Endpoints

### `GET /`

Serves the built-in Web UI.

### `GET /health`

Response:

```json
{
  "status": "ok",
  "version": "0.1.0",
  "models": {
    "scribe_v2": "configured"
  },
  "gpu": null
}
```

`models.scribe_v2` is `configured` when `ELEVENLABS_API_KEY` is set and `missing_api_key` otherwise.

### `POST /unload`

Compatibility endpoint. Scribe does not load local GPU models, so this is a no-op.

### `POST /transcribe`

Synchronous transcription. Uploads a file and blocks until the result is ready.

Multipart form parameters:

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `file` | file | - | Yes | Audio or video file |
| `context` | string | null | No | Context used by LLM refinement |
| `language` | string | null | No | Scribe language hint |
| `include_speaker` | bool | true | No | Include speaker labels in SRT |
| `diarize` | bool | true | No | Enable Scribe speaker diarization |
| `tag_audio_events` | bool | true | No | Include Scribe audio event tags |
| `num_speakers` | int | null | No | Expected speaker count, 1-32 |
| `use_llm_segmentation` | bool | true | No | Plan Scribe word-to-segment boundaries with LLM |
| `use_llm_refinement` | bool | true | No | Refine Scribe output with LLM |
| `use_alignment` | bool | ignored | No | Deprecated; ignored |
| `output_format` | string | `srt` | No | `srt` or `json` |
| `include_logs` | bool | false | No | Include refinement logs in JSON output |
| `include_intermediate` | bool | false | No | Include raw/refined SRT fields |

SRT response:

```text
1
00:00:00,000 --> 00:00:03,500
[speaker_0] 안녕하세요.
```

JSON response:

```json
{
  "segments": [
    {
      "index": 1,
      "start_time": 0.0,
      "end_time": 3.5,
      "text": "안녕하세요.",
      "speaker_id": "speaker_0",
      "confidence": 1.0
    }
  ],
  "metadata": {
    "duration": 120.5,
    "language": "ko",
    "speakers": ["speaker_0"],
    "model_version": "scribe_v2",
    "aligned": false,
    "refined": true,
    "timestamp_source": "scribe_v2",
    "segmentation_source": "llm"
  }
}
```

When `include_intermediate=true`, JSON may include:

| Field | Type | Description |
|-------|------|-------------|
| `raw_srt` | string \| null | Raw Scribe v2 result converted to SRT |
| `refined_srt` | string \| null | LLM-refined result as SRT |
| `segmentation_log` | list \| null | LLM segmentation cache/planning/fallback log when logs are requested |

### `POST /transcribe/async`

Asynchronous transcription. Same form parameters as `POST /transcribe`.

Response:

```json
{
  "job_id": "uuid",
  "status": "queued",
  "estimated_wait_seconds": 0.0,
  "estimated_processing_seconds": 60.0,
  "estimated_completion": "2026-06-09T10:00:00"
}
```

### `GET /jobs/{job_id}`

Returns active or historical job status.

Important fields:

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `queued`, `processing`, `completed`, `failed` |
| `progress` | float | 0.0-1.0 |
| `result` | string \| null | Final SRT for completed jobs |
| `raw_srt` | string \| null | Raw Scribe SRT if available |
| `refined_srt` | string \| null | Refined SRT if available |
| `progress_history` | list | Stage progress records |

Progress stages:

- `validating`
- `transcribing`
- `refining`

### `GET /jobs`

Lists completed/failed jobs from SQLite history.

### `GET /jobs/active`

Lists currently queued or processing jobs.

### `GET /jobs/{job_id}/chunks/{chunk_index}`

Compatibility endpoint for older chunk observability. Scribe v2 currently runs as a single API request, so new jobs normally do not expose chunk SRTs.

## Cache

Raw Scribe responses are stored under:

```text
results/scribe_cache/{cache_key}.json
```

The cache key is based on audio metadata and Scribe request options:

- file size, duration, format, codec, sample rate, channels
- model id, language code
- diarize, tag_audio_events, num_speakers
- timestamps granularity

LLM segmentation plans are stored under:

```text
results/segment_cache/{cache_key}.json
```

The segment cache key includes the Scribe cache key, LLM model, reasoning effort, prompt version, language code, Scribe options, and segmentation options.

## Errors

| Code | Type | HTTP | Condition |
|------|------|------|-----------|
| E1001 | `AudioTooLongError` | 400 | Duration > 36000s |
| E1002 | `UnsupportedFormatError` | 400 | Unsupported audio/video format |
| E1003 | `CorruptedFileError` | 400 | File is corrupted or unreadable |
| E1004 | `FileTooLargeError` | 400 | File size > 2GB |
| E1005 | `FilePermissionError` | 400 | Cannot read file |
| E2002 | `EmptyTranscriptionError` | 200 | No speech detected |
| E3002 | `CodexAPIError` | 503 | Codex CLI segmentation/refinement failed |
| E3003 | `CodexRateLimitError` | 429 | Codex API rate limit |
| E3005 | `ElevenLabsAPIError` | 503 | ElevenLabs Scribe request failed |
| E4001 | `DiskSpaceError` | 503 | Insufficient disk space |
| E4003 | `FFmpegNotFoundError` | 500 | ffprobe/ffmpeg unavailable |
