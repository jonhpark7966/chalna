import asyncio
import hashlib
import hmac
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from chalna import server
from chalna.db import (
    commit_provider_failure_webhook,
    commit_provider_webhook,
    count_webhook_events,
    get_job_runtime,
    init_db,
    list_recoverable_job_runtimes,
    save_job_runtime,
    update_job_runtime,
)
from chalna.exceptions import ElevenLabsAPIError
from chalna.models import ScribeOptions
from chalna.scribe_client import ScribeClient, extract_webhook_transcription


def _job(job_id: str) -> server.Job:
    return server.Job(
        job_id=job_id,
        status=server.JobStatus.QUEUED,
        created_at=datetime.utcnow(),
        use_llm_segmentation=False,
        use_llm_refinement=False,
    )


def _save_runtime(tmp_path: Path, job_id: str, *, provider_state: str = "queued") -> Path:
    audio_path = tmp_path / "input.flac"
    audio_path.write_bytes(b"fake audio")
    job = _job(job_id)
    save_job_runtime(
        job_id=job_id,
        job_status=job.status.value,
        job_json=job.model_dump(mode="json"),
        params_json={"audio_path": str(audio_path)},
        input_path=str(audio_path),
    )
    update_job_runtime(job_id, provider_state=provider_state)
    return audio_path


def _request(body: bytes, signature: str, *, client_host: str = "127.0.0.1") -> Request:
    consumed = False

    async def receive():
        nonlocal consumed
        if consumed:
            return {"type": "http.request", "body": b"", "more_body": False}
        consumed = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "https",
            "path": "/webhooks/elevenlabs/speech-to-text",
            "raw_path": b"/webhooks/elevenlabs/speech-to-text",
            "query_string": b"",
            "headers": [(b"elevenlabs-signature", signature.encode())],
            "client": (client_host, 1234),
            "server": ("localhost", 7861),
        },
        receive,
    )


def _signature(body: bytes, secret: str) -> str:
    timestamp = str(int(time.time()))
    digest = hmac.new(
        secret.encode(),
        timestamp.encode() + b"." + body,
        hashlib.sha256,
    ).hexdigest()
    return f"t={timestamp},v0={digest}"


@pytest.fixture
def runtime_db(tmp_path: Path, monkeypatch):
    init_db(tmp_path)
    pending = tmp_path / "pending"
    pending.mkdir()
    monkeypatch.setattr(server, "PENDING_DIR", pending)
    server._jobs.clear()
    server._job_params.clear()
    while not server._job_queue.empty():
        server._job_queue.get_nowait()
        server._job_queue.task_done()
    yield tmp_path
    server._jobs.clear()
    server._job_params.clear()
    while not server._job_queue.empty():
        server._job_queue.get_nowait()
        server._job_queue.task_done()


@pytest.mark.asyncio
async def test_webhook_is_verified_durable_and_idempotent(runtime_db, monkeypatch):
    job_id = "job-webhook"
    _save_runtime(runtime_db, job_id, provider_state="awaiting_webhook")
    update_job_runtime(job_id, provider_request_id="request-1", job_status="processing")
    secret = "test-secret"
    monkeypatch.setattr(server.settings, "scribe_webhook_secret", secret)

    event = {
        "type": "speech_to_text_transcription",
        "data": {
            "request_id": "request-1",
            "transcription_id": "transcript-1",
            "webhook_metadata": {"chalna_job_id": job_id},
            "transcription": {
                "language_code": "ko",
                "text": "안녕하세요",
                "words": [
                    {
                        "type": "word",
                        "text": "안녕하세요",
                        "start": 0.0,
                        "end": 1.0,
                        "speaker_id": "speaker_0",
                    }
                ],
            },
        },
    }
    body = json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode()
    signature = _signature(body, secret)

    first = await server.receive_elevenlabs_speech_to_text_webhook(
        _request(body, signature)
    )
    second = await server.receive_elevenlabs_speech_to_text_webhook(
        _request(body, signature)
    )

    assert first["duplicate"] is False
    assert second["duplicate"] is True
    assert count_webhook_events() == 1
    runtime = get_job_runtime(job_id)
    assert runtime["provider_state"] == "completed"
    assert runtime["provider_request_id"] == "request-1"
    assert runtime["provider_transcription_id"] == "transcript-1"
    stored = json.loads(Path(runtime["provider_payload_path"]).read_text())
    assert stored["text"] == "안녕하세요"


@pytest.mark.asyncio
async def test_webhook_rejects_invalid_signature(runtime_db, monkeypatch):
    monkeypatch.setattr(server.settings, "scribe_webhook_secret", "test-secret")
    request = _request(b'{"type":"speech_to_text_transcription"}', "t=1,v0=bad")

    with pytest.raises(HTTPException) as exc_info:
        await server.receive_elevenlabs_speech_to_text_webhook(request)

    assert exc_info.value.status_code == 401
    assert count_webhook_events() == 0


@pytest.mark.asyncio
async def test_webhook_rejects_malformed_signature_timestamp(runtime_db, monkeypatch):
    monkeypatch.setattr(server.settings, "scribe_webhook_secret", "test-secret")
    request = _request(b'{"type":"speech_to_text_transcription"}', "t=oops,v0=bad")

    with pytest.raises(HTTPException) as exc_info:
        await server.receive_elevenlabs_speech_to_text_webhook(request)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_terminal_failure_webhook_is_durable_and_never_auto_reposts(
    runtime_db,
    monkeypatch,
):
    job_id = "job-terminal-failure"
    audio_path = _save_runtime(runtime_db, job_id, provider_state="awaiting_webhook")
    update_job_runtime(
        job_id,
        job_status="processing",
        provider_request_id="request-terminal-failure",
    )
    secret = "test-secret"
    monkeypatch.setattr(server.settings, "scribe_webhook_secret", secret)
    event = {
        "type": "speech_to_text_transcription",
        "data": {
            "status": "failed",
            "request_id": "request-terminal-failure",
            "webhook_metadata": {"chalna_job_id": job_id},
            "error": "unclassified provider failure",
        },
    }
    body = json.dumps(event, separators=(",", ":")).encode()

    response = await server.receive_elevenlabs_speech_to_text_webhook(
        _request(body, _signature(body, secret))
    )

    runtime = get_job_runtime(job_id)
    assert response["status"] == "failure_received"
    assert runtime["provider_state"] == "failed_retryable"
    assert runtime["failure_kind"] == "provider_terminal_unknown"
    assert runtime["retryable"] is True
    assert runtime["resubmit_safe"] is False

    def unexpected_post(*args, **kwargs):
        raise AssertionError("terminal provider failure must not auto-POST")

    monkeypatch.setattr(httpx, "post", unexpected_post)
    client = ScribeClient(
        api_key="fake",
        delivery_mode="webhook",
        webhook_id="webhook-1",
    )
    with pytest.raises(ElevenLabsAPIError) as exc_info:
        client.transcribe(
            audio_path,
            language_code="ko",
            options=ScribeOptions(),
            chalna_job_id=job_id,
        )
    assert exc_info.value.details["resubmit_safe"] is False


def test_success_failure_ordering_is_monotonic(runtime_db):
    completed_job_id = "job-success-then-failure"
    _save_runtime(runtime_db, completed_job_id, provider_state="awaiting_webhook")
    update_job_runtime(completed_job_id, job_status="processing")
    success_path = runtime_db / "success.json"
    success_path.write_text('{"text":"success","words":[]}', encoding="utf-8")
    commit_provider_webhook(
        job_id=completed_job_id,
        event_key="speech_to_text:success-first",
        event_type="speech_to_text_transcription",
        provider_request_id="request-order-1",
        provider_transcription_id="transcription-order-1",
        provider_trace_id=None,
        payload_path=str(success_path),
    )
    failure_path = runtime_db / "failure-after-success.json"
    failure_path.write_text('{"status":"failed"}', encoding="utf-8")
    late_failure = commit_provider_failure_webhook(
        job_id=completed_job_id,
        event_key="speech_to_text_failure:success-first",
        event_type="speech_to_text.failed",
        provider_request_id="request-order-1",
        provider_transcription_id="transcription-order-1",
        provider_trace_id=None,
        payload_path=str(failure_path),
        provider_error="late failure",
        failure_kind="provider_terminal_unknown",
        retryable=True,
        resubmit_safe=False,
    )
    assert late_failure["ignored_due_completed"] is True
    assert late_failure["runtime"]["provider_state"] == "completed"
    assert late_failure["runtime"]["provider_payload_path"] == str(success_path)

    failed_job_id = "job-failure-then-success"
    _save_runtime(runtime_db, failed_job_id, provider_state="awaiting_webhook")
    update_job_runtime(failed_job_id, job_status="failed")
    provider_failure = runtime_db / "provider-failure.json"
    provider_failure.write_text('{"status":"failed"}', encoding="utf-8")
    commit_provider_failure_webhook(
        job_id=failed_job_id,
        event_key="speech_to_text_failure:failure-first",
        event_type="speech_to_text.failed",
        provider_request_id="request-order-2",
        provider_transcription_id="transcription-order-2",
        provider_trace_id=None,
        payload_path=str(provider_failure),
        provider_error="temporary provider failure",
        failure_kind="provider_terminal_transient",
        retryable=True,
        resubmit_safe=True,
    )
    recovered_path = runtime_db / "recovered-success.json"
    recovered_path.write_text('{"text":"recovered","words":[]}', encoding="utf-8")
    later_success = commit_provider_webhook(
        job_id=failed_job_id,
        event_key="speech_to_text:failure-first",
        event_type="speech_to_text_transcription",
        provider_request_id="request-order-2",
        provider_transcription_id="transcription-order-2",
        provider_trace_id=None,
        payload_path=str(recovered_path),
    )
    assert later_success["activated"] is True
    assert later_success["runtime"]["provider_state"] == "completed"
    assert later_success["runtime"]["job_status"] == "queued"


def test_extract_webhook_accepts_completed_and_flattened_shape():
    transcription, delivery = extract_webhook_transcription(
        {
            "type": "speech_to_text.completed",
            "data": {
                "request_id": "req",
                "webhook_metadata": '{"chalna_job_id":"job"}',
                "language_code": "en",
                "text": "hello",
                "words": [],
            },
        }
    )

    assert transcription["text"] == "hello"
    assert delivery["webhook_metadata"]["chalna_job_id"] == "job"


def test_provider_payload_atomic_write_uses_unique_temp_files(tmp_path, monkeypatch):
    payload_path = tmp_path / "provider_response.json"
    barrier = threading.Barrier(2)
    original_write_text = Path.write_text

    def synchronized_write_text(path, *args, **kwargs):
        result = original_write_text(path, *args, **kwargs)
        if path.name.endswith(".tmp"):
            barrier.wait(timeout=2)
        return result

    monkeypatch.setattr(Path, "write_text", synchronized_write_text)
    payload = {"language_code": "ko", "text": "동일 결과", "words": []}

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(server._write_provider_payload_atomic, payload_path, payload)
            for _ in range(2)
        ]
        for future in futures:
            future.result(timeout=2)

    assert json.loads(payload_path.read_text(encoding="utf-8")) == payload
    assert {path.name for path in tmp_path.iterdir()} == {"provider_response.json"}


def test_webhook_completion_before_accept_response_is_monotonic(runtime_db, monkeypatch):
    job_id = "job-webhook-before-accept"
    audio_path = _save_runtime(runtime_db, job_id, provider_state="queued")
    update_job_runtime(job_id, job_status="processing")
    provider_payload = {"language_code": "ko", "text": "먼저 완료", "words": []}
    payload_path = runtime_db / "provider_response.json"

    def webhook_then_accept(*args, **kwargs):
        server._write_provider_payload_atomic(payload_path, provider_payload)
        commit_provider_webhook(
            job_id=job_id,
            event_key="speech_to_text:transcription-early",
            event_type="speech_to_text_transcription",
            provider_request_id="request-early",
            provider_transcription_id="transcription-early",
            provider_trace_id="webhook-trace",
            payload_path=str(payload_path),
        )
        return httpx.Response(
            200,
            json={"request_id": "request-early"},
            headers={"x-trace-id": "accept-trace"},
            request=httpx.Request("POST", "https://api.elevenlabs.io/v1/speech-to-text"),
        )

    monkeypatch.setattr(httpx, "post", webhook_then_accept)
    client = ScribeClient(
        api_key="fake",
        delivery_mode="webhook",
        webhook_id="webhook-1",
    )

    client._submit_webhook_job(
        audio_path,
        language_code="ko",
        options=ScribeOptions(),
        chalna_job_id=job_id,
        initial_attempt=0,
    )

    runtime = get_job_runtime(job_id)
    assert runtime["provider_state"] == "completed"
    assert runtime["provider_payload_path"] == str(payload_path)
    assert runtime["provider_transcription_id"] == "transcription-early"
    assert runtime["provider_trace_id"] == "webhook-trace"


@pytest.mark.asyncio
async def test_atomic_late_completion_is_startup_recoverable(runtime_db):
    job_id = "job-atomic-late-completion"
    _save_runtime(runtime_db, job_id, provider_state="recovery_required")
    update_job_runtime(
        job_id,
        job_status="failed",
        failure_kind="provider_result_pending",
        retryable=True,
        resubmit_safe=False,
    )
    payload_path = runtime_db / "late-provider.json"
    payload_path.write_text('{"text":"late","words":[]}', encoding="utf-8")

    committed = commit_provider_webhook(
        job_id=job_id,
        event_key="speech_to_text:late-atomic",
        event_type="speech_to_text_transcription",
        provider_request_id="request-late-atomic",
        provider_transcription_id="late-atomic",
        provider_trace_id=None,
        payload_path=str(payload_path),
    )

    assert committed["activated"] is True
    assert committed["runtime"]["job_status"] == "queued"
    assert [item["job_id"] for item in list_recoverable_job_runtimes()] == [job_id]
    assert await server._restore_runtime_job(job_id, enqueue=False) is True
    assert server._jobs[job_id].status == server.JobStatus.QUEUED


@pytest.mark.asyncio
async def test_verified_spool_repairs_rename_before_db_commit_crash(runtime_db):
    job_id = "job-spool-before-db-commit"
    audio_path = _save_runtime(runtime_db, job_id, provider_state="recovery_required")
    update_job_runtime(
        job_id,
        job_status="failed",
        failure_kind="provider_result_pending",
        retryable=True,
        resubmit_safe=False,
    )
    provider_payload = {
        "transcription_id": "spool-transcription",
        "text": "verified spool",
        "words": [],
    }
    final_payload_path = audio_path.parent / "provider_response.json"
    server._write_provider_payload_atomic(final_payload_path, provider_payload)

    restored = await server._restore_runtime_job(job_id, enqueue=False)

    runtime = get_job_runtime(job_id)
    assert restored is True
    assert runtime["job_status"] == "queued"
    assert runtime["provider_state"] == "completed"
    assert runtime["provider_payload_path"] == str(final_payload_path)
    assert server._jobs[job_id].status == server.JobStatus.QUEUED


@pytest.mark.asyncio
async def test_verified_success_spool_recovers_prior_terminal_provider_failure(runtime_db):
    job_id = "job-spool-after-terminal-failure"
    audio_path = _save_runtime(runtime_db, job_id, provider_state="failed_retryable")
    update_job_runtime(
        job_id,
        job_status="failed",
        provider_error="temporary provider failure",
        failure_kind="provider_terminal_transient",
        retryable=True,
        resubmit_safe=True,
    )
    final_payload_path = audio_path.parent / "provider_response.json"
    server._write_provider_payload_atomic(
        final_payload_path,
        {"transcription_id": "later-success", "text": "success", "words": []},
    )

    restored = await server._restore_runtime_job(job_id, enqueue=False)

    runtime = get_job_runtime(job_id)
    assert restored is True
    assert runtime["job_status"] == "queued"
    assert runtime["provider_state"] == "completed"


@pytest.mark.asyncio
async def test_duplicate_delivery_restores_atomic_queued_activation(runtime_db, monkeypatch):
    job_id = "job-duplicate-after-commit"
    _save_runtime(runtime_db, job_id, provider_state="recovery_required")
    update_job_runtime(
        job_id,
        job_status="failed",
        failure_kind="provider_result_pending",
    )
    payload_path = runtime_db / "duplicate-committed.json"
    payload_path.write_text('{"text":"done","words":[]}', encoding="utf-8")
    commit_provider_webhook(
        job_id=job_id,
        event_key="speech_to_text:duplicate-committed",
        event_type="speech_to_text_transcription",
        provider_request_id="request-duplicate-commit",
        provider_transcription_id="duplicate-committed",
        provider_trace_id=None,
        payload_path=str(payload_path),
    )
    secret = "test-secret"
    monkeypatch.setattr(server.settings, "scribe_webhook_secret", secret)
    event = {
        "type": "speech_to_text_transcription",
        "data": {
            "request_id": "request-duplicate-commit",
            "transcription_id": "duplicate-committed",
            "webhook_metadata": {"chalna_job_id": job_id},
            "transcription": {"text": "done", "words": []},
        },
    }
    body = json.dumps(event, separators=(",", ":")).encode()

    response = await server.receive_elevenlabs_speech_to_text_webhook(
        _request(body, _signature(body, secret))
    )

    assert response["duplicate"] is True
    assert response["revived"] is True
    assert server._jobs[job_id].status == server.JobStatus.QUEUED


def test_completion_does_not_revive_downstream_failure(runtime_db):
    job_id = "job-downstream-failure"
    _save_runtime(runtime_db, job_id, provider_state="completed")
    existing_payload = runtime_db / "existing-provider.json"
    existing_payload.write_text('{"text":"done","words":[]}', encoding="utf-8")
    update_job_runtime(
        job_id,
        job_status="failed",
        provider_payload_path=str(existing_payload),
        failure_kind="application_error",
        retryable=False,
        resubmit_safe=False,
    )

    committed = commit_provider_webhook(
        job_id=job_id,
        event_key="speech_to_text:downstream-duplicate",
        event_type="speech_to_text_transcription",
        provider_request_id="request-downstream",
        provider_transcription_id="downstream-duplicate",
        provider_trace_id=None,
        payload_path=str(existing_payload),
    )

    assert committed["activated"] is False
    assert committed["runtime"]["job_status"] == "failed"
    assert committed["runtime"]["failure_kind"] == "application_error"
    assert list_recoverable_job_runtimes() == []


def test_accepted_provider_job_loads_payload_without_new_post(runtime_db, monkeypatch):
    job_id = "job-accepted"
    audio_path = _save_runtime(runtime_db, job_id, provider_state="accepted")
    payload_path = runtime_db / "provider.json"
    payload_path.write_text('{"language_code":"ko","text":"완료","words":[]}', encoding="utf-8")
    update_job_runtime(job_id, provider_payload_path=str(payload_path))

    def unexpected_post(*args, **kwargs):
        raise AssertionError("accepted provider work must never be posted again")

    monkeypatch.setattr(httpx, "post", unexpected_post)
    client = ScribeClient(
        api_key="fake",
        delivery_mode="webhook",
        webhook_id="webhook-1",
        webhook_timeout=0.1,
    )

    result = client.transcribe(
        audio_path,
        language_code="ko",
        options=ScribeOptions(),
        chalna_job_id=job_id,
    )

    assert result["text"] == "완료"


def test_lost_accept_response_is_not_blindly_reposted(runtime_db, monkeypatch):
    job_id = "job-unknown"
    audio_path = _save_runtime(runtime_db, job_id)
    calls = 0

    def disconnected(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise httpx.RemoteProtocolError("incomplete chunked read")

    monkeypatch.setattr(httpx, "post", disconnected)
    client = ScribeClient(
        api_key="fake",
        delivery_mode="webhook",
        webhook_id="webhook-1",
    )

    client._submit_webhook_job(
        audio_path,
        language_code="ko",
        options=ScribeOptions(),
        chalna_job_id=job_id,
        initial_attempt=0,
    )

    runtime = get_job_runtime(job_id)
    assert calls == 1
    assert runtime["provider_state"] == "submission_unknown"
    assert runtime["failure_kind"] == "incomplete_read"
    assert runtime["retryable"] is True
    assert runtime["resubmit_safe"] is False


@pytest.mark.asyncio
async def test_lost_accept_response_completes_from_webhook_without_duplicate_post(
    runtime_db,
    monkeypatch,
):
    job_id = "job-lost-response-integration"
    audio_path = _save_runtime(runtime_db, job_id)
    calls = 0

    def disconnected(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise httpx.RemoteProtocolError("connection ended before response body")

    monkeypatch.setattr(httpx, "post", disconnected)
    secret = "test-secret"
    monkeypatch.setattr(server.settings, "scribe_webhook_secret", secret)
    client = ScribeClient(
        api_key="fake",
        delivery_mode="webhook",
        webhook_id="webhook-1",
        webhook_timeout=2.0,
        webhook_poll_interval=0.01,
    )

    task = asyncio.create_task(
        asyncio.to_thread(
            client.transcribe,
            audio_path,
            language_code="ko",
            options=ScribeOptions(),
            chalna_job_id=job_id,
        )
    )
    for _ in range(100):
        if get_job_runtime(job_id)["provider_state"] == "submission_unknown":
            break
        await asyncio.sleep(0.01)

    event = {
        "type": "speech_to_text_transcription",
        "data": {
            "request_id": "request-delivered",
            "transcription_id": "transcript-delivered",
            "webhook_metadata": {"chalna_job_id": job_id},
            "transcription": {"language_code": "ko", "text": "회수 완료", "words": []},
        },
    }
    body = json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode()
    await server.receive_elevenlabs_speech_to_text_webhook(
        _request(body, _signature(body, secret))
    )
    result = await task

    assert calls == 1
    assert result["text"] == "회수 완료"
    assert get_job_runtime(job_id)["provider_state"] == "completed"


def test_provider_5xx_is_not_automatically_reposted(runtime_db, monkeypatch):
    job_id = "job-provider-5xx"
    audio_path = _save_runtime(runtime_db, job_id)
    calls = 0

    def provider_error(*args, **kwargs):
        nonlocal calls
        calls += 1
        return httpx.Response(
            503,
            text="temporarily unavailable",
            request=httpx.Request("POST", "https://api.elevenlabs.io/v1/speech-to-text"),
        )

    monkeypatch.setattr(httpx, "post", provider_error)
    client = ScribeClient(
        api_key="fake",
        delivery_mode="webhook",
        webhook_id="webhook-1",
    )

    with pytest.raises(ElevenLabsAPIError):
        client._submit_webhook_job(
            audio_path,
            language_code="ko",
            options=ScribeOptions(),
            chalna_job_id=job_id,
            initial_attempt=0,
        )

    runtime = get_job_runtime(job_id)
    assert calls == 1
    assert runtime["attempt_count"] == 1
    assert runtime["provider_state"] == "failed_retryable"
    assert runtime["retryable"] is True
    assert runtime["resubmit_safe"] is True

    # The worker records the provider-stage exception as a failed Chalna job.
    # A valid callback that arrives just afterward must reactivate only that
    # provider-stage failure, never an unrelated downstream failure.
    update_job_runtime(job_id, job_status="failed")
    late_payload_path = runtime_db / "success-after-5xx.json"
    late_payload_path.write_text('{"text":"late success","words":[]}', encoding="utf-8")
    late_success = commit_provider_webhook(
        job_id=job_id,
        event_key="speech_to_text:success-after-5xx",
        event_type="speech_to_text_transcription",
        provider_request_id="request-success-after-5xx",
        provider_transcription_id="success-after-5xx",
        provider_trace_id=None,
        payload_path=str(late_payload_path),
    )
    assert late_success["activated"] is True
    assert late_success["runtime"]["job_status"] == "queued"
    assert late_success["runtime"]["provider_state"] == "completed"


@pytest.mark.parametrize("status_code", [408, 429])
def test_transient_4xx_is_explicitly_retryable_without_auto_post(
    runtime_db,
    monkeypatch,
    status_code,
):
    job_id = f"job-provider-{status_code}"
    audio_path = _save_runtime(runtime_db, job_id)
    calls = 0

    def provider_error(*args, **kwargs):
        nonlocal calls
        calls += 1
        return httpx.Response(
            status_code,
            text="temporary provider response",
            request=httpx.Request("POST", "https://api.elevenlabs.io/v1/speech-to-text"),
        )

    monkeypatch.setattr(httpx, "post", provider_error)
    client = ScribeClient(
        api_key="fake",
        delivery_mode="webhook",
        webhook_id="webhook-1",
    )

    with pytest.raises(ElevenLabsAPIError) as exc_info:
        client._submit_webhook_job(
            audio_path,
            language_code="ko",
            options=ScribeOptions(),
            chalna_job_id=job_id,
            initial_attempt=0,
        )

    runtime = get_job_runtime(job_id)
    assert calls == 1
    assert exc_info.value.details["failure_kind"] == "provider_4xx_transient"
    assert runtime["provider_state"] == "failed_retryable"
    assert runtime["retryable"] is True
    assert runtime["resubmit_safe"] is True


def test_5xx_cannot_overwrite_webhook_that_completes_before_response(
    runtime_db,
    monkeypatch,
):
    job_id = "job-5xx-after-webhook"
    audio_path = _save_runtime(runtime_db, job_id)
    payload_path = runtime_db / "provider-before-5xx.json"
    payload = {"transcription_id": "transcription-before-5xx", "text": "done", "words": []}
    calls = 0

    def webhook_then_5xx(*args, **kwargs):
        nonlocal calls
        calls += 1
        server._write_provider_payload_atomic(payload_path, payload)
        commit_provider_webhook(
            job_id=job_id,
            event_key="speech_to_text:transcription-before-5xx",
            event_type="speech_to_text_transcription",
            provider_request_id="request-before-5xx",
            provider_transcription_id="transcription-before-5xx",
            provider_trace_id="webhook-before-5xx",
            payload_path=str(payload_path),
        )
        return httpx.Response(
            503,
            text="ambiguous upstream 5xx",
            request=httpx.Request("POST", "https://api.elevenlabs.io/v1/speech-to-text"),
        )

    monkeypatch.setattr(httpx, "post", webhook_then_5xx)
    client = ScribeClient(
        api_key="fake",
        delivery_mode="webhook",
        webhook_id="webhook-1",
    )

    client._submit_webhook_job(
        audio_path,
        language_code="ko",
        options=ScribeOptions(),
        chalna_job_id=job_id,
        initial_attempt=0,
    )

    runtime = get_job_runtime(job_id)
    assert calls == 1
    assert runtime["provider_state"] == "completed"
    assert runtime["provider_payload_path"] == str(payload_path)
    assert runtime["provider_transcription_id"] == "transcription-before-5xx"


@pytest.mark.parametrize("post_outcome", ["accept", "disconnect", "http_5xx"])
def test_terminal_failure_webhook_wins_before_any_post_outcome(
    runtime_db,
    monkeypatch,
    post_outcome,
):
    job_id = f"job-failure-before-{post_outcome}"
    audio_path = _save_runtime(runtime_db, job_id)
    failure_path = runtime_db / f"failure-before-{post_outcome}.json"
    failure_path.write_text('{"status":"failed"}', encoding="utf-8")

    def failure_then_post_outcome(*args, **kwargs):
        commit_provider_failure_webhook(
            job_id=job_id,
            event_key=f"speech_to_text_failure:{post_outcome}",
            event_type="speech_to_text.failed",
            provider_request_id=f"request-{post_outcome}",
            provider_transcription_id=f"transcription-{post_outcome}",
            provider_trace_id="failure-trace",
            payload_path=str(failure_path),
            provider_error="verified terminal failure",
            failure_kind="provider_terminal_transient",
            retryable=True,
            resubmit_safe=True,
        )
        if post_outcome == "disconnect":
            raise httpx.RemoteProtocolError("response connection lost")
        if post_outcome == "http_5xx":
            return httpx.Response(
                503,
                text="ambiguous 5xx",
                request=httpx.Request("POST", "https://api.elevenlabs.io/v1/speech-to-text"),
            )
        return httpx.Response(
            200,
            json={"request_id": f"request-{post_outcome}"},
            request=httpx.Request("POST", "https://api.elevenlabs.io/v1/speech-to-text"),
        )

    monkeypatch.setattr(httpx, "post", failure_then_post_outcome)
    client = ScribeClient(
        api_key="fake",
        delivery_mode="webhook",
        webhook_id="webhook-1",
    )

    client._submit_webhook_job(
        audio_path,
        language_code="ko",
        options=ScribeOptions(),
        chalna_job_id=job_id,
        initial_attempt=0,
    )

    runtime = get_job_runtime(job_id)
    assert runtime["provider_state"] == "failed_retryable"
    assert runtime["failure_kind"] == "provider_terminal_transient"
    assert runtime["provider_error"] == "verified terminal failure"
    assert runtime["provider_trace_id"] == "failure-trace"
    assert runtime["retryable"] is True
    assert runtime["resubmit_safe"] is True


def test_expired_persisted_deadline_recovers_by_transcription_id(runtime_db, monkeypatch):
    job_id = "job-deadline-get-recovery"
    _save_runtime(runtime_db, job_id, provider_state="awaiting_webhook")
    persisted_deadline = (datetime.utcnow() - timedelta(seconds=1)).isoformat()
    update_job_runtime(
        job_id,
        job_status="processing",
        provider_request_id="request-deadline",
        provider_transcription_id="transcription-deadline",
        deadline_at=persisted_deadline,
    )
    recovered = {
        "transcription_id": "transcription-deadline",
        "language_code": "ko",
        "text": "GET 회수",
        "words": [],
    }
    client = ScribeClient(
        api_key="fake",
        delivery_mode="webhook",
        webhook_id="webhook-1",
        webhook_timeout=7200,
    )
    calls = 0

    def fake_get_transcript(transcription_id):
        nonlocal calls
        calls += 1
        assert transcription_id == "transcription-deadline"
        return recovered, {
            "provider_request_id": "request-deadline",
            "provider_transcription_id": transcription_id,
            "provider_trace_id": "get-trace",
        }

    monkeypatch.setattr(client, "get_transcript", fake_get_transcript)

    result = client._wait_for_webhook(job_id)

    runtime = get_job_runtime(job_id)
    assert result == recovered
    assert calls == 1
    assert runtime["deadline_at"] == persisted_deadline
    assert runtime["provider_state"] == "completed"
    assert runtime["provider_payload_path"]
    assert json.loads(Path(runtime["provider_payload_path"]).read_text()) == recovered


def test_restart_uses_remaining_persisted_deadline_before_recovery_required(
    runtime_db,
    monkeypatch,
):
    job_id = "job-restart-deadline"
    _save_runtime(runtime_db, job_id, provider_state="awaiting_webhook")
    persisted_deadline = (datetime.utcnow() + timedelta(seconds=0.05)).isoformat()
    update_job_runtime(
        job_id,
        provider_transcription_id="missing-transcription",
        deadline_at=persisted_deadline,
    )
    client = ScribeClient(
        api_key="fake",
        delivery_mode="webhook",
        webhook_id="webhook-1",
        webhook_timeout=7200,
        webhook_poll_interval=0.01,
    )

    def missing_transcript(transcription_id):
        raise ElevenLabsAPIError(
            "not found",
            status_code=404,
            details={"failure_kind": "provider_4xx", "retryable": False},
        )

    monkeypatch.setattr(client, "get_transcript", missing_transcript)
    started = time.monotonic()

    with pytest.raises(ElevenLabsAPIError) as exc_info:
        client._wait_for_webhook(job_id)

    assert time.monotonic() - started < 1.0
    assert exc_info.value.details["failure_kind"] == "provider_result_pending"
    runtime = get_job_runtime(job_id)
    assert runtime["deadline_at"] == persisted_deadline
    assert runtime["provider_state"] == "recovery_required"
    assert runtime["resubmit_safe"] is False


def test_timeout_cas_never_overwrites_terminal_failure(runtime_db):
    job_id = "job-timeout-terminal-race"
    _save_runtime(runtime_db, job_id, provider_state="failed_retryable")
    update_job_runtime(
        job_id,
        provider_error="terminal failure won",
        failure_kind="provider_terminal_unknown",
        retryable=True,
        resubmit_safe=False,
        deadline_at=(datetime.utcnow() - timedelta(seconds=1)).isoformat(),
    )
    client = ScribeClient(
        api_key="fake",
        delivery_mode="webhook",
        webhook_id="webhook-1",
    )

    with pytest.raises(ElevenLabsAPIError) as exc_info:
        client._wait_for_webhook(job_id)

    runtime = get_job_runtime(job_id)
    assert str(exc_info.value) == "terminal failure won"
    assert runtime["provider_state"] == "failed_retryable"
    assert runtime["failure_kind"] == "provider_terminal_unknown"


@pytest.mark.asyncio
async def test_late_webhook_revives_recovery_job(runtime_db, monkeypatch):
    job_id = "job-late-webhook"
    _save_runtime(runtime_db, job_id, provider_state="recovery_required")
    update_job_runtime(
        job_id,
        provider_request_id="request-late",
        job_status="failed",
        failure_kind="provider_result_pending",
        retryable=True,
        resubmit_safe=False,
    )
    secret = "test-secret"
    monkeypatch.setattr(server.settings, "scribe_webhook_secret", secret)
    event = {
        "type": "speech_to_text_transcription",
        "data": {
            "request_id": "request-late",
            "webhook_metadata": {"chalna_job_id": job_id},
            "transcription": {"language_code": "ko", "text": "늦은 결과", "words": []},
        },
    }
    body = json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode()

    result = await server.receive_elevenlabs_speech_to_text_webhook(
        _request(body, _signature(body, secret))
    )

    assert result["revived"] is True
    assert server._jobs[job_id].status == server.JobStatus.QUEUED
    runtime = get_job_runtime(job_id)
    assert runtime["job_status"] == "queued"
    assert runtime["provider_state"] == "completed"


@pytest.mark.asyncio
async def test_restart_rehydrates_accepted_job_without_resetting_provider_state(runtime_db):
    job_id = "job-restart"
    _save_runtime(runtime_db, job_id, provider_state="awaiting_webhook")
    update_job_runtime(job_id, provider_request_id="req-restart", job_status="processing")

    restored = await server._restore_runtime_job(job_id, enqueue=False)

    assert restored is True
    assert server._jobs[job_id].status == server.JobStatus.QUEUED
    assert get_job_runtime(job_id)["provider_request_id"] == "req-restart"


@pytest.mark.asyncio
async def test_local_recovery_returns_raw_json_srt_and_metadata(runtime_db, monkeypatch):
    provider_payload = {
        "language_code": "ko",
        "audio_duration_secs": 1.2,
        "text": "안녕하세요",
        "words": [
            {
                "type": "word",
                "text": "안녕하세요",
                "start": 0.0,
                "end": 1.0,
                "speaker_id": "speaker_0",
            }
        ],
    }

    def fake_get_transcript(self, transcription_id):
        assert self.timeout == server.settings.scribe_recovery_timeout == 90.0
        assert transcription_id == "transcript-existing"
        return provider_payload, {
            "provider_transcription_id": transcription_id,
            "provider_trace_id": "trace-1",
            "provider_request_id": None,
        }

    monkeypatch.setattr(ScribeClient, "get_transcript", fake_get_transcript)
    real_to_thread = asyncio.to_thread
    offloaded = False

    async def tracked_to_thread(func, *args, **kwargs):
        nonlocal offloaded
        offloaded = True
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(server.asyncio, "to_thread", tracked_to_thread)
    request = _request(b"", "", client_host="127.0.0.1")

    response = await server.recover_provider_transcript("transcript-existing", request)

    assert response["scribe_response"] == provider_payload
    assert "안녕하세요" in response["raw_srt"]
    assert response["metadata"]["provider_trace_id"] == "trace-1"
    assert response["metadata"]["word_count"] == 1
    assert response["metadata"]["speakers"] == ["speaker_0"]
    assert offloaded is True


@pytest.mark.asyncio
async def test_local_recovery_respects_include_audio_events(runtime_db, monkeypatch):
    provider_payload = {
        "language_code": "ko",
        "text": "안녕하세요",
        "words": [
            {
                "type": "audio_event",
                "text": "웃음",
                "start": 0.0,
                "end": 0.3,
            },
            {
                "type": "word",
                "text": "안녕하세요",
                "start": 0.4,
                "end": 1.0,
                "speaker_id": "speaker_0",
            },
        ],
    }

    def fake_get_transcript(self, transcription_id):
        return provider_payload, {
            "provider_transcription_id": transcription_id,
            "provider_trace_id": None,
            "provider_request_id": None,
        }

    monkeypatch.setattr(ScribeClient, "get_transcript", fake_get_transcript)
    request = _request(b"", "", client_host="127.0.0.1")

    included = await server.recover_provider_transcript(
        "transcript-events",
        request,
        include_audio_events=True,
    )
    excluded = await server.recover_provider_transcript(
        "transcript-events",
        request,
        include_audio_events=False,
    )

    assert "웃음" in included["raw_srt"]
    assert "웃음" not in excluded["raw_srt"]
    assert "안녕하세요" in included["raw_srt"]
    assert "안녕하세요" in excluded["raw_srt"]


@pytest.mark.asyncio
async def test_provider_recovery_rejects_non_local_call(runtime_db):
    request = _request(b"", "", client_host="203.0.113.10")

    with pytest.raises(HTTPException) as exc_info:
        await server.recover_provider_transcript("transcript-existing", request)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_provider_recovery_preserves_provider_404(runtime_db, monkeypatch):
    def missing_transcript(self, transcription_id):
        raise ElevenLabsAPIError(
            "not found",
            status_code=404,
            details={
                "failure_kind": "provider_4xx",
                "retryable": False,
                "resubmit_safe": False,
            },
        )

    monkeypatch.setattr(ScribeClient, "get_transcript", missing_transcript)
    request = _request(b"", "", client_host="127.0.0.1")

    with pytest.raises(HTTPException) as exc_info:
        await server.recover_provider_transcript("missing-transcript", request)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail["retryable"] is False
    assert exc_info.value.detail["provider_transcription_id"] == "missing-transcript"


_CURRENT_RECOVERY_FIXTURE = Path("/tmp/eogum-joGEIfW4JKH14wJALU0h.json")


@pytest.mark.skipif(
    not _CURRENT_RECOVERY_FIXTURE.exists(),
    reason="one-time production recovery fixture is not present",
)
def test_current_project_raw_srt_contract_hash():
    payload = json.loads(_CURRENT_RECOVERY_FIXTURE.read_text(encoding="utf-8"))

    raw_srt = server._raw_srt_from_provider_response(payload)

    assert hashlib.sha256(raw_srt.encode()).hexdigest() == (
        "3508d961fc0feafe90ba49251db872ffd7d6d1c22552fb042af6d095dec4a6d7"
    )
