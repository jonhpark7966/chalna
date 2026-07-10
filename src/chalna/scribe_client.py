"""ElevenLabs Scribe API client with durable webhook delivery."""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

from chalna.db import (
    begin_provider_submission_if_incomplete,
    commit_provider_webhook,
    get_job_runtime,
    mark_provider_http_failure_if_incomplete,
    mark_provider_recovery_required_if_incomplete,
    mark_provider_submission_unknown_if_incomplete,
    merge_provider_acceptance,
    update_job_runtime,
)
from chalna.exceptions import ElevenLabsAPIError
from chalna.models import ScribeOptions
from chalna.scribe_cache import TIMESTAMPS_GRANULARITY
from chalna.settings import settings

logger = logging.getLogger(__name__)


def recover_verified_provider_spool(
    chalna_job_id: str,
    runtime: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Repair the rename-before-SQL-commit crash window from the verified spool."""
    if runtime.get("provider_payload_path"):
        return None
    input_path = runtime.get("input_path")
    if not input_path:
        return None
    payload_path = Path(str(input_path)).parent / "provider_response.json"
    if not payload_path.is_file():
        return None
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Invalid verified provider spool for job_id=%s", chalna_job_id)
        return None
    if not isinstance(payload, dict):
        return None
    transcription_id = payload.get("transcription_id") or runtime.get(
        "provider_transcription_id"
    )
    commit_provider_webhook(
        job_id=chalna_job_id,
        event_key=f"verified_spool_recovery:{transcription_id or chalna_job_id}",
        event_type="verified_spool_recovery",
        provider_request_id=runtime.get("provider_request_id"),
        provider_transcription_id=(str(transcription_id) if transcription_id else None),
        provider_trace_id=runtime.get("provider_trace_id"),
        payload_path=str(payload_path),
    )
    return payload


def extract_webhook_transcription(event: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the raw Scribe response and delivery metadata from supported event shapes."""
    event_type = str(event.get("type") or "")
    if event_type not in {
        "speech_to_text_transcription",
        "speech_to_text.completed",
        "speech_to_text_transcription_failed",
        "speech_to_text.failed",
    }:
        raise ValueError(f"Unsupported ElevenLabs event type: {event_type or 'missing'}")

    data = event.get("data")
    if not isinstance(data, dict):
        raise ValueError("ElevenLabs webhook data must be an object")
    transcription = data.get("transcription")
    status = str(data.get("status") or "").strip().lower()
    terminal_failure = (
        status in {"failed", "error"}
        or event_type in {"speech_to_text_transcription_failed", "speech_to_text.failed"}
    )
    if not isinstance(transcription, dict):
        # Be tolerant of providers that flatten the transcription into data.
        if isinstance(data.get("words"), list) or isinstance(data.get("transcripts"), list):
            transcription = {
                key: value
                for key, value in data.items()
                if key not in {"request_id", "webhook_metadata"}
            }
        elif terminal_failure:
            transcription = {}
        else:
            raise ValueError("ElevenLabs webhook did not contain a transcription")

    metadata = data.get("webhook_metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}

    delivery = {
        "event_type": event_type,
        "request_id": data.get("request_id") or event.get("request_id"),
        "transcription_id": (
            data.get("transcription_id")
            or transcription.get("transcription_id")
            or event.get("transcription_id")
        ),
        "webhook_metadata": metadata,
        "terminal_failure": terminal_failure,
        "provider_error": data.get("error") or data.get("message") or status,
        "provider_status_code": data.get("status_code"),
    }
    return transcription, delivery


class ScribeClient:
    """Small wrapper around ElevenLabs speech-to-text."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        model_id: Optional[str] = None,
        delivery_mode: Optional[str] = None,
        webhook_id: Optional[str] = None,
        webhook_timeout: Optional[float] = None,
        webhook_poll_interval: Optional[float] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.elevenlabs_api_key
        self.base_url = (base_url or settings.elevenlabs_base_url).rstrip("/")
        self.timeout = timeout if timeout is not None else settings.scribe_timeout
        self.model_id = model_id or settings.scribe_model_id
        self.delivery_mode = (delivery_mode or settings.scribe_delivery_mode).strip().lower()
        self.webhook_id = webhook_id if webhook_id is not None else settings.scribe_webhook_id
        self.webhook_timeout = (
            webhook_timeout
            if webhook_timeout is not None
            else settings.scribe_webhook_timeout
        )
        self.webhook_poll_interval = (
            webhook_poll_interval
            if webhook_poll_interval is not None
            else settings.scribe_webhook_poll_interval
        )

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language_code: Optional[str],
        options: ScribeOptions,
        chalna_job_id: Optional[str] = None,
    ) -> dict:
        """Transcribe an input, using webhook delivery for durable server jobs."""
        if self.delivery_mode == "webhook":
            if not chalna_job_id:
                raise ElevenLabsAPIError(
                    "Webhook delivery requires a durable Chalna job ID",
                    details={
                        "failure_kind": "configuration",
                        "retryable": False,
                        "resubmit_safe": False,
                    },
                )
            return self._transcribe_via_webhook(
                audio_path,
                language_code=language_code,
                options=options,
                chalna_job_id=chalna_job_id,
            )
        if self.delivery_mode != "sync":
            raise ElevenLabsAPIError(
                f"Unsupported Scribe delivery mode: {self.delivery_mode}",
                details={
                    "failure_kind": "configuration",
                    "retryable": False,
                    "resubmit_safe": False,
                },
            )
        return self._transcribe_sync(
            audio_path,
            language_code=language_code,
            options=options,
        )

    def get_transcript(self, transcription_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Retrieve an already-created provider transcript without starting a new job."""
        self._require_api_key()
        try:
            response = httpx.get(
                f"{self.base_url}/v1/speech-to-text/transcripts/{transcription_id}",
                headers={"xi-api-key": self.api_key},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise ElevenLabsAPIError(
                f"ElevenLabs transcript recovery failed: {exc}",
                cause=exc,
                details={
                    "failure_kind": self._http_failure_kind(exc),
                    "retryable": True,
                    "resubmit_safe": False,
                    "provider_transcription_id": transcription_id,
                },
            ) from exc
        if response.status_code >= 400:
            retryable_status = response.status_code >= 500 or response.status_code in {
                408,
                429,
            }
            raise ElevenLabsAPIError(
                f"ElevenLabs transcript recovery returned HTTP {response.status_code}: "
                f"{response.text[:1000]}",
                status_code=response.status_code,
                details={
                    "failure_kind": (
                        "provider_5xx"
                        if response.status_code >= 500
                        else (
                            "provider_4xx_transient"
                            if retryable_status
                            else "provider_4xx"
                        )
                    ),
                    "retryable": retryable_status,
                    "resubmit_safe": False,
                    "provider_transcription_id": transcription_id,
                },
            )
        payload = self._json_object(response)
        response_transcription_id = payload.get("transcription_id") or transcription_id
        trace_id = response.headers.get("x-trace-id") or response.headers.get("trace-id")
        return payload, {
            "provider_transcription_id": response_transcription_id,
            "provider_trace_id": trace_id,
            "provider_request_id": response.headers.get("request-id"),
        }

    def _transcribe_via_webhook(
        self,
        audio_path: str | Path,
        *,
        language_code: Optional[str],
        options: ScribeOptions,
        chalna_job_id: str,
    ) -> dict:
        self._require_api_key()
        if not self.webhook_id:
            raise ElevenLabsAPIError(
                "ELEVENLABS_WEBHOOK_ID is not configured",
                details={
                    "failure_kind": "configuration",
                    "retryable": False,
                    "resubmit_safe": False,
                },
            )

        runtime = get_job_runtime(chalna_job_id)
        if runtime is None:
            raise ElevenLabsAPIError(
                f"Missing durable runtime for Chalna job {chalna_job_id}",
                details={"failure_kind": "runtime_state", "retryable": False},
            )
        recovered_spool = recover_verified_provider_spool(chalna_job_id, runtime)
        if recovered_spool is not None:
            return recovered_spool
        runtime = get_job_runtime(chalna_job_id) or runtime
        payload = self._load_provider_payload(runtime)
        if payload is not None:
            return payload

        state = str(runtime.get("provider_state") or "queued")
        if state in {"failed_permanent", "failed_retryable"}:
            raise self._runtime_failure(runtime)

        # An accepted or ambiguous POST is never submitted again. Its webhook may still arrive.
        if state not in {
            "accepted",
            "awaiting_webhook",
            "submission_unknown",
            "recovery_required",
            "completed",
            "submitting",
        }:
            self._submit_webhook_job(
                Path(audio_path),
                language_code=language_code,
                options=options,
                chalna_job_id=chalna_job_id,
                initial_attempt=int(runtime.get("attempt_count") or 0),
            )

        return self._wait_for_webhook(chalna_job_id)

    def _submit_webhook_job(
        self,
        audio_path: Path,
        *,
        language_code: Optional[str],
        options: ScribeOptions,
        chalna_job_id: str,
        initial_attempt: int,
    ) -> None:
        data = self._request_data(language_code=language_code, options=options)
        data.update(
            {
                "webhook": "true",
                "webhook_id": self.webhook_id,
                "webhook_metadata": json.dumps(
                    {"chalna_job_id": chalna_job_id}, separators=(",", ":")
                ),
            }
        )
        del initial_attempt  # DB state is authoritative across process restarts.
        deadline = datetime.utcnow() + timedelta(seconds=self.webhook_timeout)
        claim = begin_provider_submission_if_incomplete(
            chalna_job_id,
            deadline_at=deadline.isoformat(),
        )
        if not claim["started"]:
            return
        try:
            with audio_path.open("rb") as source:
                response = httpx.post(
                    f"{self.base_url}/v1/speech-to-text",
                    headers={"xi-api-key": self.api_key},
                    data=data,
                    files={"file": (audio_path.name, source, "application/octet-stream")},
                    timeout=self.timeout,
                )
        except httpx.HTTPError as exc:
            # The provider may have accepted the upload before the response connection failed.
            # Wait for the correlated webhook instead of issuing a duplicate billable POST.
            mark_provider_submission_unknown_if_incomplete(
                chalna_job_id,
                provider_trace_id=self._trace_id_from_exception(exc),
                failure_kind=self._http_failure_kind(exc),
            )
            logger.warning(
                "ElevenLabs webhook submission response was lost job_id=%s kind=%s",
                chalna_job_id,
                self._http_failure_kind(exc),
            )
            return
        except OSError as exc:
            failure = mark_provider_http_failure_if_incomplete(
                chalna_job_id,
                provider_trace_id=None,
                provider_error=f"Failed to read input file: {audio_path}",
                failure_kind="input_read",
                retryable=False,
                resubmit_safe=False,
            )
            if not failure["applied"]:
                return
            raise ElevenLabsAPIError(
                f"Failed to read input file: {audio_path}", cause=exc
            ) from exc

        trace_id = response.headers.get("x-trace-id") or response.headers.get("trace-id")
        if response.status_code >= 400:
            retryable_status = response.status_code >= 500 or response.status_code in {
                408,
                429,
            }
            failure_kind = (
                "provider_5xx"
                if response.status_code >= 500
                else (
                    "provider_4xx_transient" if retryable_status else "provider_4xx"
                )
            )
            message = (
                f"ElevenLabs returned HTTP {response.status_code}: {response.text[:1000]}"
            )
            failure = mark_provider_http_failure_if_incomplete(
                chalna_job_id,
                provider_trace_id=trace_id,
                provider_error=message,
                failure_kind=failure_kind,
                retryable=retryable_status,
                resubmit_safe=retryable_status,
            )
            if not failure["applied"]:
                return
            raise ElevenLabsAPIError(
                message,
                status_code=response.status_code,
                details={
                    "failure_kind": failure_kind,
                    "retryable": retryable_status,
                    "resubmit_safe": retryable_status,
                    "provider_trace_id": trace_id,
                },
            )

        try:
            payload = self._json_object(response)
        except ElevenLabsAPIError:
            mark_provider_submission_unknown_if_incomplete(
                chalna_job_id,
                provider_trace_id=trace_id,
                failure_kind="invalid_accept_response",
            )
            return
        request_id = payload.get("request_id") or response.headers.get("request-id")
        transcription_id = payload.get("transcription_id")
        if not request_id:
            mark_provider_submission_unknown_if_incomplete(
                chalna_job_id,
                provider_trace_id=trace_id,
                failure_kind="invalid_accept_response",
            )
            return
        merge_provider_acceptance(
            chalna_job_id,
            provider_request_id=str(request_id),
            provider_transcription_id=(
                str(transcription_id) if transcription_id is not None else None
            ),
            provider_trace_id=trace_id,
        )

    def _wait_for_webhook(self, chalna_job_id: str) -> dict[str, Any]:
        runtime = get_job_runtime(chalna_job_id) or {}
        recovered_spool = recover_verified_provider_spool(chalna_job_id, runtime)
        if recovered_spool is not None:
            return recovered_spool
        deadline = self._provider_deadline(runtime)
        if runtime and not runtime.get("deadline_at"):
            update_job_runtime(chalna_job_id, deadline_at=deadline.isoformat())
        while datetime.utcnow() < deadline:
            runtime = get_job_runtime(chalna_job_id)
            if runtime is None:
                break
            payload = self._load_provider_payload(runtime)
            if payload is not None:
                return payload
            recovered_spool = recover_verified_provider_spool(chalna_job_id, runtime)
            if recovered_spool is not None:
                return recovered_spool
            if runtime.get("provider_state") in {"failed_permanent", "failed_retryable"}:
                raise self._runtime_failure(runtime)
            remaining = max(0.0, (deadline - datetime.utcnow()).total_seconds())
            time.sleep(min(remaining, max(0.05, self.webhook_poll_interval)))

        runtime = get_job_runtime(chalna_job_id) or {}
        payload = self._load_provider_payload(runtime)
        if payload is not None:
            return payload
        transcription_id = runtime.get("provider_transcription_id")
        if transcription_id:
            try:
                recovered_payload, provider_metadata = self.get_transcript(
                    str(transcription_id)
                )
            except ElevenLabsAPIError as exc:
                logger.warning(
                    "ElevenLabs transcript recovery unavailable job_id=%s transcription_id=%s "
                    "kind=%s",
                    chalna_job_id,
                    transcription_id,
                    exc.details.get("failure_kind"),
                )
            else:
                payload_path = self._write_recovered_provider_payload(
                    chalna_job_id,
                    runtime,
                    recovered_payload,
                )
                commit_provider_webhook(
                    job_id=chalna_job_id,
                    event_key=f"provider_recovery:{transcription_id}",
                    event_type="provider_transcript_recovery",
                    provider_request_id=provider_metadata.get("provider_request_id"),
                    provider_transcription_id=str(transcription_id),
                    provider_trace_id=provider_metadata.get("provider_trace_id"),
                    payload_path=str(payload_path),
                )
                return recovered_payload
        if not mark_provider_recovery_required_if_incomplete(chalna_job_id):
            runtime = get_job_runtime(chalna_job_id) or {}
            payload = self._load_provider_payload(runtime)
            if payload is not None:
                return payload
            if runtime.get("provider_state") in {"failed_permanent", "failed_retryable"}:
                raise self._runtime_failure(runtime)
        raise ElevenLabsAPIError(
            "Timed out waiting for the accepted ElevenLabs transcription webhook",
            details={
                "failure_kind": "provider_result_pending",
                "retryable": True,
                "resubmit_safe": False,
                "provider_request_id": runtime.get("provider_request_id"),
                "provider_transcription_id": runtime.get("provider_transcription_id"),
                "provider_trace_id": runtime.get("provider_trace_id"),
            },
        )

    def _provider_deadline(self, runtime: dict[str, Any]) -> datetime:
        value = runtime.get("deadline_at")
        if isinstance(value, str) and value:
            try:
                parsed = datetime.fromisoformat(value)
                if parsed.tzinfo is not None:
                    return parsed.astimezone(timezone.utc).replace(tzinfo=None)
                return parsed
            except ValueError:
                logger.warning("Invalid provider deadline_at value: %s", value)
        return datetime.utcnow() + timedelta(seconds=self.webhook_timeout)

    @staticmethod
    def _write_recovered_provider_payload(
        chalna_job_id: str,
        runtime: dict[str, Any],
        payload: dict[str, Any],
    ) -> Path:
        input_path = runtime.get("input_path")
        if not input_path:
            raise ElevenLabsAPIError(
                f"Durable input path is missing for Chalna job {chalna_job_id}",
                details={
                    "failure_kind": "runtime_state",
                    "retryable": False,
                    "resubmit_safe": False,
                },
            )
        payload_path = Path(str(input_path)).parent / "provider_response.json"
        temporary_path = payload_path.with_name(
            f".{payload_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary_path.replace(payload_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return payload_path

    def _transcribe_sync(
        self,
        audio_path: str | Path,
        *,
        language_code: Optional[str],
        options: ScribeOptions,
    ) -> dict:
        """Compatibility path with response/header/byte observability."""
        self._require_api_key()
        audio_path = Path(audio_path)
        data = self._request_data(language_code=language_code, options=options)
        expected_bytes = None
        received_bytes = 0
        status_code: Optional[int] = None
        trace_id: Optional[str] = None
        request_id: Optional[str] = None
        transcription_id: Optional[str] = None
        try:
            with audio_path.open("rb") as source, httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/v1/speech-to-text",
                    headers={"xi-api-key": self.api_key},
                    data=data,
                    files={"file": (audio_path.name, source, "application/octet-stream")},
                ) as response:
                    status_code = response.status_code
                    expected_header = response.headers.get("content-length")
                    try:
                        expected_bytes = int(expected_header) if expected_header else None
                    except ValueError:
                        expected_bytes = None
                    trace_id = response.headers.get("x-trace-id") or response.headers.get(
                        "trace-id"
                    )
                    request_id = response.headers.get("request-id")
                    transcription_id = response.headers.get("transcription-id")
                    chunks: list[bytes] = []
                    for chunk in response.iter_bytes():
                        received_bytes += len(chunk)
                        chunks.append(chunk)
                    body = b"".join(chunks)
        except httpx.HTTPError as exc:
            if transcription_id:
                try:
                    recovered, _ = self.get_transcript(transcription_id)
                    return recovered
                except ElevenLabsAPIError:
                    pass
            details = {
                "failure_kind": self._http_failure_kind(exc),
                "retryable": True,
                "resubmit_safe": False,
                "expected_response_bytes": expected_bytes,
                "received_response_bytes": received_bytes,
                "provider_request_id": request_id,
                "provider_trace_id": trace_id,
                "provider_transcription_id": transcription_id,
            }
            response = getattr(exc, "response", None)
            if response is not None:
                details["provider_trace_id"] = (
                    response.headers.get("x-trace-id") or trace_id
                )
                details["provider_request_id"] = (
                    response.headers.get("request-id") or request_id
                )
            raise ElevenLabsAPIError(
                f"ElevenLabs request failed: {exc}", cause=exc, details=details
            ) from exc
        except OSError as exc:
            raise ElevenLabsAPIError(f"Failed to read input file: {audio_path}", cause=exc) from exc

        logger.info(
            "ElevenLabs sync response status=%s request_id=%s trace_id=%s expected_bytes=%s "
            "received_bytes=%s",
            status_code,
            request_id,
            trace_id,
            expected_bytes,
            received_bytes,
        )
        if status_code >= 400:
            raise ElevenLabsAPIError(
                f"ElevenLabs returned HTTP {status_code}: {body.decode('utf-8', 'replace')[:1000]}",
                status_code=status_code,
                details={
                    "failure_kind": "provider_4xx" if status_code < 500 else "provider_5xx",
                    "retryable": status_code >= 500,
                    "resubmit_safe": status_code >= 500,
                    "provider_request_id": request_id,
                    "provider_trace_id": trace_id,
                    "expected_response_bytes": expected_bytes,
                    "received_response_bytes": received_bytes,
                },
            )
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ElevenLabsAPIError(
                "ElevenLabs returned invalid JSON",
                cause=exc,
                details={
                    "failure_kind": "invalid_response",
                    "retryable": True,
                    "resubmit_safe": False,
                    "provider_request_id": request_id,
                    "provider_trace_id": trace_id,
                },
            ) from exc
        if not isinstance(payload, dict):
            raise ElevenLabsAPIError("ElevenLabs returned an unexpected response shape")
        return payload

    def _request_data(
        self, *, language_code: Optional[str], options: ScribeOptions
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "model_id": self.model_id,
            "diarize": str(options.diarize).lower(),
            "tag_audio_events": str(options.tag_audio_events).lower(),
            "timestamps_granularity": TIMESTAMPS_GRANULARITY,
        }
        if language_code:
            data["language_code"] = language_code
        if options.num_speakers is not None:
            data["num_speakers"] = str(options.num_speakers)
        return data

    def _require_api_key(self) -> None:
        if not self.api_key:
            raise ElevenLabsAPIError("ELEVENLABS_API_KEY is not configured")

    @staticmethod
    def _json_object(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ElevenLabsAPIError("ElevenLabs returned invalid JSON", cause=exc) from exc
        if not isinstance(payload, dict):
            raise ElevenLabsAPIError("ElevenLabs returned an unexpected response shape")
        return payload

    @staticmethod
    def _load_provider_payload(runtime: dict[str, Any]) -> Optional[dict[str, Any]]:
        payload_path = runtime.get("provider_payload_path")
        if not payload_path:
            return None
        try:
            payload = json.loads(Path(str(payload_path)).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ElevenLabsAPIError(
                "Stored ElevenLabs webhook payload is unavailable or invalid",
                cause=exc,
                details={"failure_kind": "provider_payload_invalid", "retryable": True},
            ) from exc
        if not isinstance(payload, dict):
            raise ElevenLabsAPIError("Stored ElevenLabs webhook payload must be an object")
        return payload

    @staticmethod
    def _runtime_failure(runtime: dict[str, Any]) -> ElevenLabsAPIError:
        provider_error = runtime.get("provider_error")
        return ElevenLabsAPIError(
            str(provider_error or "ElevenLabs transcription cannot be recovered automatically"),
            details={
                "failure_kind": runtime.get("failure_kind") or "provider_failure",
                "retryable": bool(runtime.get("retryable")),
                "resubmit_safe": bool(runtime.get("resubmit_safe")),
                "provider_request_id": runtime.get("provider_request_id"),
                "provider_transcription_id": runtime.get("provider_transcription_id"),
                "provider_trace_id": runtime.get("provider_trace_id"),
                "provider_error": provider_error,
            },
        )

    @staticmethod
    def _http_failure_kind(exc: httpx.HTTPError) -> str:
        if isinstance(exc, httpx.RemoteProtocolError):
            return "incomplete_read"
        if isinstance(exc, httpx.TimeoutException):
            return "timeout"
        if isinstance(exc, httpx.ConnectError):
            return "connection_error"
        return "connection_ended"

    @staticmethod
    def _trace_id_from_exception(exc: httpx.HTTPError) -> Optional[str]:
        response = getattr(exc, "response", None)
        if response is None:
            return None
        return response.headers.get("x-trace-id") or response.headers.get("trace-id")
