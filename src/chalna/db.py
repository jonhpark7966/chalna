"""
SQLite job persistence store for Chalna.

DB file lives at RESULTS_DIR/chalna.db alongside result files.
The DB is the source of truth for job history; results/ files may be
cleaned up independently.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Will be set by init_db()
_db_path: Optional[Path] = None


def _connect() -> sqlite3.Connection:
    assert _db_path is not None, "Call init_db() first"
    conn = sqlite3.connect(str(_db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(results_dir: Path) -> None:
    """Create tables if needed. Must be called once at startup."""
    global _db_path
    _db_path = results_dir / "chalna.db"

    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id          TEXT PRIMARY KEY,
                status          TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                started_at      TEXT,
                completed_at    TEXT,
                audio_duration  REAL,
                error           TEXT,
                refined         INTEGER,
                results_dir     TEXT,
                has_result_files INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_created
            ON jobs (created_at DESC)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS job_runtime (
                job_id                      TEXT PRIMARY KEY,
                job_status                  TEXT NOT NULL,
                job_json                    TEXT NOT NULL,
                params_json                 TEXT NOT NULL,
                input_path                  TEXT,
                provider_state              TEXT NOT NULL DEFAULT 'queued',
                provider_request_id         TEXT,
                provider_transcription_id   TEXT,
                provider_trace_id           TEXT,
                provider_payload_path       TEXT,
                provider_error              TEXT,
                failure_kind                TEXT,
                retryable                   INTEGER NOT NULL DEFAULT 0,
                resubmit_safe               INTEGER NOT NULL DEFAULT 0,
                attempt_count               INTEGER NOT NULL DEFAULT 0,
                deadline_at                 TEXT,
                updated_at                  TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_job_runtime_status
            ON job_runtime (job_status, updated_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_job_runtime_provider_request
            ON job_runtime (provider_request_id)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS webhook_events (
                event_key                   TEXT PRIMARY KEY,
                event_type                  TEXT NOT NULL,
                job_id                      TEXT,
                provider_request_id         TEXT,
                provider_transcription_id   TEXT,
                payload_path                TEXT NOT NULL,
                received_at                 TEXT NOT NULL,
                processed_at                TEXT
            )
        """)
        runtime_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(job_runtime)").fetchall()
        }
        if "provider_error" not in runtime_columns:
            conn.execute("ALTER TABLE job_runtime ADD COLUMN provider_error TEXT")


def save_job(job: Dict[str, Any]) -> None:
    """INSERT OR REPLACE a job record."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO jobs
                (job_id, status, created_at, started_at, completed_at,
                 audio_duration, error, refined, results_dir, has_result_files)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job["job_id"],
                job["status"],
                job["created_at"],
                job.get("started_at"),
                job.get("completed_at"),
                job.get("audio_duration"),
                job.get("error"),
                _bool_to_int(job.get("refined")),
                job.get("results_dir"),
                1 if job.get("has_result_files") else 0,
            ),
        )


def list_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """List jobs ordered by created_at DESC."""
    with _connect() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get a single job by ID."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def count_jobs(status: Optional[str] = None) -> int:
    """Count total jobs, optionally filtered by status."""
    with _connect() as conn:
        if status:
            row = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = ?", (status,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
    return row[0]


def save_job_runtime(
    *,
    job_id: str,
    job_status: str,
    job_json: Dict[str, Any],
    params_json: Dict[str, Any],
    input_path: Optional[str],
) -> None:
    """Persist enough state to requeue a job after a process restart."""
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO job_runtime
                (job_id, job_status, job_json, params_json, input_path, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                job_status = excluded.job_status,
                job_json = excluded.job_json,
                params_json = excluded.params_json,
                input_path = excluded.input_path,
                updated_at = excluded.updated_at
            """,
            (
                job_id,
                job_status,
                json.dumps(job_json, ensure_ascii=False, default=str),
                json.dumps(params_json, ensure_ascii=False, default=str),
                input_path,
                now,
            ),
        )


def update_job_runtime(job_id: str, **fields: Any) -> None:
    """Update whitelisted runtime/provider fields atomically."""
    allowed = {
        "job_status",
        "job_json",
        "params_json",
        "input_path",
        "provider_state",
        "provider_request_id",
        "provider_transcription_id",
        "provider_trace_id",
        "provider_payload_path",
        "provider_error",
        "failure_kind",
        "retryable",
        "resubmit_safe",
        "attempt_count",
        "deadline_at",
    }
    values: Dict[str, Any] = {}
    for key, value in fields.items():
        if key not in allowed:
            raise ValueError(f"Unsupported job_runtime field: {key}")
        if key in {"job_json", "params_json"} and isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False, default=str)
        if key in {"retryable", "resubmit_safe"}:
            value = 1 if value else 0
        values[key] = value
    if not values:
        return
    values["updated_at"] = datetime.utcnow().isoformat()
    assignments = ", ".join(f"{key} = ?" for key in values)
    with _connect() as conn:
        cursor = conn.execute(
            f"UPDATE job_runtime SET {assignments} WHERE job_id = ?",  # noqa: S608
            (*values.values(), job_id),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"Unknown runtime job: {job_id}")


def merge_provider_acceptance(
    job_id: str,
    *,
    provider_request_id: Optional[str],
    provider_transcription_id: Optional[str],
    provider_trace_id: Optional[str],
) -> Dict[str, Any]:
    """Merge a POST acceptance without regressing an earlier webhook completion."""
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            """
            UPDATE job_runtime
            SET provider_state = CASE
                    WHEN provider_payload_path IS NOT NULL OR provider_state = 'completed'
                    THEN 'completed'
                    WHEN provider_state IN ('failed_retryable', 'failed_permanent')
                    THEN provider_state
                    ELSE 'awaiting_webhook'
                END,
                provider_request_id = COALESCE(provider_request_id, ?),
                provider_transcription_id = COALESCE(provider_transcription_id, ?),
                provider_trace_id = COALESCE(provider_trace_id, ?),
                failure_kind = CASE
                    WHEN provider_payload_path IS NOT NULL
                         OR provider_state = 'completed'
                         OR provider_state IN ('failed_retryable', 'failed_permanent')
                    THEN failure_kind
                    ELSE NULL
                END,
                retryable = CASE
                    WHEN provider_payload_path IS NOT NULL
                         OR provider_state = 'completed'
                         OR provider_state IN ('failed_retryable', 'failed_permanent')
                    THEN retryable
                    ELSE 0
                END,
                resubmit_safe = CASE
                    WHEN provider_state IN ('failed_retryable', 'failed_permanent')
                    THEN resubmit_safe
                    ELSE 0
                END,
                updated_at = ?
            WHERE job_id = ?
            """,
            (
                provider_request_id,
                provider_transcription_id,
                provider_trace_id,
                datetime.utcnow().isoformat(),
                job_id,
            ),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"Unknown runtime job: {job_id}")
        row = conn.execute(
            "SELECT * FROM job_runtime WHERE job_id = ?", (job_id,)
        ).fetchone()
    assert row is not None
    return _runtime_row_to_dict(row)


def begin_provider_submission_if_incomplete(
    job_id: str,
    *,
    deadline_at: str,
) -> Dict[str, Any]:
    """Claim the one allowed initial provider POST without regressing completion."""
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        previous = conn.execute(
            "SELECT * FROM job_runtime WHERE job_id = ?", (job_id,)
        ).fetchone()
        if previous is None:
            raise KeyError(f"Unknown runtime job: {job_id}")
        started = (
            previous["provider_payload_path"] is None
            and previous["provider_state"] == "queued"
            and int(previous["attempt_count"] or 0) == 0
        )
        if started:
            conn.execute(
                """
                UPDATE job_runtime
                SET provider_state = 'submitting',
                    attempt_count = 1,
                    deadline_at = ?,
                    failure_kind = NULL,
                    provider_error = NULL,
                    retryable = 0,
                    resubmit_safe = 0,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (deadline_at, datetime.utcnow().isoformat(), job_id),
            )
        current = conn.execute(
            "SELECT * FROM job_runtime WHERE job_id = ?", (job_id,)
        ).fetchone()
    assert current is not None
    return {"runtime": _runtime_row_to_dict(current), "started": started}


def mark_provider_http_failure_if_incomplete(
    job_id: str,
    *,
    provider_trace_id: Optional[str],
    provider_error: str,
    failure_kind: str,
    retryable: bool,
    resubmit_safe: bool,
) -> Dict[str, Any]:
    """Persist an explicit HTTP/input failure only if completion has not won."""
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        previous = conn.execute(
            "SELECT * FROM job_runtime WHERE job_id = ?", (job_id,)
        ).fetchone()
        if previous is None:
            raise KeyError(f"Unknown runtime job: {job_id}")
        applied = (
            previous["provider_payload_path"] is None
            and previous["provider_state"]
            in {"queued", "accepted", "awaiting_webhook", "submission_unknown", "submitting"}
        )
        if applied:
            conn.execute(
                """
                UPDATE job_runtime
                SET provider_state = ?,
                    provider_trace_id = COALESCE(provider_trace_id, ?),
                    provider_error = ?,
                    failure_kind = ?,
                    retryable = ?,
                    resubmit_safe = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    "failed_retryable" if retryable else "failed_permanent",
                    provider_trace_id,
                    provider_error,
                    failure_kind,
                    1 if retryable else 0,
                    1 if resubmit_safe else 0,
                    datetime.utcnow().isoformat(),
                    job_id,
                ),
            )
        current = conn.execute(
            "SELECT * FROM job_runtime WHERE job_id = ?", (job_id,)
        ).fetchone()
    assert current is not None
    return {"runtime": _runtime_row_to_dict(current), "applied": applied}


def mark_provider_submission_unknown_if_incomplete(
    job_id: str,
    *,
    provider_trace_id: Optional[str],
    failure_kind: str,
) -> Dict[str, Any]:
    """Record an ambiguous POST only when no webhook payload has completed first."""
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            """
            UPDATE job_runtime
            SET provider_state = 'submission_unknown',
                provider_trace_id = COALESCE(provider_trace_id, ?),
                failure_kind = ?,
                retryable = 1,
                resubmit_safe = 0,
                updated_at = ?
            WHERE job_id = ?
              AND provider_payload_path IS NULL
              AND provider_state IN (
                  'queued', 'accepted', 'awaiting_webhook',
                  'submission_unknown', 'submitting'
              )
            """,
            (
                provider_trace_id,
                failure_kind,
                datetime.utcnow().isoformat(),
                job_id,
            ),
        )
        if cursor.rowcount not in {0, 1}:
            raise RuntimeError(f"Unexpected runtime update count for {job_id}")
        row = conn.execute(
            "SELECT * FROM job_runtime WHERE job_id = ?", (job_id,)
        ).fetchone()
    if row is None:
        raise KeyError(f"Unknown runtime job: {job_id}")
    return _runtime_row_to_dict(row)


def get_job_runtime(job_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM job_runtime WHERE job_id = ?", (job_id,)
        ).fetchone()
    return _runtime_row_to_dict(row) if row else None


def mark_provider_recovery_required_if_incomplete(job_id: str) -> bool:
    """Mark a wait timeout only if a webhook has not committed its payload."""
    with _connect() as conn:
        cursor = conn.execute(
            """
            UPDATE job_runtime
            SET provider_state = 'recovery_required',
                failure_kind = 'provider_result_pending',
                retryable = 1,
                resubmit_safe = 0,
                updated_at = ?
            WHERE job_id = ?
              AND provider_payload_path IS NULL
              AND provider_state IN (
                  'queued', 'accepted', 'awaiting_webhook',
                  'submission_unknown', 'submitting'
              )
            """,
            (datetime.utcnow().isoformat(), job_id),
        )
    return cursor.rowcount == 1


def find_job_runtime_by_provider_request(request_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM job_runtime WHERE provider_request_id = ?",
            (request_id,),
        ).fetchone()
    return _runtime_row_to_dict(row) if row else None


def list_recoverable_job_runtimes() -> List[Dict[str, Any]]:
    """Return non-terminal jobs that should be restored at startup."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM job_runtime
            WHERE job_status IN ('queued', 'processing')
            ORDER BY updated_at ASC
            """
        ).fetchall()
    return [_runtime_row_to_dict(row) for row in rows]


def list_failed_provider_recovery_runtimes() -> List[Dict[str, Any]]:
    """Return provider-failed jobs whose success spool may have beaten its DB commit."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM job_runtime
            WHERE job_status = 'failed'
              AND (
                    (
                        provider_state = 'recovery_required'
                        AND failure_kind = 'provider_result_pending'
                    )
                    OR (
                        provider_state IN ('failed_retryable', 'failed_permanent')
                        AND (
                            failure_kind LIKE 'provider_terminal_%'
                            OR failure_kind IN (
                                'provider_4xx', 'provider_4xx_transient', 'provider_5xx'
                            )
                        )
                    )
              )
            ORDER BY updated_at ASC
            """
        ).fetchall()
    return [_runtime_row_to_dict(row) for row in rows]


def record_webhook_event(
    *,
    event_key: str,
    event_type: str,
    job_id: Optional[str],
    provider_request_id: Optional[str],
    provider_transcription_id: Optional[str],
    payload_path: str,
) -> bool:
    """Record a delivery once. Returns False for an already-seen event."""
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO webhook_events
                (event_key, event_type, job_id, provider_request_id,
                 provider_transcription_id, payload_path, received_at, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_key,
                event_type,
                job_id,
                provider_request_id,
                provider_transcription_id,
                payload_path,
                datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat(),
            ),
        )
    return cursor.rowcount == 1


def commit_provider_webhook(
    *,
    job_id: str,
    event_key: str,
    event_type: str,
    provider_request_id: Optional[str],
    provider_transcription_id: Optional[str],
    provider_trace_id: Optional[str],
    payload_path: str,
) -> Dict[str, Any]:
    """Commit provider completion and failed-job activation in one transaction."""
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        previous = conn.execute(
            "SELECT * FROM job_runtime WHERE job_id = ?", (job_id,)
        ).fetchone()
        if previous is None:
            raise KeyError(f"Unknown runtime job: {job_id}")
        duplicate = conn.execute(
            "SELECT 1 FROM webhook_events WHERE event_key = ?", (event_key,)
        ).fetchone() is not None
        timed_out_wait = (
            previous["provider_state"] == "recovery_required"
            and previous["failure_kind"] == "provider_result_pending"
        )
        terminal_provider_failure = (
            previous["provider_state"] in {"failed_retryable", "failed_permanent"}
            and str(previous["failure_kind"] or "").startswith("provider_terminal_")
        )
        provider_http_failure = (
            previous["provider_state"] in {"failed_retryable", "failed_permanent"}
            and previous["failure_kind"]
            in {"provider_4xx", "provider_4xx_transient", "provider_5xx"}
        )
        activated = previous["job_status"] == "failed" and (
            timed_out_wait or terminal_provider_failure or provider_http_failure
        )
        preserve_downstream_failure = previous["job_status"] == "failed" and not activated
        conn.execute(
            """
            UPDATE job_runtime
            SET job_status = ?,
                provider_state = 'completed',
                provider_request_id = COALESCE(provider_request_id, ?),
                provider_transcription_id = COALESCE(provider_transcription_id, ?),
                provider_trace_id = COALESCE(provider_trace_id, ?),
                provider_payload_path = ?,
                provider_error = NULL,
                failure_kind = ?,
                retryable = ?,
                resubmit_safe = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (
                "queued" if activated else previous["job_status"],
                provider_request_id,
                provider_transcription_id,
                provider_trace_id,
                payload_path,
                previous["failure_kind"] if preserve_downstream_failure else None,
                previous["retryable"] if preserve_downstream_failure else 0,
                previous["resubmit_safe"] if preserve_downstream_failure else 0,
                now,
                job_id,
            ),
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO webhook_events
                (event_key, event_type, job_id, provider_request_id,
                 provider_transcription_id, payload_path, received_at, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_key,
                event_type,
                job_id,
                provider_request_id,
                provider_transcription_id,
                payload_path,
                now,
                now,
            ),
        )
        current = conn.execute(
            "SELECT * FROM job_runtime WHERE job_id = ?", (job_id,)
        ).fetchone()
    assert current is not None
    return {
        "runtime": _runtime_row_to_dict(current),
        "duplicate": duplicate,
        "activated": activated,
    }


def commit_provider_failure_webhook(
    *,
    job_id: str,
    event_key: str,
    event_type: str,
    provider_request_id: Optional[str],
    provider_transcription_id: Optional[str],
    provider_trace_id: Optional[str],
    payload_path: str,
    provider_error: str,
    failure_kind: str,
    retryable: bool,
    resubmit_safe: bool,
) -> Dict[str, Any]:
    """Persist a terminal provider failure unless completion already won the race."""
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        previous = conn.execute(
            "SELECT * FROM job_runtime WHERE job_id = ?", (job_id,)
        ).fetchone()
        if previous is None:
            raise KeyError(f"Unknown runtime job: {job_id}")
        duplicate = conn.execute(
            "SELECT 1 FROM webhook_events WHERE event_key = ?", (event_key,)
        ).fetchone() is not None
        ignored_due_completed = (
            previous["provider_state"] == "completed"
            or previous["provider_payload_path"] is not None
        )
        if not ignored_due_completed:
            conn.execute(
                """
                UPDATE job_runtime
                SET provider_state = ?,
                    provider_request_id = COALESCE(provider_request_id, ?),
                    provider_transcription_id = COALESCE(provider_transcription_id, ?),
                    provider_trace_id = COALESCE(provider_trace_id, ?),
                    provider_error = ?,
                    failure_kind = ?,
                    retryable = ?,
                    resubmit_safe = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    "failed_retryable" if retryable else "failed_permanent",
                    provider_request_id,
                    provider_transcription_id,
                    provider_trace_id,
                    provider_error,
                    failure_kind,
                    1 if retryable else 0,
                    1 if resubmit_safe else 0,
                    now,
                    job_id,
                ),
            )
        conn.execute(
            """
            INSERT OR IGNORE INTO webhook_events
                (event_key, event_type, job_id, provider_request_id,
                 provider_transcription_id, payload_path, received_at, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_key,
                event_type,
                job_id,
                provider_request_id,
                provider_transcription_id,
                payload_path,
                now,
                now,
            ),
        )
        current = conn.execute(
            "SELECT * FROM job_runtime WHERE job_id = ?", (job_id,)
        ).fetchone()
    assert current is not None
    return {
        "runtime": _runtime_row_to_dict(current),
        "duplicate": duplicate,
        "ignored_due_completed": ignored_due_completed,
    }


def webhook_event_exists(event_key: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM webhook_events WHERE event_key = ?", (event_key,)
        ).fetchone()
    return row is not None


def count_webhook_events() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) FROM webhook_events").fetchone()
    return int(row[0])


def migrate_from_results_dir(results_dir: Path) -> int:
    """Scan existing results/ directories and insert missing jobs into DB.

    Returns number of jobs migrated.
    """
    migrated = 0

    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir():
            continue
        # Skip non-job directories (like chalna.db file)
        dir_name = sub.name
        if len(dir_name) != 8:
            continue

        # Find JSON files (result or error)
        json_files = list(sub.glob("*.json"))
        if not json_files:
            continue

        # Pick the main JSON (prefer non-error, take first)
        main_json = None
        error_json = None
        for jf in json_files:
            if "_scribe_response" in jf.name:
                continue
            if "_error" in jf.name:
                error_json = jf
            else:
                main_json = jf

        target = main_json or error_json
        if not target:
            continue

        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        job_id = data.get("job_id")
        if not job_id:
            continue

        # Skip if already in DB
        if get_job(job_id) is not None:
            continue

        # Extract metadata
        status = data.get("status", "completed")
        created_at = data.get("created_at", "")
        completed_at = data.get("completed_at")
        error = data.get("error")

        # Audio duration from result metadata
        audio_duration = None
        result_meta = data.get("result", {})
        if isinstance(result_meta, dict):
            meta = result_meta.get("metadata", {})
            if isinstance(meta, dict):
                audio_duration = meta.get("duration")

        # Refined flag
        refined = None
        if isinstance(result_meta, dict):
            meta = result_meta.get("metadata", {})
            if isinstance(meta, dict) and "refined" in meta:
                refined = meta["refined"]

        # Check for SRT files
        has_srt = any(f.suffix == ".srt" for f in sub.iterdir())

        save_job({
            "job_id": job_id,
            "status": status,
            "created_at": created_at,
            "started_at": None,
            "completed_at": completed_at,
            "audio_duration": audio_duration,
            "error": error,
            "refined": refined,
            "results_dir": dir_name,
            "has_result_files": has_srt,
        })
        migrated += 1

    return migrated


# --- Internal helpers ---

def _bool_to_int(val: Optional[bool]) -> Optional[int]:
    if val is None:
        return None
    return 1 if val else 0


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    # Convert refined back to bool
    if d.get("refined") is not None:
        d["refined"] = bool(d["refined"])
    d["has_result_files"] = bool(d.get("has_result_files"))
    return d


def _runtime_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    result = dict(row)
    for key in ("job_json", "params_json"):
        try:
            result[key] = json.loads(result[key])
        except (TypeError, json.JSONDecodeError):
            result[key] = {}
    result["retryable"] = bool(result.get("retryable"))
    result["resubmit_safe"] = bool(result.get("resubmit_safe"))
    return result
