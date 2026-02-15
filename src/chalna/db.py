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
