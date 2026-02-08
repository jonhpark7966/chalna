#!/usr/bin/env python3
"""
Chalna Benchmark: Duration-based performance + Job Queue concurrency test.

1. Extract clips of various durations from full_video.mp4
2. Send async requests with LLM refinement ON, extract per-stage timings
3. Send concurrent requests to verify FIFO queue serialization

Usage:
    python tests/benchmark_queue.py [--base-url http://localhost:7861]
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# ─── Config ──────────────────────────────────────────────────────────────────

SAMPLES_DIR = Path(__file__).parent / "samples"
FULL_VIDEO = SAMPLES_DIR / "full_video.mp4"

# Durations to benchmark (seconds)
DURATIONS = [120, 300, 600, 1800]  # 2min, 5min, 10min, 30min

# Concurrent queue test
QUEUE_TEST_DURATION = 120  # 2min clips
QUEUE_TEST_COUNT = 3


@dataclass
class BenchmarkResult:
    label: str
    duration_sec: float
    total_time: float = 0.0
    status: str = ""
    error: Optional[str] = None
    refined: Optional[bool] = None
    job_id: Optional[str] = None
    queue_position_seen: list = field(default_factory=list)
    # Per-stage timings (seconds)
    t_validate: float = 0.0
    t_transcribe: float = 0.0
    t_align: float = 0.0
    t_refine: float = 0.0
    t_overhead: float = 0.0  # model load, formatting, etc.
    segments_count: int = 0
    chunks_total: int = 0
    progress_history: list = field(default_factory=list)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def extract_clip(source: Path, start: float, duration: float, out_path: Path):
    """Extract a clip using ffmpeg fast seek."""
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(source),
        "-t", str(duration),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        out_path,
    ], capture_output=True, check=True)


def get_duration(path: Path) -> float:
    """Get audio/video duration via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def _parse_stage_timings(progress_history: List[Dict[str, Any]]) -> Dict[str, float]:
    """Extract per-stage durations from progress_history timestamps.

    Each entry has {"stage": ..., "progress": ..., "timestamp": "ISO8601"}.
    We compute stage duration = last timestamp in stage - first timestamp in stage.
    """
    from datetime import datetime as dt

    stage_ranges: Dict[str, List[str]] = {}  # stage -> [timestamps]
    for entry in progress_history:
        stage = entry.get("stage", "")
        ts = entry.get("timestamp", "")
        if stage and ts:
            stage_ranges.setdefault(stage, []).append(ts)

    def _parse_ts(s: str) -> float:
        """Parse ISO timestamp to epoch seconds."""
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                return dt.strptime(s, fmt).timestamp()
            except ValueError:
                continue
        return 0.0

    result = {}
    ordered_stages = []
    for entry in progress_history:
        s = entry.get("stage", "")
        if s and s not in ordered_stages:
            ordered_stages.append(s)

    for stage in ordered_stages:
        timestamps = stage_ranges.get(stage, [])
        if len(timestamps) >= 2:
            t_first = _parse_ts(timestamps[0])
            t_last = _parse_ts(timestamps[-1])
            result[stage] = t_last - t_first
        elif len(timestamps) == 1:
            result[stage] = 0.0

    # Compute inter-stage gaps (overhead)
    all_ts = []
    for tss in stage_ranges.values():
        for ts in tss:
            all_ts.append(_parse_ts(ts))
    if all_ts:
        total_span = max(all_ts) - min(all_ts)
        stage_sum = sum(result.values())
        result["_overhead"] = max(0, total_span - stage_sum)

    return result


def send_async_request(
    base_url: str,
    audio_path: Path,
    label: str,
    use_llm_refinement: bool = True,
    poll_interval: float = 3.0,
) -> BenchmarkResult:
    """Send async transcription request and poll until complete."""
    result = BenchmarkResult(
        label=label,
        duration_sec=get_duration(audio_path),
    )

    t0 = time.monotonic()
    try:
        # Submit
        with open(audio_path, "rb") as f:
            resp = requests.post(
                f"{base_url}/transcribe/async",
                files={"file": (audio_path.name, f, "video/mp4")},
                data={
                    "output_format": "json",
                    "use_llm_refinement": "true" if use_llm_refinement else "false",
                    "include_logs": "true",
                },
                timeout=120,
            )

        if resp.status_code != 200:
            result.total_time = time.monotonic() - t0
            result.status = f"submit_http_{resp.status_code}"
            result.error = resp.text[:300]
            print(f"  [{label}] SUBMIT FAILED ({resp.status_code})")
            return result

        job_id = resp.json()["job_id"]
        result.job_id = job_id
        print(f"  [{label}] Submitted job {job_id[:8]}", flush=True)

        # Poll
        last_status = ""
        while True:
            time.sleep(poll_interval)
            poll = requests.get(f"{base_url}/jobs/{job_id}", timeout=30)
            if poll.status_code != 200:
                continue

            data = poll.json()
            status = data["status"]
            qpos = data.get("queue_position")
            progress = data.get("progress", 0)

            if qpos is not None:
                result.queue_position_seen.append(qpos)

            if status == "completed":
                result.total_time = time.monotonic() - t0
                result.status = "ok"
                result.refined = data.get("refined")
                result.chunks_total = data.get("total_chunks", 0)
                result.progress_history = data.get("progress_history", [])

                # Parse per-stage timings
                timings = _parse_stage_timings(result.progress_history)
                result.t_validate = timings.get("validating", 0)
                result.t_transcribe = timings.get("transcribing", 0)
                result.t_align = timings.get("aligning", 0)
                result.t_refine = timings.get("refining", 0)
                result.t_overhead = timings.get("_overhead", 0)
                # loading_models is part of overhead
                result.t_overhead += timings.get("loading_models", 0)

                # Count segments from result JSON
                if data.get("result"):
                    try:
                        res_data = json.loads(data["result"])
                        result.segments_count = len(res_data.get("segments", []))
                    except (json.JSONDecodeError, TypeError):
                        pass

                print(f"  [{label}] COMPLETED in {result.total_time:.1f}s "
                      f"(ASR={result.t_transcribe:.1f}s Align={result.t_align:.1f}s "
                      f"Refine={result.t_refine:.1f}s refined={result.refined})",
                      flush=True)
                break

            elif status == "failed":
                result.total_time = time.monotonic() - t0
                result.status = "failed"
                result.error = data.get("error", "unknown")
                result.progress_history = data.get("progress_history", [])
                print(f"  [{label}] FAILED in {result.total_time:.1f}s: {result.error[:100]}",
                      flush=True)
                break

            else:
                elapsed = time.monotonic() - t0
                status_str = f"{status} {progress:.0%} qpos={qpos}"
                if status_str != last_status:
                    print(f"  [{label}] {status_str} ({elapsed:.0f}s)", flush=True)
                    last_status = status_str

    except Exception as e:
        result.total_time = time.monotonic() - t0
        result.status = "error"
        result.error = str(e)
        print(f"  [{label}] ERROR: {e}", flush=True)

    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chalna benchmark")
    parser.add_argument("--base-url", default="http://localhost:7861")
    args = parser.parse_args()

    base_url = args.base_url

    # Verify server
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        print(f"Server OK: {r.json()}")
    except Exception as e:
        print(f"Server not reachable at {base_url}: {e}")
        sys.exit(1)

    if not FULL_VIDEO.exists():
        print(f"Source video not found: {FULL_VIDEO}")
        sys.exit(1)

    source_duration = get_duration(FULL_VIDEO)
    print(f"Source: {FULL_VIDEO.name} ({source_duration:.0f}s / {source_duration/60:.1f}min)")

    # ── Phase 1: Extract clips ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 1: Extracting clips")
    print("=" * 70)

    clips = {}  # duration_sec -> Path
    tmpdir = Path(tempfile.mkdtemp(prefix="chalna_bench_"))
    print(f"Temp dir: {tmpdir}")

    for dur in DURATIONS:
        if dur > source_duration:
            print(f"  Skipping {dur}s (source only {source_duration:.0f}s)")
            continue

        label = f"{dur // 60}min"
        clip_path = tmpdir / f"clip_{label}.mp4"
        print(f"  Extracting {label} clip...", end=" ", flush=True)
        extract_clip(FULL_VIDEO, start=0, duration=dur, out_path=clip_path)
        actual = get_duration(clip_path)
        print(f"OK ({actual:.1f}s, {clip_path.stat().st_size / 1024 / 1024:.1f}MB)")
        clips[dur] = clip_path

    # Also add full video
    clips[int(source_duration)] = FULL_VIDEO

    # ── Phase 2: Sequential benchmark (LLM refinement ON) ───────────────
    print("\n" + "=" * 70)
    print("Phase 2: Sequential benchmark (async endpoint, LLM refinement ON)")
    print("=" * 70)

    sequential_results: List[BenchmarkResult] = []
    for dur in sorted(clips.keys()):
        clip = clips[dur]
        label = f"{dur // 60}min" if dur >= 60 else f"{dur}s"
        actual_dur = get_duration(clip)
        print(f"\n[{label}] Submitting ({actual_dur:.0f}s audio, LLM refine=ON)...",
              flush=True)
        result = send_async_request(
            base_url, clip, label,
            use_llm_refinement=True,
            poll_interval=5.0,
        )
        sequential_results.append(result)

    # ── Phase 3: Concurrent queue test ───────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Phase 3: Concurrent queue test "
          f"({QUEUE_TEST_COUNT} x {QUEUE_TEST_DURATION // 60}min, async, refine=ON)")
    print("=" * 70)

    queue_clip = clips.get(QUEUE_TEST_DURATION)
    queue_results: List[BenchmarkResult] = []
    if queue_clip is None:
        print("Queue test clip not available, skipping")
    else:
        t_queue_start = time.monotonic()

        with ThreadPoolExecutor(max_workers=QUEUE_TEST_COUNT) as pool:
            futures = {}
            for i in range(QUEUE_TEST_COUNT):
                label = f"queue_{i + 1}"
                fut = pool.submit(
                    send_async_request, base_url, queue_clip, label,
                    True, 3.0,
                )
                futures[fut] = label
                time.sleep(0.5)  # slight stagger

            for fut in as_completed(futures):
                queue_results.append(fut.result())

        t_queue_total = time.monotonic() - t_queue_start
        print(f"\n  Total wall time for {QUEUE_TEST_COUNT} concurrent jobs: "
              f"{t_queue_total:.1f}s", flush=True)

    # ── Report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    hdr = (f"{'Duration':<10} {'Audio':<8} {'Total':<8} {'ASR':<8} "
           f"{'Align':<8} {'Refine':<8} {'Other':<8} {'RTF':<7} "
           f"{'Segs':<6} {'Chunks':<7} {'Refined':<8} {'Status'}")
    print(f"\n## Sequential (LLM refinement ON)")
    print(hdr)
    print("-" * 110)
    for r in sequential_results:
        rtf = r.total_time / r.duration_sec if r.duration_sec > 0 else 0
        other = r.total_time - r.t_transcribe - r.t_align - r.t_refine
        print(
            f"{r.label:<10} {r.duration_sec:<8.0f} {r.total_time:<8.1f} "
            f"{r.t_transcribe:<8.1f} {r.t_align:<8.1f} {r.t_refine:<8.1f} "
            f"{other:<8.1f} {rtf:<7.3f} {r.segments_count:<6} "
            f"{r.chunks_total:<7} {str(r.refined):<8} {r.status}"
        )

    if queue_results:
        print(f"\n## Concurrent Queue Test "
              f"({QUEUE_TEST_COUNT} x {QUEUE_TEST_DURATION // 60}min, refine=ON)")
        print(f"{'Label':<10} {'Total':<8} {'ASR':<8} {'Align':<8} "
              f"{'Refine':<8} {'Status':<8} {'Queue Positions':<35} {'Job ID'}")
        print("-" * 105)

        queue_results.sort(key=lambda r: r.total_time)
        for r in queue_results:
            positions = str(r.queue_position_seen[:12])
            jid = r.job_id[:8] if r.job_id else "N/A"
            print(
                f"{r.label:<10} {r.total_time:<8.1f} {r.t_transcribe:<8.1f} "
                f"{r.t_align:<8.1f} {r.t_refine:<8.1f} {r.status:<8} "
                f"{positions:<35} {jid}"
            )

        print("\n  Queue serialization check:")
        ok_results = [r for r in queue_results if r.status == "ok"]
        if len(ok_results) == QUEUE_TEST_COUNT:
            times = sorted(r.total_time for r in ok_results)
            print(f"    Fastest single job:  {times[0]:.1f}s")
            print(f"    Slowest job:         {times[-1]:.1f}s")
            print(f"    Sum of all jobs:     {sum(times):.1f}s")
            ratio = times[-1] / (times[0] * QUEUE_TEST_COUNT)
            verdict = "SERIALIZED" if ratio > 0.7 else "PARALLEL (unexpected)"
            print(f"    -> Jobs were {verdict} (wall/expected = {ratio:.2f})")
        else:
            print("    Some jobs failed, cannot verify serialization")

    # Save report
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "server": base_url,
        "source_video": str(FULL_VIDEO),
        "source_duration_sec": source_duration,
        "sequential": [
            {
                "label": r.label,
                "duration_sec": r.duration_sec,
                "total_time_sec": r.total_time,
                "t_transcribe": r.t_transcribe,
                "t_align": r.t_align,
                "t_refine": r.t_refine,
                "t_overhead": r.t_overhead,
                "rtf": r.total_time / r.duration_sec if r.duration_sec > 0 else 0,
                "status": r.status,
                "refined": r.refined,
                "segments_count": r.segments_count,
                "chunks_total": r.chunks_total,
                "error": r.error,
                "progress_history": r.progress_history,
            }
            for r in sequential_results
        ],
        "queue_test": [
            {
                "label": r.label,
                "total_time_sec": r.total_time,
                "t_transcribe": r.t_transcribe,
                "t_align": r.t_align,
                "t_refine": r.t_refine,
                "status": r.status,
                "refined": r.refined,
                "queue_positions_seen": r.queue_position_seen,
                "job_id": r.job_id,
                "error": r.error,
            }
            for r in queue_results
        ],
    }

    report_path = Path(__file__).parent / "benchmark_results.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nFull report saved to: {report_path}")

    # Cleanup temp clips
    for p in tmpdir.iterdir():
        p.unlink()
    tmpdir.rmdir()
    print("Temp clips cleaned up.")


if __name__ == "__main__":
    main()
