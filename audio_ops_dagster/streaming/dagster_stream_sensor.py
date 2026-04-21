"""
Stream Sensor & Assets
======================
Dagster integration for continuous RTSP stream inference.

Architecture
------------
  MediaMTX API (:9997)
       │
       ▼
  stream_discovery_sensor   ← polls every 10s for active paths
       │
       ▼
  RunRequest → stream_inference_job
       │
       ▼
  register_streams_op       ← calls Inference API to add/remove streams
       │
       ▼
  Inference API (:8000)     ← StreamInferenceEngine runs continuously
       │
       ▼
  InfluxDB                  ← results written per chunk
       │
       ▼
  Grafana                   ← dashboards read InfluxDB

The sensor is the bridge between "what MediaMTX has live" and
"what the inference engine is currently consuming".

Usage
-----
Register in definitions.py:
    from audio_ops_dagster.streaming.dagster_stream_sensor import (
        stream_discovery_sensor,
        stream_inference_job,
        stream_config_asset,
    )

    defs = Definitions(
        assets=[..., stream_config_asset],
        jobs=[..., stream_inference_job],
        sensors=[..., stream_discovery_sensor],
        resources={...},
    )
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import requests
from dagster import (
    Field,
    AssetKey,
    DefaultSensorStatus,
    RunConfig,
    RunRequest,
    SensorEvaluationContext,
    asset,
    job,
    op,
    sensor,
    SkipReason,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Config                                                              #
# ------------------------------------------------------------------ #

MEDIAMTX_API = os.environ.get("MEDIAMTX_API_URL", "http://mediamtx:9997")
INFERENCE_API = os.environ.get("INFERENCE_API_URL", "http://inference:8000")

# Paths to ignore (internal / test paths we don't want to infer on by default)
IGNORED_PATHS = {"test_tone"}

# Tasks to run on stream chunks — must match loaded model names
DEFAULT_TASKS = os.environ.get("STREAM_INFERENCE_TASKS", "sentiment,nonverbal").split(",")


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _get_mediamtx_paths() -> list[dict]:
    """Query MediaMTX API for all active (publishing) paths."""
    try:
        resp = requests.get(f"{MEDIAMTX_API}/v3/paths/list", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        # Only return paths that have an active source publishing to them
        return [
            p for p in data.get("items", [])
            if p.get("ready") and p["name"] not in IGNORED_PATHS
        ]
    except Exception as exc:
        logger.warning("Cannot reach MediaMTX API: %s", exc)
        return []


def _get_inference_streams() -> dict[str, dict]:
    """Get streams currently registered in the inference engine."""
    try:
        resp = requests.get(f"{INFERENCE_API}/streams", timeout=5)
        resp.raise_for_status()
        streams = resp.json().get("streams", [])
        return {s["stream_id"]: s for s in streams}
    except Exception as exc:
        logger.warning("Cannot reach Inference API: %s", exc)
        return {}


def _register_stream(path_name: str, rtsp_url: str, chunk_ms: int = 1000) -> bool:
    try:
        resp = requests.post(
            f"{INFERENCE_API}/streams",
            json={
                "url": rtsp_url,
                "stream_id": path_name,
                "chunk_duration_ms": chunk_ms,
                "overlap_duration_ms": 0,
            },
            timeout=10,
        )
        resp.raise_for_status()
        logger.info("Registered stream: %s → %s", path_name, rtsp_url)
        return True
    except Exception as exc:
        logger.warning("Failed to register stream %s: %s", path_name, exc)
        return False


def _remove_stream(stream_id: str) -> bool:
    try:
        resp = requests.delete(f"{INFERENCE_API}/streams/{stream_id}", timeout=5)
        resp.raise_for_status()
        logger.info("Removed stream: %s", stream_id)
        return True
    except Exception as exc:
        logger.warning("Failed to remove stream %s: %s", stream_id, exc)
        return False


# ------------------------------------------------------------------ #
# Ops                                                                  #
# ------------------------------------------------------------------ #

@op(config_schema={
    "mediamtx_api": Field(str, default_value=MEDIAMTX_API, is_required=False),
    "inference_api": Field(str, default_value=INFERENCE_API, is_required=False),
    "chunk_duration_ms": Field(int, default_value=1000, is_required=False),
    "ignored_paths": Field(list, default_value=list(IGNORED_PATHS), is_required=False),
    "dry_run": Field(bool, default_value=False, is_required=False),
})
def discover_active_streams(context) -> list[dict]:
    """
    Query MediaMTX for all paths that currently have a publisher.
    Returns list of {name, rtsp_url} dicts.
    """
    cfg = context.op_config
    ignored = set(cfg.get("ignored_paths", list(IGNORED_PATHS)))

    try:
        resp = requests.get(f"{cfg['mediamtx_api']}/v3/paths/list", timeout=5)
        resp.raise_for_status()
        paths = resp.json().get("items", [])
    except Exception as exc:
        context.log.warning("MediaMTX unreachable: %s", exc)
        return []

    active = []
    for p in paths:
        name = p.get("name", "")
        if not p.get("ready"):
            continue
        if name in ignored:
            continue
        rtsp_url = f"rtsp://mediamtx:8554/{name}"
        active.append({"stream_id": name, "url": rtsp_url})
        context.log.info("Active stream: %s → %s", name, rtsp_url)

    context.log.info("Found %d active streams in MediaMTX", len(active))
    return active


@op(config_schema={
    "mediamtx_api": Field(str, default_value=MEDIAMTX_API, is_required=False),
    "inference_api": Field(str, default_value=INFERENCE_API, is_required=False),
    "chunk_duration_ms": Field(int, default_value=1000, is_required=False),
    "ignored_paths": Field(list, default_value=list(IGNORED_PATHS), is_required=False),
    "dry_run": Field(bool, default_value=False, is_required=False),
})
def sync_inference_streams(
    context,
    active_streams: list,
) -> dict:
    """
    Reconcile: register new streams, remove dead ones.
    Returns a summary of what changed.
    """
    cfg = context.op_config

    # Current state in inference engine
    try:
        resp = requests.get(f"{cfg['inference_api']}/streams", timeout=5)
        resp.raise_for_status()
        existing = {s["stream_id"]: s for s in resp.json().get("streams", [])}
    except Exception as exc:
        context.log.warning("Inference API unreachable: %s", exc)
        existing = {}

    active_ids = {s["stream_id"] for s in active_streams}
    existing_ids = set(existing.keys())

    to_add = [s for s in active_streams if s["stream_id"] not in existing_ids]
    to_remove = existing_ids - active_ids

    registered = []
    removed = []

    for stream in to_add:
        if cfg.get("dry_run", False):
            context.log.info("[DRY RUN] Would register: %s", stream["stream_id"])
            registered.append(stream["stream_id"])
            continue
        try:
            resp = requests.post(
                f"{cfg['inference_api']}/streams",
                json={
                    "url": stream["url"],
                    "stream_id": stream["stream_id"],
                    "chunk_duration_ms": cfg.get("chunk_duration_ms", 1000),
                    "overlap_duration_ms": 0,
                },
                timeout=10,
            )
            resp.raise_for_status()
            registered.append(stream["stream_id"])
            context.log.info("✓ Registered: %s", stream["stream_id"])
        except Exception as exc:
            context.log.warning("✗ Failed to register %s: %s", stream["stream_id"], exc)

    for stream_id in to_remove:
        if cfg.get("dry_run", False):
            context.log.info("[DRY RUN] Would remove: %s", stream_id)
            removed.append(stream_id)
            continue
        try:
            resp = requests.delete(
                f"{cfg['inference_api']}/streams/{stream_id}", timeout=5
            )
            resp.raise_for_status()
            removed.append(stream_id)
            context.log.info("✓ Removed: %s", stream_id)
        except Exception as exc:
            context.log.warning("✗ Failed to remove %s: %s", stream_id, exc)

    summary = {
        "active_in_mediamtx": list(active_ids),
        "registered": registered,
        "removed": removed,
        "unchanged": list(existing_ids & active_ids),
        "timestamp": time.time(),
    }
    context.log.info("Sync complete: +%d -%d =%d", len(registered), len(removed), len(summary["unchanged"]))
    return summary


@op
def log_stream_sync_to_influx(
    context,
    sync_summary: dict,
) -> None:
    """Write stream sync event to InfluxDB for monitoring."""
    try:
        from audio_ops_dagster.monitoring.influx_writer import get_writer, AudioMetric

        writer = get_writer()
        writer.write(AudioMetric(
            measurement="stream_sync",
            tags={"source": "dagster_sensor"},
            fields={
                "active_streams": len(sync_summary.get("active_in_mediamtx", [])),
                "registered": len(sync_summary.get("registered", [])),
                "removed": len(sync_summary.get("removed", [])),
                "unchanged": len(sync_summary.get("unchanged", [])),
            },
        ))
        context.log.info("Sync event written to InfluxDB")
    except Exception as exc:
        context.log.warning("Could not write to InfluxDB: %s", exc)


# ------------------------------------------------------------------ #
# Job                                                                  #
# ------------------------------------------------------------------ #

@job(name="stream_inference_sync_job", description="Sync active MediaMTX streams with the inference engine")
def stream_inference_job():
    """
    Discovers active RTSP streams from MediaMTX and registers/removes
    them in the inference engine. Runs on sensor trigger or manually.
    """
    active = discover_active_streams()
    summary = sync_inference_streams(active)
    log_stream_sync_to_influx(summary)


# ------------------------------------------------------------------ #
# Sensor                                                               #
# ------------------------------------------------------------------ #

@sensor(
    job=stream_inference_job,
    name="mediamtx_stream_discovery_sensor",
    description="Polls MediaMTX every 15s and triggers sync when streams change",
    default_status=DefaultSensorStatus.RUNNING,
    minimum_interval_seconds=15,
)
def stream_discovery_sensor(context: SensorEvaluationContext):
    """
    Polls MediaMTX API for active streams. If the set of active streams
    has changed since the last evaluation, triggers a sync job.

    Uses sensor cursor to track last known stream set — only fires a
    RunRequest when something actually changes, avoiding noise.
    """
    # Get current active streams from MediaMTX
    active_paths = _get_mediamtx_paths()
    current_ids = sorted(p["name"] for p in active_paths)
    current_key = ",".join(current_ids)

    # Compare with last known state (stored in cursor)
    last_key = context.cursor or ""

    if current_key == last_key:
        return SkipReason(f"No stream changes detected. Active: [{current_key}]")

    # Something changed — trigger a sync
    added = set(current_ids) - set(last_key.split(",") if last_key else [])
    removed = set(last_key.split(",") if last_key else []) - set(current_ids)

    context.update_cursor(current_key)

    description = []
    if added:
        description.append(f"new: {', '.join(added)}")
    if removed:
        description.append(f"removed: {', '.join(removed)}")

    logger.info("Stream change detected — %s", "; ".join(description))

    yield RunRequest(
        run_key=f"stream_sync_{int(time.time())}",
        run_config={
            "ops": {
                "discover_active_streams": {
                    "config": {
                        "mediamtx_api": MEDIAMTX_API,
                        "inference_api": INFERENCE_API,
                        "chunk_duration_ms": 1000,
                        "ignored_paths": list(IGNORED_PATHS),
                        "dry_run": False,
                    }
                },
                "sync_inference_streams": {
                    "config": {
                        "mediamtx_api": MEDIAMTX_API,
                        "inference_api": INFERENCE_API,
                        "chunk_duration_ms": 1000,
                        "ignored_paths": list(IGNORED_PATHS),
                        "dry_run": False,
                    }
                },
            }
        },
        tags={"stream_change": "; ".join(description)},
    )


# ------------------------------------------------------------------ #
# Stream config asset (materialisable from Dagster UI)                #
# ------------------------------------------------------------------ #

@asset(
    name="stream_inference_config",
    group_name="streaming",
    description="Current stream inference configuration — materialize to force a sync",
)
def stream_config_asset(context) -> dict:
    """
    Materializing this asset forces an immediate sync between MediaMTX
    and the inference engine regardless of sensor state.
    Useful for manual triggering from the Dagster UI.
    """
    active = _get_mediamtx_paths()
    current_streams = _get_inference_streams()

    result = {
        "mediamtx_active": [p["name"] for p in active],
        "inference_registered": list(current_streams.keys()),
        "timestamp": time.time(),
    }

    context.log.info("MediaMTX active: %s", result["mediamtx_active"])
    context.log.info("Inference registered: %s", result["inference_registered"])

    # Auto-register any missing
    existing_ids = set(current_streams.keys())
    for path in active:
        name = path["name"]
        if name not in existing_ids:
            rtsp_url = f"rtsp://mediamtx:8554/{name}"
            ok = _register_stream(name, rtsp_url)
            if ok:
                result.setdefault("newly_registered", []).append(name)
                context.log.info("Auto-registered: %s", name)

    return result


# ------------------------------------------------------------------ #
# Stream status asset (read-only view)                                 #
# ------------------------------------------------------------------ #

@asset(
    name="stream_inference_status",
    group_name="streaming",
    deps=[stream_config_asset],
    description="Live status of all registered inference streams",
)
def stream_status_asset(context) -> dict:
    """Read current stream status from the inference API."""
    try:
        resp = requests.get(f"{INFERENCE_API}/streams", timeout=5)
        resp.raise_for_status()
        streams = resp.json()
    except Exception as exc:
        context.log.warning("Could not reach inference API: %s", exc)
        streams = {"streams": [], "error": str(exc)}

    try:
        health_resp = requests.get(f"{INFERENCE_API}/health", timeout=5)
        health = health_resp.json()
    except Exception:
        health = {}

    status = {
        "streams": streams.get("streams", []),
        "engine": health.get("engine", {}),
        "models": health.get("models", []),
        "influxdb": health.get("influxdb", False),
        "timestamp": time.time(),
    }

    context.log.info(
        "Active streams: %d | Models: %s | InfluxDB: %s",
        len(status["streams"]),
        status["models"],
        status["influxdb"],
    )
    return status