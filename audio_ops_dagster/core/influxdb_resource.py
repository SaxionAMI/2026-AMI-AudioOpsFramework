"""
InfluxDBMetricsResource — Generic time-series metrics writer for Audio Ops.

Provides a Dagster ConfigurableResource that any asset or task can use to
push time-stamped results into InfluxDB.  The design is intentionally
generic: each task defines *what* it writes (measurement name, tags, fields)
while this resource handles *how* (connection, batching, error handling).

Measurement conventions
-----------------------
  audio_inference     — per-segment inference results (sentiment labels,
                        transcription words, non-verbal events, …)
  audio_training      — epoch-level training metrics (loss, WER, accuracy)
  audio_pipeline      — pipeline-level events (job started, completed, …)
  audio_system        — system health (GPU util, latency, throughput)

All measurements carry at least these tags:
  task_name           — e.g. "sentiment", "speech_to_text", "nonverbal"
  run_id              — Dagster run id (links back to the orchestrator)
  environment         — "development" / "production"

Usage inside a Dagster asset
----------------------------
    metrics = context.resources.influxdb_metrics
    metrics.write_inference_result(
        task_name="sentiment",
        run_id=context.run_id,
        segment_id="clip_001_seg_003",
        timestamp=segment_start_time,     # datetime or None → now()
        tags={"audio_source": "rtsp_cam1"},
        fields={"label": "positive", "confidence": 0.92},
    )
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from dagster import ConfigurableResource, EnvVar

logger = logging.getLogger(__name__)


class InfluxDBMetricsResource(ConfigurableResource):
    """
    Dagster resource wrapping the InfluxDB v2 write API.

    All fields have sensible defaults matching docker-compose.yml so the
    resource works out of the box in local dev.
    """

    url: str = "http://influxdb:8086"
    token: str = EnvVar("INFLUXDB_ADMIN_TOKEN")
    org: str = "audioops"
    bucket: str = "audio_metrics"
    enabled: bool = True
    batch_size: int = 500
    flush_interval_ms: int = 1000

    # ── Lazy client ──────────────────────────────────────────────────────────

    def _get_write_api(self):
        cache_key = id(self)
        if cache_key not in _write_api_cache:
            try:
                from influxdb_client import InfluxDBClient, WriteOptions
                from influxdb_client.client.write_api import SYNCHRONOUS

                client = InfluxDBClient(
                    url=self.url,
                    token=self.token,
                    org=self.org,
                )
                write_api = client.write_api(
                    write_options=WriteOptions(
                        batch_size=self.batch_size,
                        flush_interval=self.flush_interval_ms,
                    )
                )
                _write_api_cache[cache_key] = (client, write_api)
            except ImportError:
                logger.warning(
                    "influxdb-client not installed — metrics will be no-ops"
                )
                _write_api_cache[cache_key] = (None, None)
            except Exception as exc:
                logger.error("InfluxDB connection failed: %s", exc)
                _write_api_cache[cache_key] = (None, None)

        return _write_api_cache[cache_key][1]

    # ── Generic write ────────────────────────────────────────────────────────

    def write_point(
        self,
        measurement: str,
        tags: Dict[str, str],
        fields: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Write a single point to InfluxDB."""
        if not self.enabled:
            return
        api = self._get_write_api()
        if api is None:
            return

        from influxdb_client import Point

        p = Point(measurement)
        for k, v in tags.items():
            p = p.tag(k, str(v))
        for k, v in fields.items():
            if isinstance(v, str):
                p = p.field(k, v)
            elif isinstance(v, bool):
                p = p.field(k, v)
            elif isinstance(v, int):
                p = p.field(k, v)
            elif isinstance(v, float):
                p = p.field(k, v)
            else:
                p = p.field(k, str(v))
        p = p.time(timestamp or datetime.now(timezone.utc))

        try:
            api.write(bucket=self.bucket, org=self.org, record=p)
        except Exception as exc:
            logger.error("InfluxDB write failed: %s", exc)

    def write_points(
        self,
        measurement: str,
        records: List[Dict[str, Any]],
        common_tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Batch-write a list of records.

        Each record dict must have:
          - "fields": dict of field key→value
        And optionally:
          - "tags": dict of additional tags (merged with common_tags)
          - "timestamp": datetime
        """
        if not self.enabled:
            return
        api = self._get_write_api()
        if api is None:
            return

        from influxdb_client import Point

        points = []
        for rec in records:
            p = Point(measurement)
            merged_tags = {**(common_tags or {}), **(rec.get("tags", {}))}
            for k, v in merged_tags.items():
                p = p.tag(k, str(v))
            for k, v in rec["fields"].items():
                if isinstance(v, (int, float, bool, str)):
                    p = p.field(k, v)
                else:
                    p = p.field(k, str(v))
            p = p.time(rec.get("timestamp", datetime.now(timezone.utc)))
            points.append(p)

        try:
            api.write(bucket=self.bucket, org=self.org, record=points)
        except Exception as exc:
            logger.error("InfluxDB batch write failed (%d points): %s", len(points), exc)

    # ── Convenience writers for common audio patterns ────────────────────────

    def write_inference_result(
        self,
        task_name: str,
        run_id: str,
        segment_id: str,
        fields: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
        audio_offset_sec: Optional[float] = None,
    ) -> None:
        """
        Log one inference result for a single audio segment.

        For sentiment: fields = {"label": "positive", "confidence": 0.92}
        For STT:       fields = {"transcription": "hello world", "confidence": 0.87}
        For nonverbal: fields = {"event": "cough", "confidence": 0.95, "duration_ms": 340}
        """
        merged_tags = {
            "task_name": task_name,
            "run_id": run_id,
            "segment_id": segment_id,
            **(tags or {}),
        }
        merged_fields = dict(fields)
        if audio_offset_sec is not None:
            merged_fields["audio_offset_sec"] = audio_offset_sec
        self.write_point("audio_inference", merged_tags, merged_fields, timestamp)

    def write_training_metric(
        self,
        task_name: str,
        run_id: str,
        epoch: int,
        fields: Dict[str, float],
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log an epoch-level training metric."""
        merged_tags = {
            "task_name": task_name,
            "run_id": run_id,
            **(tags or {}),
        }
        merged_fields = {"epoch": epoch, **fields}
        self.write_point("audio_training", merged_tags, merged_fields)

    def write_pipeline_event(
        self,
        task_name: str,
        run_id: str,
        event: str,
        fields: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log a pipeline lifecycle event (start, complete, fail)."""
        merged_tags = {
            "task_name": task_name,
            "run_id": run_id,
            "event": event,
            **(tags or {}),
        }
        self.write_point("audio_pipeline", merged_tags, fields or {"value": 1})

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "InfluxDBMetricsResource":
        enabled = os.getenv("ENABLE_INFLUXDB", "true").lower() != "false"
        host = os.getenv("INFLUXDB_HOST", "influxdb")
        port = os.getenv("INFLUXDB_PORT", "8086")
        scheme = "https" if os.getenv("INFLUXDB_USE_SSL", "false").lower() == "true" else "http"

        return cls(
            url=os.getenv("INFLUXDB_URL", f"{scheme}://{host}:{port}"),
            token=os.getenv("INFLUXDB_TOKEN", os.getenv("INFLUXDB_ADMIN_TOKEN", "audioopstoken")),
            org=os.getenv("INFLUXDB_ORG", "audioops"),
            bucket=os.getenv("INFLUXDB_BUCKET", "audio_metrics"),
            enabled=enabled,
        )


_write_api_cache: dict[int, tuple] = {}
