"""
InfluxDB Writer
===============
Generic time-series writer for all audio task metrics.

Schema design
-------------
Every metric is an ``AudioMetric`` with:
  - measurement : str   — top-level grouping, e.g. "inference_sentiment"
  - tags        : dict  — indexed labels (low cardinality): task, model,
                          stream_id, source_type, hostname
  - fields      : dict  — actual numeric / string values stored
  - timestamp   : float — Unix epoch seconds (optional, defaults to now)

Pre-defined measurement families
---------------------------------
  inference_{task_name}   — real-time inference results (file or stream)
  training_{task_name}    — per-epoch/step training metrics
  stream_{task_name}      — continuous stream inference results
  system_health           — service heartbeats, queue depths
"""

from __future__ import annotations

import logging
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS, WriteOptions

logger = logging.getLogger(__name__)

_HOSTNAME = socket.gethostname()


# ------------------------------------------------------------------ #
# Data model                                                          #
# ------------------------------------------------------------------ #

@dataclass
class AudioMetric:
    """A single time-series data point for any audio task."""
    measurement: str
    fields: dict[str, Any]
    tags: dict[str, str] = field(default_factory=dict)
    timestamp: Optional[float] = None   # Unix seconds; None = server time

    def to_point(self) -> Point:
        p = Point(self.measurement)

        # Always tag with hostname so multi-node deployments are distinguishable
        p.tag("hostname", _HOSTNAME)
        for k, v in self.tags.items():
            p.tag(k, str(v))

        # Write fields — skip None values, coerce types
        for k, v in self.fields.items():
            if v is None:
                continue
            if isinstance(v, bool):
                p.field(k, int(v))
            elif isinstance(v, (int, float)):
                p.field(k, float(v))
            elif isinstance(v, str):
                p.field(k, v)

        if self.timestamp:
            p.time(int(self.timestamp * 1e9), WritePrecision.NANOSECONDS)

        return p


# ------------------------------------------------------------------ #
# Pre-built metric constructors                                        #
# ------------------------------------------------------------------ #

class Metrics:
    """
    Factory methods that produce AudioMetric objects with the correct
    measurement names, tags, and field schemas for each use-case.
    """

    @staticmethod
    def inference_result(
        task: str,
        model: str,
        label: str,
        confidence: float,
        scores: Optional[dict[str, float]] = None,
        source_type: str = "file",   # "file" | "stream"
        stream_id: str = "",
        audio_duration_s: float = 0.0,
        rms: float = 0.0,
        extra: Optional[dict] = None,
    ) -> AudioMetric:
        """Inference result for any task (sentiment, speech_to_text, etc.)."""
        fields: dict[str, Any] = {
            "confidence": confidence,
            "audio_duration_s": audio_duration_s,
            "rms": rms,
            "label": label,
        }
        if scores:
            for class_name, score in scores.items():
                fields[f"score_{class_name}"] = float(score)
        if extra:
            fields.update({k: v for k, v in extra.items() if v is not None})

        return AudioMetric(
            measurement=f"inference_{task}",
            tags={
                "task": task,
                "model": model,
                "source_type": source_type,
                "stream_id": stream_id or "none",
            },
            fields=fields,
        )

    @staticmethod
    def speech_to_text(
        model: str,
        transcript: str,
        word_count: int,
        confidence: float,
        language: str = "en",
        source_type: str = "file",
        stream_id: str = "",
        audio_duration_s: float = 0.0,
    ) -> AudioMetric:
        """Transcription result — stores transcript as string field."""
        return AudioMetric(
            measurement="inference_speech_to_text",
            tags={
                "task": "speech_to_text",
                "model": model,
                "source_type": source_type,
                "stream_id": stream_id or "none",
                "language": language,
            },
            fields={
                "transcript": transcript[:1000],   # InfluxDB field limit guard
                "word_count": word_count,
                "confidence": confidence,
                "chars_per_second": len(transcript) / max(audio_duration_s, 0.001),
                "audio_duration_s": audio_duration_s,
            },
        )

    @staticmethod
    def training_epoch(
        task: str,
        model: str,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_acc: Optional[float] = None,
        val_acc: Optional[float] = None,
        learning_rate: float = 0.0,
        epoch_duration_s: float = 0.0,
        extra: Optional[dict] = None,
    ) -> AudioMetric:
        """Per-epoch training metrics."""
        fields: dict[str, Any] = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "train_loss": train_loss,
            "learning_rate": learning_rate,
            "epoch_duration_s": epoch_duration_s,
            "progress_pct": 100.0 * epoch / max(total_epochs, 1),
        }
        if val_loss is not None:
            fields["val_loss"] = val_loss
        if train_acc is not None:
            fields["train_acc"] = train_acc
        if val_acc is not None:
            fields["val_acc"] = val_acc
        if extra:
            fields.update(extra)

        return AudioMetric(
            measurement=f"training_{task}",
            tags={"task": task, "model": model},
            fields=fields,
        )

    @staticmethod
    def training_batch(
        task: str,
        model: str,
        epoch: int,
        step: int,
        loss: float,
        learning_rate: float = 0.0,
    ) -> AudioMetric:
        """Per-batch training metrics (higher frequency)."""
        return AudioMetric(
            measurement=f"training_{task}_batch",
            tags={"task": task, "model": model},
            fields={
                "epoch": epoch,
                "step": step,
                "loss": loss,
                "learning_rate": learning_rate,
            },
        )

    @staticmethod
    def stream_heartbeat(
        stream_id: str,
        url: str,
        active: bool,
        frames_processed: int,
        error_count: int,
        ingest_queue_depth: int,
        rms: float = 0.0,
    ) -> AudioMetric:
        """Heartbeat metric for a running RTSP stream."""
        return AudioMetric(
            measurement="stream_health",
            tags={"stream_id": stream_id, "url": url},
            fields={
                "active": int(active),
                "frames_processed": frames_processed,
                "error_count": error_count,
                "ingest_queue_depth": ingest_queue_depth,
                "rms": rms,
            },
        )

    @staticmethod
    def system_health(
        service: str,
        healthy: bool,
        latency_ms: float = 0.0,
        extra: Optional[dict] = None,
    ) -> AudioMetric:
        fields: dict[str, Any] = {
            "healthy": int(healthy),
            "latency_ms": latency_ms,
        }
        if extra:
            fields.update(extra)
        return AudioMetric(
            measurement="system_health",
            tags={"service": service},
            fields=fields,
        )


# ------------------------------------------------------------------ #
# Writer                                                              #
# ------------------------------------------------------------------ #

class InfluxWriter:
    """
    Thread-safe wrapper around the InfluxDB v2 Python client.

    Supports synchronous single writes, batch writes, and a context
    manager for batched usage.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        org: Optional[str] = None,
        bucket: Optional[str] = None,
        batch_size: int = 100,
        flush_interval_ms: int = 2000,
    ):
        self.url = url or os.environ.get("INFLUXDB_URL", "http://influxdb:8086")
        self.token = token or os.environ.get("INFLUXDB_TOKEN", "audio_ops_token")
        self.org = org or os.environ.get("INFLUXDB_ORG", "audio_ops")
        self.bucket = bucket or os.environ.get("INFLUXDB_BUCKET", "audio_metrics")

        self._client = InfluxDBClient(
            url=self.url,
            token=self.token,
            org=self.org,
        )
        # Use async-batch write API for normal writes
        self._write_api = self._client.write_api(
            write_options=WriteOptions(
                batch_size=batch_size,
                flush_interval=flush_interval_ms,
                jitter_interval=0,
                retry_interval=5_000,
                max_retries=3,
                max_retry_delay=30_000,
                exponential_base=2,
            )
        )
        # Synchronous write API for critical single points
        self._write_api_sync = self._client.write_api(write_options=SYNCHRONOUS)
        logger.info("InfluxWriter initialised — %s / %s / %s", self.url, self.org, self.bucket)

    # ------------------------------------------------------------------ #
    # Write methods                                                        #
    # ------------------------------------------------------------------ #

    def write(self, metric: AudioMetric) -> None:
        """Write a single metric (batched internally)."""
        try:
            self._write_api.write(
                bucket=self.bucket,
                org=self.org,
                record=metric.to_point(),
            )
        except Exception as exc:
            logger.warning("InfluxDB write failed: %s", exc)

    def write_sync(self, metric: AudioMetric) -> None:
        """Write a single metric synchronously (blocks until confirmed)."""
        try:
            self._write_api_sync.write(
                bucket=self.bucket,
                org=self.org,
                record=metric.to_point(),
            )
        except Exception as exc:
            logger.warning("InfluxDB sync write failed: %s", exc)

    def write_batch(self, metrics: list[AudioMetric]) -> None:
        """Write a batch of metrics synchronously."""
        if not metrics:
            return
        try:
            points = [m.to_point() for m in metrics]
            self._write_api_sync.write(
                bucket=self.bucket,
                org=self.org,
                record=points,
            )
        except Exception as exc:
            logger.warning("InfluxDB batch write failed (%d points): %s", len(metrics), exc)

    # ------------------------------------------------------------------ #
    # Health / query helpers                                               #
    # ------------------------------------------------------------------ #

    def ping(self) -> bool:
        try:
            self._client.ping()
            return True
        except Exception:
            return False

    def query(self, flux: str) -> list[dict]:
        """Run a Flux query and return results as a list of dicts."""
        try:
            query_api = self._client.query_api()
            tables = query_api.query(flux, org=self.org)
            rows = []
            for table in tables:
                for record in table.records:
                    rows.append({
                        "time": record.get_time(),
                        "measurement": record.get_measurement(),
                        "field": record.get_field(),
                        "value": record.get_value(),
                        **record.values,
                    })
            return rows
        except Exception as exc:
            logger.warning("InfluxDB query failed: %s", exc)
            return []

    def close(self) -> None:
        try:
            self._write_api.close()
            self._client.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ------------------------------------------------------------------ #
# Module-level singleton (lazy-init)                                  #
# ------------------------------------------------------------------ #

_writer: Optional[InfluxWriter] = None


def get_writer() -> InfluxWriter:
    """Return a module-level singleton InfluxWriter."""
    global _writer
    if _writer is None:
        _writer = InfluxWriter()
    return _writer