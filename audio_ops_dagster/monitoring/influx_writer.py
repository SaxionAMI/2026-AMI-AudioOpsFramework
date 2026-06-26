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
  stt_word                — per-word timeline events from STT (one point per word)
  nonverbal_event         — cough, silence, noise events from a parallel classifier
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
    fields:      dict[str, Any]
    tags:        dict[str, str]   = field(default_factory=dict)
    timestamp:   Optional[float]  = None   # Unix seconds; None = server time

    def to_point(self) -> Point:
        p = Point(self.measurement)

        # Always tag with hostname for multi-node distinguishability
        p.tag("hostname", _HOSTNAME)
        for k, v in self.tags.items():
            p.tag(k, str(v))

        # Write fields — skip None, coerce types
        for k, v in self.fields.items():
            if v is None:
                continue
            if isinstance(v, bool):
                p.field(k, int(v))
            elif isinstance(v, (int, float)):
                p.field(k, float(v))
            elif isinstance(v, str):
                p.field(k, v)

        if self.timestamp is not None:
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

    # ── Classifier inference ──────────────────────────────────────────

    @staticmethod
    def inference_result(
        task:             str,
        model:            str,
        label:            str,
        confidence:       float,
        scores:           Optional[dict[str, float]] = None,
        source_type:      str   = "file",
        stream_id:        str   = "",
        audio_duration_s: float = 0.0,
        rms:              float = 0.0,
        extra:            Optional[dict] = None,
    ) -> AudioMetric:
        """Inference result for any classifier task (sentiment, nonverbal, etc.)."""
        fields: dict[str, Any] = {
            "confidence":       confidence,
            "audio_duration_s": audio_duration_s,
            "rms":              rms,
            "label":            label,
        }
        if scores:
            for class_name, score in scores.items():
                fields[f"score_{class_name}"] = float(score)
        if extra:
            fields.update({k: v for k, v in extra.items() if v is not None})

        return AudioMetric(
            measurement=f"inference_{task}",
            tags={
                "task":        task,
                "model":       model,
                "source_type": source_type,
                "stream_id":   stream_id or "none",
            },
            fields=fields,
        )

    # ── STT — full chunk result ───────────────────────────────────────

    @staticmethod
    def speech_to_text(
        model:            str,
        transcript:       str,
        word_count:       int,
        confidence:       float,
        language:         str   = "nl",
        source_type:      str   = "file",
        stream_id:        str   = "",
        audio_duration_s: float = 0.0,
        rms:              float = 0.0,
    ) -> AudioMetric:
        """
        Transcription result for one audio chunk / buffer window.
        Stored in measurement: inference_speech_to_text
        Fields: transcript, word_count, confidence, chars_per_second, rms
        """
        return AudioMetric(
            measurement="inference_speech_to_text",
            tags={
                "task":        "speech_to_text",
                "model":       model,
                "source_type": source_type,
                "stream_id":   stream_id or "none",
                "language":    language,
            },
            fields={
                "transcript":       transcript[:1000],  # InfluxDB string field limit guard
                "word_count":       word_count,
                "confidence":       confidence,
                "rms":              rms,
                "audio_duration_s": audio_duration_s,
                "chars_per_second": len(transcript) / max(audio_duration_s, 0.001),
            },
        )

    # ── STT — per-word timeline ───────────────────────────────────────

    @staticmethod
    def stt_word(
        word:             str,
        confidence:       float,
        t_start:          float,   # seconds relative to chunk start
        t_end:            float,
        stream_id:        str,
        language:         str,
        chunk_wall_time:  float,   # time.time() when this chunk was processed
        buffer_seconds:   float,   # how many seconds the buffer covers
        task:             str = "speech_to_text",
    ) -> AudioMetric:
        """
        One InfluxDB point per spoken word.
        Stored in measurement: stt_word

        Timestamp is the estimated wall-clock time when the word was spoken:
          word_wall_time = chunk_wall_time - buffer_seconds + t_start

        This produces a proper time-series in Grafana where each word
        appears at the moment it was actually spoken, not when it was decoded.
        """
        word_wall_time = chunk_wall_time - buffer_seconds + t_start
        duration_s     = round(t_end - t_start, 3)

        return AudioMetric(
            measurement="stt_word",
            tags={
                "task":      task,
                "stream_id": stream_id or "none",
                "language":  language,
            },
            fields={
                "word":       word.strip(),
                "confidence": float(confidence),
                "duration_s": duration_s,
                # t_start relative to chunk — useful for replay/alignment
                "t_start":    round(t_start, 3),
            },
            timestamp=word_wall_time,
        )

    @staticmethod
    def stt_words_batch(
        words:           list[dict],
        stream_id:       str,
        language:        str,
        chunk_wall_time: float,
        buffer_seconds:  float,
        task:            str = "speech_to_text",
    ) -> list[AudioMetric]:
        """
        Convert the full words list from a Whisper result into a list
        of AudioMetric objects — one per word — ready for write_batch().

        words format (from faster-whisper):
          [{"word": "vandaag.", "t_start": 4.18, "t_end": 4.94, "confidence": 0.892}, ...]
        """
        return [
            Metrics.stt_word(
                word=w.get("word", ""),
                confidence=float(w.get("confidence") or 0.0),
                t_start=float(w.get("t_start", 0.0)),
                t_end=float(w.get("t_end", 0.0)),
                stream_id=stream_id,
                language=language,
                chunk_wall_time=chunk_wall_time,
                buffer_seconds=buffer_seconds,
                task=task,
            )
            for w in words
            if w.get("word", "").strip()   # skip empty/whitespace words
        ]

    # ── Non-verbal event (cough, silence, noise) ──────────────────────

    @staticmethod
    def nonverbal_event(
        label:       str,
        confidence:  float,
        scores:      Optional[dict[str, float]],
        stream_id:   str,
        rms:         float,
        source_type: str = "stream",
        task:        str = "nonverbal",
        model:       str = "nonverbal",
    ) -> AudioMetric:
        """
        Non-verbal audio event — cough, silence, background noise, etc.
        Stored in measurement: inference_nonverbal

        In Grafana, these can be overlaid as annotations on the word
        timeline so you can see coughs relative to spoken words.
        """
        fields: dict[str, Any] = {
            "label":      label,
            "confidence": confidence,
            "rms":        rms,
        }
        if scores:
            for cls, score in scores.items():
                fields[f"score_{cls}"] = float(score)

        return AudioMetric(
            measurement="inference_nonverbal",
            tags={
                "task":        task,
                "model":       model,
                "source_type": source_type,
                "stream_id":   stream_id or "none",
                "event_type":  label,   # tag for easy filtering in Grafana
            },
            fields=fields,
        )

    # ── Training ──────────────────────────────────────────────────────

    @staticmethod
    def training_epoch(
        task:             str,
        model:            str,
        epoch:            int,
        total_epochs:     int,
        train_loss:       float,
        val_loss:         Optional[float] = None,
        train_acc:        Optional[float] = None,
        val_acc:          Optional[float] = None,
        learning_rate:    float = 0.0,
        epoch_duration_s: float = 0.0,
        extra:            Optional[dict] = None,
    ) -> AudioMetric:
        """Per-epoch training metrics."""
        fields: dict[str, Any] = {
            "epoch":            epoch,
            "total_epochs":     total_epochs,
            "train_loss":       train_loss,
            "learning_rate":    learning_rate,
            "epoch_duration_s": epoch_duration_s,
            "progress_pct":     100.0 * epoch / max(total_epochs, 1),
        }
        if val_loss is not None:   fields["val_loss"]   = val_loss
        if train_acc is not None:  fields["train_acc"]  = train_acc
        if val_acc is not None:    fields["val_acc"]    = val_acc
        if extra:                  fields.update(extra)

        return AudioMetric(
            measurement=f"training_{task}",
            tags={"task": task, "model": model},
            fields=fields,
        )

    @staticmethod
    def training_batch(
        task:          str,
        model:         str,
        epoch:         int,
        step:          int,
        loss:          float,
        learning_rate: float = 0.0,
    ) -> AudioMetric:
        """Per-batch training metrics (higher frequency)."""
        return AudioMetric(
            measurement=f"training_{task}_batch",
            tags={"task": task, "model": model},
            fields={
                "epoch":         epoch,
                "step":          step,
                "loss":          loss,
                "learning_rate": learning_rate,
            },
        )

    # ── Stream / system health ────────────────────────────────────────

    @staticmethod
    def stream_heartbeat(
        stream_id:          str,
        url:                str,
        active:             bool,
        frames_processed:   int,
        error_count:        int,
        ingest_queue_depth: int,
        rms:                float = 0.0,
    ) -> AudioMetric:
        """Heartbeat metric for a running RTSP stream."""
        return AudioMetric(
            measurement="stream_health",
            tags={"stream_id": stream_id, "url": url},
            fields={
                "active":              int(active),
                "frames_processed":    frames_processed,
                "error_count":         error_count,
                "ingest_queue_depth":  ingest_queue_depth,
                "rms":                 rms,
            },
        )

    @staticmethod
    def system_health(
        service:    str,
        healthy:    bool,
        latency_ms: float = 0.0,
        extra:      Optional[dict] = None,
    ) -> AudioMetric:
        fields: dict[str, Any] = {
            "healthy":    int(healthy),
            "latency_ms": latency_ms,
        }
        if extra:
            fields.update(extra)
        return AudioMetric(
            measurement="system_health",
            tags={"service": service},
            fields=fields,
        )

    # ── Pipeline events (Dagster) ─────────────────────────────────────

    @staticmethod
    def pipeline_event(
        task_name: str,
        run_id:    str,
        event:     str,
        fields:    Optional[dict] = None,
    ) -> AudioMetric:
        """Dagster pipeline lifecycle event."""
        f: dict[str, Any] = {"event": event}
        if fields:
            f.update({k: v for k, v in fields.items() if v is not None})
        return AudioMetric(
            measurement="pipeline_events",
            tags={
                "task":   task_name,
                "run_id": run_id,
                "event":  event,
            },
            fields=f,
        )


# ------------------------------------------------------------------ #
# Writer                                                              #
# ------------------------------------------------------------------ #

class InfluxWriter:
    """
    Thread-safe wrapper around the InfluxDB v2 Python client.

    Supports synchronous single writes, batch writes, and a context
    manager for batched usage.

    Usage in inference service (standalone, no Dagster):
        influx = InfluxWriter()
        influx.write(Metrics.inference_result(...))
        influx.write_batch(Metrics.stt_words_batch(...))

    Usage in Dagster assets via resource:
        influx = context.resources.influxdb_metrics
        influx.write_point(Metrics.training_epoch(...))
    """

    def __init__(
        self,
        url:              Optional[str] = None,
        token:            Optional[str] = None,
        org:              Optional[str] = None,
        bucket:           Optional[str] = None,
        batch_size:       int  = 100,
        flush_interval_ms: int = 2000,
    ):
        self.url    = url    or os.environ.get("INFLUXDB_URL",    "http://influxdb:8086")
        self.token  = token  or os.environ.get("INFLUXDB_TOKEN",  "")
        self.org    = org    or os.environ.get("INFLUXDB_ORG",    "audioops")
        self.bucket = bucket or os.environ.get("INFLUXDB_BUCKET", "audio_metrics")

        if not self.token:
            logger.warning("No INFLUXDB_TOKEN set — metrics will not be written")
            self._ok = False
            return

        try:
            self._client = InfluxDBClient(url=self.url, token=self.token, org=self.org)

            # Batched async write for high-frequency stream metrics
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

            # Synchronous write for critical single points (training events etc.)
            self._write_api_sync = self._client.write_api(write_options=SYNCHRONOUS)

            self._ok = True
            logger.info("InfluxWriter ready — %s / %s / %s", self.url, self.org, self.bucket)

        except Exception as exc:
            logger.warning("InfluxWriter init failed: %s", exc)
            self._ok = False

    # ------------------------------------------------------------------ #
    # Write methods                                                        #
    # ------------------------------------------------------------------ #

    def write(self, metric: AudioMetric) -> None:
        """Write a single metric (batched internally — non-blocking)."""
        if not self._ok:
            return
        try:
            self._write_api.write(
                bucket=self.bucket, org=self.org, record=metric.to_point()
            )
        except Exception as exc:
            logger.warning("InfluxDB write failed: %s", exc)

    def write_sync(self, metric: AudioMetric) -> None:
        """Write synchronously — blocks until InfluxDB confirms."""
        if not self._ok:
            return
        try:
            self._write_api_sync.write(
                bucket=self.bucket, org=self.org, record=metric.to_point()
            )
        except Exception as exc:
            logger.warning("InfluxDB sync write failed: %s", exc)

    def write_batch(self, metrics: list[AudioMetric]) -> None:
        """
        Write a list of metrics as a single batch (synchronous).
        Used for per-word events — sends all words from one chunk at once.
        """
        if not self._ok or not metrics:
            return
        try:
            points = [m.to_point() for m in metrics]
            self._write_api_sync.write(
                bucket=self.bucket, org=self.org, record=points
            )
            logger.debug("Batch wrote %d points to InfluxDB", len(points))
        except Exception as exc:
            logger.warning("InfluxDB batch write failed (%d points): %s", len(metrics), exc)

    # Dagster resource compatibility — write_point is an alias for write_sync
    def write_point(self, metric: AudioMetric) -> None:
        self.write_sync(metric)

    # Dagster resource compatibility — write_pipeline_event convenience wrapper
    def write_pipeline_event(
        self,
        task_name: str,
        run_id:    str,
        event:     str,
        fields:    Optional[dict] = None,
    ) -> None:
        self.write_sync(Metrics.pipeline_event(task_name, run_id, event, fields))

    # Dagster resource compatibility — write_points used by assets.py
    def write_points(
        self,
        measurement: str,
        records:     list[dict],
        common_tags: Optional[dict] = None,
    ) -> None:
        """
        Write a list of raw field/tag dicts to InfluxDB.
        Used by Dagster assets.py for batch inference result logging.
        """
        if not self._ok or not records:
            return
        try:
            points = []
            for rec in records:
                p = Point(measurement)
                p.tag("hostname", _HOSTNAME)
                for k, v in (common_tags or {}).items():
                    p.tag(k, str(v))
                for k, v in rec.get("tags", {}).items():
                    p.tag(k, str(v))
                for k, v in rec.get("fields", {}).items():
                    if v is None:
                        continue
                    if isinstance(v, bool):
                        p.field(k, int(v))
                    elif isinstance(v, (int, float)):
                        p.field(k, float(v))
                    elif isinstance(v, str):
                        p.field(k, v)
                points.append(p)
            self._write_api_sync.write(bucket=self.bucket, org=self.org, record=points)
        except Exception as exc:
            logger.warning("InfluxDB write_points failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Health / query                                                       #
    # ------------------------------------------------------------------ #

    def ping(self) -> bool:
        if not self._ok:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            return False

    def query(self, flux: str) -> list[dict]:
        """Run a Flux query and return results as a list of dicts."""
        if not self._ok:
            return []
        try:
            query_api = self._client.query_api()
            tables    = query_api.query(flux, org=self.org)
            rows = []
            for table in tables:
                for record in table.records:
                    rows.append({
                        "time":        record.get_time(),
                        "measurement": record.get_measurement(),
                        "field":       record.get_field(),
                        "value":       record.get_value(),
                        **record.values,
                    })
            return rows
        except Exception as exc:
            logger.warning("InfluxDB query failed: %s", exc)
            return []

    def close(self) -> None:
        if not self._ok:
            return
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
    """Return module-level singleton InfluxWriter."""
    global _writer
    if _writer is None:
        _writer = InfluxWriter()
    return _writer