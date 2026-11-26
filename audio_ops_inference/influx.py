"""
influx.py — optional InfluxDB writer for the streaming service.

Off unless ``ENABLE_INFLUXDB=true`` (or, for backwards compatibility, an
``INFLUXDB_TOKEN`` is set without ``ENABLE_INFLUXDB``). Every write swallows
errors: metrics must never break inference. Requires the ``metrics`` extra.
"""

from __future__ import annotations

import socket
import time

from audio_ops_dagster._log import get_logger

logger = get_logger("audio_ops_inference.influx")

_HOSTNAME = socket.gethostname()


class InfluxWriter:
    def __init__(self, enabled: bool = False, url: str = "", token: str = "",
                 org: str = "audioops", bucket: str = "audio_metrics"):
        self._ok = False
        self._org, self._bucket = org, bucket
        if not enabled:
            return
        if not token:
            logger.warning("ENABLE_INFLUXDB=true but no INFLUXDB_TOKEN — metrics will not be written")
            return
        try:
            from influxdb_client import InfluxDBClient
            from influxdb_client.client.write_api import SYNCHRONOUS
            self._client = InfluxDBClient(url=url, token=token, org=org)
            self._write = self._client.write_api(write_options=SYNCHRONOUS)
            self._ok    = True
            logger.info("InfluxDB connected: %s", url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("InfluxDB init failed: %s", exc)

    @property
    def enabled(self) -> bool:
        return self._ok

    def write_classifier(self, task, label, confidence, scores, rms, source,
                         stream_id="", duration_s=0.0, media_ts=0.0):
        if not self._ok:
            return
        try:
            import math

            from influxdb_client import Point

            def _f(x, default=0.0):
                try:
                    v = float(x)
                    return v if math.isfinite(v) else default
                except (TypeError, ValueError):
                    return default

            p = (Point(f"inference_{task}")
                 .tag("hostname", _HOSTNAME).tag("task", task)
                 .tag("source_type", source).tag("stream_id", stream_id or "none")
                 .field("label", str(label))
                 .field("confidence", _f(confidence))
                 .field("rms", _f(rms))
                 .field("audio_duration_s", _f(duration_s))
                 .field("media_ts", _f(media_ts)))
            for cls, score in (scores or {}).items():
                p = p.field(f"score_{cls}", _f(score))
            self._write.write(bucket=self._bucket, org=self._org, record=p)
        except Exception as exc:  # noqa: BLE001
            logger.warning("InfluxDB classifier write error: %s", exc)

    def write_stt(self, task, transcription, confidence, word_count, language,
                  rms, source, stream_id="", duration_s=0.0, media_ts=0.0):
        if not self._ok:
            return
        try:
            from influxdb_client import Point
            p = (Point(f"inference_{task}")
                 .tag("hostname", _HOSTNAME).tag("task", task)
                 .tag("source_type", source).tag("stream_id", stream_id or "none")
                 .tag("language", language or "unknown")
                 .field("transcription", transcription[:1000])
                 .field("confidence", confidence).field("word_count", word_count)
                 .field("rms", rms).field("audio_duration_s", duration_s)
                 .field("media_ts", float(media_ts))
                 .field("chars_per_second", len(transcription) / max(duration_s, 0.001)))
            self._write.write(bucket=self._bucket, org=self._org, record=p)
        except Exception as exc:  # noqa: BLE001
            logger.debug("InfluxDB write error: %s", exc)

    def write_stt_words(self, task, words, stream_id, language):
        """words carry media-clock t_start already (set by the wrapper)."""
        if not self._ok or not words:
            return
        try:
            from influxdb_client import Point, WritePrecision
            records = []
            base_wall = time.time()
            for w in words:
                word = (w.get("word") or "").strip()
                if not word:
                    continue
                t_start = float(w.get("t_start", 0.0))
                p = (Point("stt_word")
                     .tag("hostname", _HOSTNAME).tag("task", task)
                     .tag("stream_id", stream_id or "none")
                     .tag("language", language or "unknown")
                     .field("word", word)
                     .field("confidence", float(w.get("confidence") or 0.0))
                     .field("media_ts", t_start)
                     .field("t_start", round(t_start, 3))
                     .time(int(base_wall * 1e9), WritePrecision.NANOSECONDS))
                records.append(p)
            if records:
                self._write.write(bucket=self._bucket, org=self._org, record=records)
        except Exception as exc:  # noqa: BLE001
            logger.debug("InfluxDB word write error: %s", exc)

    def write_nonverbal(self, task, primary_event, primary_event_conf, curated_scores,
                        top_classes_json, rms, source, stream_id="", duration_s=0.0,
                        inference_time_ms=0.0, media_ts=0.0):
        if not self._ok:
            return
        try:
            from influxdb_client import Point
            p = (Point(f"inference_{task}")
                 .tag("hostname", _HOSTNAME).tag("task", task)
                 .tag("source_type", source).tag("stream_id", stream_id or "none")
                 .field("primary_event", primary_event)
                 .field("primary_event_conf", float(primary_event_conf))
                 .field("top_classes_json", top_classes_json[:2000])
                 .field("rms", float(rms)).field("audio_duration_s", float(duration_s))
                 .field("inference_time_ms", float(inference_time_ms))
                 .field("media_ts", float(media_ts)))
            for group, conf in (curated_scores or {}).items():
                p = p.field(f"{group}_conf", float(conf))
            self._write.write(bucket=self._bucket, org=self._org, record=p)
        except Exception as exc:  # noqa: BLE001
            logger.debug("InfluxDB nonverbal write error: %s", exc)

    def write_fusion(self, stream_id, media_ts, transcript, n_segments,
                     primary_sentiment, n_nonverbal):
        if not self._ok:
            return
        try:
            from influxdb_client import Point
            p = (Point("inference_fusion")
                 .tag("hostname", _HOSTNAME).tag("stream_id", stream_id or "none")
                 .field("fused_transcript", (transcript or "")[:4000])
                 .field("media_ts", float(media_ts))
                 .field("segments", int(n_segments))
                 .field("primary_sentiment", primary_sentiment or "")
                 .field("nonverbal_count", int(n_nonverbal)))
            self._write.write(bucket=self._bucket, org=self._org, record=p)
        except Exception as exc:  # noqa: BLE001
            logger.debug("InfluxDB fusion write error: %s", exc)

    def write_stream_health(self, stream_id, active, frames, errors, rms):
        if not self._ok:
            return
        try:
            from influxdb_client import Point
            p = (Point("stream_health")
                 .tag("hostname", _HOSTNAME).tag("stream_id", stream_id)
                 .field("active", int(active)).field("frames_processed", frames)
                 .field("error_count", errors).field("rms", rms))
            self._write.write(bucket=self._bucket, org=self._org, record=p)
        except Exception as exc:  # noqa: BLE001
            logger.debug("InfluxDB write error: %s", exc)

    def ping(self):
        if not self._ok:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            return False
