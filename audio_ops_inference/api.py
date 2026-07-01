"""
Audio Ops — Standalone Inference Service v4.0
=============================================
What changed vs v3.1
--------------------
1. SINGLE MEDIA CLOCK
   Every chunk processed (RTSP-pull OR browser-push) carries `media_t0`,
   the seconds-since-stream-start of that chunk. Every wrapper stamps its
   result with `media_ts_start` / `media_ts_end`, and STT word timestamps are
   placed on the media clock. This is what makes alignment possible — nothing
   is timestamped by "wall clock at the moment inference finished" anymore.

2. BROWSER AUDIO INGEST (microphone OR a file the browser plays)
   New WebSocket endpoint  /ws/ingest/{stream_id}  accepts 16 kHz mono PCM16
   frames pushed from the browser and runs the exact same inference + fusion
   path as the RTSP worker. Because the browser owns the audio clock, the
   events it gets back line up with its own playhead. This is the proper fix
   for "the audio I hear doesn't align with the events".

3. FUSION LAYER
   One TranscriptFuser per stream merges STT + nonverbal + sentiment onto the
   media timeline. A new SSE/WS event {"task":"fusion", ...} carries the
   rolling fused transcript + structured segments. New endpoints expose the
   timeline and an on-demand LLM summary.

The classic RTSP file demo (/triage/start) still works unchanged.
"""

from __future__ import annotations

import asyncio
import collections
import io
import json
import logging
import os
import re
import socket
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import (
    FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("inference")

# ------------------------------------------------------------------ #
# Model caches -> ONE persistent directory                            #
# ------------------------------------------------------------------ #
# Set BEFORE torch / tensorflow / faster-whisper are imported (they read these
# at import time). Mounting MODEL_CACHE_DIR as a Docker volume makes Whisper and
# every HuggingFace asset persist across restarts/rebuilds and stops re-downloads.
# YAMNet (TF-Hub) is cached via TFHUB_CACHE_DIR. After a one-time prefetch
# (see download_models.py) set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 to
# forbid all runtime network access.
_MODEL_CACHE = os.environ.get("MODEL_CACHE_DIR", "/app/models")
os.environ.setdefault("HF_HOME",               os.path.join(_MODEL_CACHE, "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(_MODEL_CACHE, "hf"))
os.environ.setdefault("TRANSFORMERS_CACHE",    os.path.join(_MODEL_CACHE, "hf"))
os.environ.setdefault("TFHUB_CACHE_DIR",       os.path.join(_MODEL_CACHE, "tfhub"))
for _p in ("hf", "tfhub"):
    try:
        os.makedirs(os.path.join(_MODEL_CACHE, _p), exist_ok=True)
    except Exception:
        pass

# --- Fusion layer (OPTIONAL, off by default; enable with ENABLE_FUSION=true) --
# Fusion is a project-specific add-on (transcript + nonverbal + sentiment merged
# on one timeline). It lives in audio_ops_dagster/extras and is not part of the
# generic core. When disabled or unavailable, the API serves everything else.
_FUSION_ENABLED = os.getenv("ENABLE_FUSION", "false").lower() == "true"
if _FUSION_ENABLED:
    try:
        from audio_ops_dagster.extras.fusion import TranscriptFuser
        _FUSION_OK = True
    except Exception as _exc:  # pragma: no cover
        logger.warning("Fusion enabled but unavailable (%s) — running without fusion", _exc)
        TranscriptFuser = None  # type: ignore
        _FUSION_OK = False
else:
    TranscriptFuser = None  # type: ignore
    _FUSION_OK = False

# --- LLM client (optional, opt-in via ENABLE_LLM) --------------------------
try:
    from audio_ops_dagster.core.llm_client import LLMClient
    _llm_client = LLMClient.from_env()
except Exception as _exc:  # pragma: no cover
    logger.warning("LLMClient unavailable (%s)", _exc)
    _llm_client = None

MINIO_ENDPOINT   = os.environ.get("MINIO_ENDPOINT",      "minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY",    "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY",    "minioadmin")
MINIO_BUCKET     = os.environ.get("MINIO_MODELS_BUCKET", "audioops-bucket")
MODELS_DIR       = os.environ.get("MODELS_DIR",          "/app/models")
TASKS_DIR        = os.environ.get("TASKS_DIR",           "/app/tasks")

INFLUXDB_URL     = os.environ.get("INFLUXDB_URL",        "http://influxdb:8086")
INFLUXDB_TOKEN   = os.environ.get("INFLUXDB_TOKEN",      "")
INFLUXDB_ORG     = os.environ.get("INFLUXDB_ORG",        "audioops")
INFLUXDB_BUCKET  = os.environ.get("INFLUXDB_BUCKET",     "audio_metrics")

SAMPLE_RATE = 16_000
CHUNK_MS    = int(os.environ.get("CHUNK_MS", "1000"))
_HOSTNAME   = socket.gethostname()

# A WAV the browser can fetch + play (audible) AND stream for analysis, so the
# demo is heard in sync with the events. Mount your test_audio into the
# inference container, or point this at any readable wav.
DEMO_AUDIO_PATH = os.environ.get("DEMO_AUDIO_PATH",
                                 "/test_audio/long_dutch_consult_doctor.wav")

# How often to emit a rolling fusion event (in processed chunks). 1 = every chunk.
FUSION_EMIT_EVERY   = int(os.environ.get("FUSION_EMIT_EVERY", "1"))
FUSION_EMIT_WINDOW_S = float(os.environ.get("FUSION_EMIT_WINDOW_S", "120"))
FUSION_LLM_WINDOW_S = float(os.environ.get("FUSION_LLM_WINDOW_S", "60"))

# Non-verbal leniency: surface the best curated group above this floor even when
# speech/silence is the dominant class. Lower it to catch more, raise to reduce
# noise. Groups in NONVERBAL_BACKGROUND are never surfaced as events.
NONVERBAL_MIN_CONF  = float(os.environ.get("NONVERBAL_MIN_CONF", "0.05"))
NONVERBAL_BACKGROUND = set(
    s.strip() for s in os.environ.get("NONVERBAL_BACKGROUND", "speech,silence,unknown").split(",")
    if s.strip())

# Max chunks queued for inference before we drop the oldest to stay real-time.
# Bounds end-to-end latency regardless of model speed (a slow Whisper pass can no
# longer make the whole stream fall behind). ~ seconds of audio at CHUNK_MS.
PROC_QUEUE_MAX = int(os.environ.get("PROC_QUEUE_MAX", "5"))
# The speech lane gets a deeper queue: a 5s Whisper buffer plus a slow pass can
# briefly back up several chunks; we don't want to lose speech to that.
STT_QUEUE_MAX = int(os.environ.get("STT_QUEUE_MAX", "12"))

# ------------------------------------------------------------------ #
# Stream auto-discovery (generic — no project-specific stream names).  #
#                                                                      #
# When STREAM_AUTODISCOVER is on, the service polls the MediaMTX API   #
# and runs the loaded task models on whatever paths are publishing —   #
# it knows nothing about what those streams represent. Projects decide  #
# stream names by what they publish; the dashboard keys off stream_id. #
# ------------------------------------------------------------------ #
AUTODISCOVER          = os.environ.get("STREAM_AUTODISCOVER", "false").lower() in ("1", "true", "yes")
MEDIAMTX_API_URL      = os.environ.get("MEDIAMTX_API_URL", os.environ.get("MEDIAMTX_API", "http://mediamtx:9997"))
RTSP_BASE_URL         = os.environ.get("RTSP_BASE_URL", "rtsp://mediamtx:8554")
AUTODISCOVER_INTERVAL = float(os.environ.get("STREAM_AUTODISCOVER_INTERVAL", "5"))
AUTODISCOVER_FILTER   = os.environ.get("STREAM_AUTODISCOVER_FILTER", "")  # optional regex on path name

# stream_ids the auto-discovery loop created (so it never removes manual ones)
_autodiscovered: set = set()


# ------------------------------------------------------------------ #
# InfluxDB writer                                                     #
# ------------------------------------------------------------------ #

class InfluxWriter:
    def __init__(self):
        self._ok = False
        if not INFLUXDB_TOKEN:
            logger.warning("No INFLUXDB_TOKEN — metrics will not be written")
            return
        try:
            from influxdb_client import InfluxDBClient
            from influxdb_client.client.write_api import SYNCHRONOUS
            self._client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
            self._write = self._client.write_api(write_options=SYNCHRONOUS)
            self._ok    = True
            logger.info("InfluxDB connected: %s", INFLUXDB_URL)
        except Exception as exc:
            logger.warning("InfluxDB init failed: %s", exc)

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
            self._write.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=p)
        except Exception as exc:
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
            self._write.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=p)
        except Exception as exc:
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
                self._write.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=records)
        except Exception as exc:
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
            self._write.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=p)
        except Exception as exc:
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
            self._write.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=p)
        except Exception as exc:
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
            self._write.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=p)
        except Exception as exc:
            logger.debug("InfluxDB write error: %s", exc)

    def ping(self):
        if not self._ok:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            return False


# ------------------------------------------------------------------ #
# Model (re)construction helpers — unchanged from v3.1                #
# ------------------------------------------------------------------ #

def _load_task_config(task: str) -> dict:
    import yaml
    path = os.path.join(TASKS_DIR, task, "config.yaml")
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _build_architecture(architecture: str, num_classes: int, input_channels: int = 3):
    import torch.nn as nn
    from torchvision.models import get_model
    model = get_model(architecture, weights=None)
    if input_channels != 3:
        if hasattr(model, "conv1"):
            old = model.conv1
            model.conv1 = nn.Conv2d(input_channels, old.out_channels,
                                    kernel_size=old.kernel_size, stride=old.stride,
                                    padding=old.padding, bias=old.bias is not None)
        elif hasattr(model, "features"):
            old = model.features[0][0]
            model.features[0][0] = nn.Conv2d(input_channels, old.out_channels,
                                             kernel_size=old.kernel_size, stride=old.stride,
                                             padding=old.padding, bias=old.bias is not None)
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        import torch.nn as nn2
        if isinstance(model.classifier, nn2.Sequential):
            in_f = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_f, num_classes)
        else:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif hasattr(model, "head"):
        model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def _reconstruct_model(loaded, task: str):
    import torch.nn as nn
    if isinstance(loaded, nn.Module):
        return loaded
    if not isinstance(loaded, dict):
        raise ValueError(f"Unknown checkpoint type: {type(loaded)}")
    if "model" in loaded and isinstance(loaded["model"], nn.Module):
        model = loaded["model"]
        model.class_names    = loaded.get("class_names",        getattr(model, "class_names", []))
        model.input_shape    = loaded.get("input_shape",        getattr(model, "input_shape", [1, 128, 87]))
        model.architecture   = loaded.get("model_architecture", getattr(model, "architecture", ""))
        model.task_name      = loaded.get("task_name",          task)
        model.feature_config = loaded.get("feature_config", {})
        return model
    if "model_state_dict" in loaded:
        state_dict   = loaded["model_state_dict"]
        class_names  = loaded.get("class_names", [])
        num_classes  = loaded.get("num_classes", len(class_names))
        architecture = loaded.get("model_architecture", "")
        input_shape  = loaded.get("input_shape", None)
    else:
        state_dict, class_names, num_classes, architecture, input_shape = loaded, [], 0, "", None
    cfg          = _load_task_config(task)
    num_classes  = num_classes  or cfg.get("num_classes", 2)
    class_names  = class_names  or cfg.get("class_labels", [str(i) for i in range(num_classes)])
    architecture = architecture or cfg.get("model_architecture", "")
    input_shape  = input_shape  or cfg.get("input_shape", [3, 224, 224])
    if not architecture:
        raise ValueError(f"No model_architecture for task '{task}'")
    model = _build_architecture(architecture, num_classes, input_shape[0] if input_shape else 3)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning("State dict mismatch '%s' — missing: %s unexpected: %s",
                       task, missing[:3], unexpected[:3])
    model.class_names  = class_names
    model.input_shape  = input_shape
    model.architecture = architecture
    return model


# ------------------------------------------------------------------ #
# Model wrappers — now media-clock aware                              #
# ------------------------------------------------------------------ #

class ModelWrapper:
    """Spectrogram classifier (e.g. sentiment). Runs per chunk."""

    def __init__(self, model, task_name: str):
        self.task_name   = task_name
        self.model       = model
        self.class_names = getattr(model, "class_names", [])
        self.input_shape = getattr(model, "input_shape", [1, 128, 87])
        self.feature_cfg = getattr(model, "feature_config", {
            "feature_type": "mel_spectrogram", "sample_rate": 16000,
            "n_mels": 128, "n_fft": 512, "hop_length": 160, "clip_duration_s": 3.0,
        })

    def _extract_features(self, audio: np.ndarray):
        import torch
        cfg = self.feature_cfg
        ft  = cfg.get("feature_type", "spectrogram")
        sr  = int(cfg.get("sample_rate", SAMPLE_RATE))

        # Use the SAME feature extractor the model was trained with, from the
        # framework's registry. This guarantees train/serve parity — a mismatch
        # here feeds the model malformed input and produces NaN/garbage output.
        try:
            from audio_ops_dagster.core.features import get_feature_extractor
            extractor = get_feature_extractor(ft)
            features = extractor(audio.astype("float32"), sr)   # -> (F, T) float32
        except Exception as exc:  # noqa: BLE001
            logger.warning("feature extractor '%s' unavailable (%s) — falling back to raw", ft, exc)
            return torch.from_numpy(np.ascontiguousarray(audio)).float()

        # waveform/raw come back 1-D; return as-is, predict() will batch it
        if features.ndim == 1:
            return torch.from_numpy(np.ascontiguousarray(features)).float()

        # pad/crop the time axis to the model's expected width
        target_w = self.input_shape[2] if len(self.input_shape) == 3 else features.shape[1]
        if features.shape[1] < target_w:
            features = np.pad(features, ((0, 0), (0, target_w - features.shape[1])))
        elif features.shape[1] > target_w:
            features = features[:, :target_w]

        t = torch.from_numpy(np.ascontiguousarray(features)).float()
        # guard against any non-finite values reaching the model
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        return t   # (F, T) — predict() normalizes to 4D

    def predict(self, audio: np.ndarray, media_t0: float = 0.0) -> dict:
        import torch
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        dur = len(audio) / SAMPLE_RATE
        tensor = self._extract_features(audio)
        # Conv2D classifiers want (batch, channel, F, T). Extractors may return
        # (F,T), (C,F,T), or already-batched tensors — normalize to exactly 4D.
        while tensor.dim() < 4:
            tensor = tensor.unsqueeze(0)
        while tensor.dim() > 4:
            tensor = tensor.squeeze(0)
        with torch.no_grad():
            out = self.model(tensor)
        logits = out.logits if hasattr(out, "logits") else out
        probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        idx    = int(np.argmax(probs))
        labels = self.class_names or [str(i) for i in range(len(probs))]
        return {
            "label":          labels[idx] if idx < len(labels) else str(idx),
            "confidence":     float(np.max(probs)),
            "scores":         {lbl: float(p) for lbl, p in zip(labels, probs)},
            "media_ts_start": round(media_t0, 3),
            "media_ts_end":   round(media_t0 + dur, 3),
        }


# Whisper hallucinates stock phrases on silence/noise (YouTube training data).
# These are the common Dutch ones; matched case-insensitively, punctuation-stripped.
_STT_HALLUCINATIONS = {
    "bedankt voor het kijken", "bedankt voor het kijken en tot de volgende keer",
    "ondertiteld door de amara.org gemeenschap",
    "ondertiteling door de amara.org gemeenschap",
    "tot de volgende keer", "tot de volgende video", "abonneer je",
    "abonneer op het kanaal", "amara.org", "dank je wel", "dankjewel",
    "ondertitels ingediend door de amara.org gemeenschap",
}


def _norm_text(t: str) -> str:
    return (t or "").strip().lower().strip(" .!?-…")


def _is_hallucination(text: str) -> bool:
    t = _norm_text(text)
    if not t:
        return True
    if t in _STT_HALLUCINATIONS:
        return True
    words = t.split()
    # degenerate repetition like "ja ja ja ja" or a single token echoed
    if len(words) >= 4 and len(set(words)) <= 2:
        return True
    return False


class STTModelWrapper:
    """Buffers chunks to buffer_seconds, then transcribes. Word ts on media clock."""

    def __init__(self, task, task_name: str, buffer_seconds: float = 5.0,
                 silence_rms: float = 0.006):
        self.task            = task
        self.task_name       = task_name
        self.class_names     = []
        self.input_shape     = []
        self._last_prompt    = ""
        self._last_text      = ""
        self._silence_rms    = silence_rms
        self._buffer_seconds = buffer_seconds
        self._buffer         = collections.deque()
        self._buffer_samples = 0
        self._target_samples = int(SAMPLE_RATE * buffer_seconds)
        self._pending_start  = None       # media_t0 of the first chunk buffered
        self._last_result    = {
            "transcription": "", "confidence": 0.0, "word_count": 0,
            "words": [], "language": "", "buffering": True,
            "media_ts_start": 0.0, "media_ts_end": 0.0,
        }

    def predict(self, audio: np.ndarray, media_t0: float = 0.0) -> dict:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        if self._pending_start is None:
            self._pending_start = media_t0

        self._buffer.append(audio)
        self._buffer_samples += len(audio)

        if self._buffer_samples < self._target_samples:
            return {**self._last_result, "buffering": True}

        combined = np.concatenate(list(self._buffer))
        self._buffer.clear()
        self._buffer_samples = 0
        buffer_start = self._pending_start if self._pending_start is not None else media_t0
        self._pending_start = None
        buffer_dur = len(combined) / SAMPLE_RATE

        # --- silence gate: don't transcribe near-silence (kills hallucinations
        #     like "Bedankt voor het kijken!" and saves the Whisper call) ---
        rms = float(np.sqrt(np.mean(combined ** 2))) if combined.size else 0.0
        if rms < self._silence_rms:
            self._last_prompt = ""          # reset so we don't prime a loop
            self._last_text   = ""
            self._last_result = {
                "transcription": "", "confidence": 0.0, "word_count": 0,
                "words": [], "language": self._last_result.get("language", ""),
                "buffering": False, "silence": True,
                "media_ts_start": round(buffer_start, 3),
                "media_ts_end":   round(buffer_start + buffer_dur, 3),
            }
            return self._last_result

        try:
            import inspect
            sig = inspect.signature(self.task.predict)
            # Pass t_offset=buffer_start so word timestamps land on the media clock.
            # Feeding the previous transcript back as initial_prompt makes Whisper
            # loop ("Ik heb een dokter nodig. Ik... Ik heb een...") on ambiguous /
            # noisy audio, so it is OFF unless STT_USE_PROMPT=1.
            use_prompt = os.environ.get("STT_USE_PROMPT", "0") == "1"
            if use_prompt and "initial_prompt" in sig.parameters:
                result = self.task.predict(combined, t_offset=buffer_start,
                                           initial_prompt=self._last_prompt)
            else:
                result = self.task.predict(combined, t_offset=buffer_start)

            new_text = result.get("transcription", "")
            # --- drop hallucinated / duplicated transcripts ---
            if _is_hallucination(new_text) or _norm_text(new_text) == self._last_text:
                result["transcription"] = ""
                result["words"]         = []
                result["word_count"]    = 0
                # do NOT feed this back as a prompt (that's what loops the phrase)
            else:
                self._last_text   = _norm_text(new_text)
                self._last_prompt = (self._last_prompt + " " + new_text).strip()[-200:]

            result["media_ts_start"] = round(buffer_start, 3)
            result["media_ts_end"]   = round(buffer_start + buffer_dur, 3)
            self._last_result = {**result, "buffering": False}
        except Exception as exc:
            logger.warning("STT predict error [%s]: %s", self.task_name, exc)
            self._last_result = {**self._last_result, "buffering": False, "error": str(exc)}

        return self._last_result


class YamnetModelWrapper:
    """Buffers chunks to buffer_seconds, then runs YAMNet. Event on media clock."""
    WRAPPER_TYPE = "nonverbal_pretrained"

    def __init__(self, task, task_name: str, buffer_seconds: float = 2.0):
        self.task            = task
        self.task_name       = task_name
        self.class_names     = []
        self.input_shape     = []
        self._buffer         = collections.deque()
        self._buffer_samples = 0
        self._target_samples = int(SAMPLE_RATE * buffer_seconds)
        self._buffer_seconds = buffer_seconds
        self._pending_start  = None
        self._last_result    = {
            "label": "unknown", "confidence": 0.0, "scores": {},
            "primary_event": "unknown", "primary_event_conf": 0.0,
            "top_classes": [], "duration_sec": 0.0, "inference_time_ms": 0.0,
            "buffering": True, "media_ts_start": 0.0, "media_ts_end": 0.0,
            "event_media_ts": 0.0,
        }

    def predict(self, audio: np.ndarray, media_t0: float = 0.0) -> dict:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        if self._pending_start is None:
            self._pending_start = media_t0

        self._buffer.append(audio)
        self._buffer_samples += len(audio)
        if self._buffer_samples < self._target_samples:
            return {**self._last_result, "buffering": True}

        combined = np.concatenate(list(self._buffer))
        self._buffer.clear()
        self._buffer_samples = 0
        buffer_start = self._pending_start if self._pending_start is not None else media_t0
        self._pending_start = None
        buffer_dur = len(combined) / SAMPLE_RATE

        try:
            result = self.task.predict(combined, sample_rate=SAMPLE_RATE, source="stream")
            # locate the best frame's offset within the buffer for precise placement
            event_offset = 0.0
            yr = result.get("_yamnet_result")
            if yr is not None and getattr(yr, "frames", None):
                best = max(yr.frames, key=lambda f: f.primary_event_conf)
                event_offset = float(best.offset_sec)
            result["media_ts_start"] = round(buffer_start, 3)
            result["media_ts_end"]   = round(buffer_start + buffer_dur, 3)
            result["event_media_ts"] = round(buffer_start + event_offset, 3)
            # _yamnet_result is not JSON-serialisable — drop before broadcast
            result.pop("_yamnet_result", None)
            self._last_result = {**result, "buffering": False}
        except Exception as exc:
            logger.warning("YAMNet predict error [%s]: %s", self.task_name, exc)
            self._last_result = {**self._last_result, "buffering": False, "error": str(exc)}
        return self._last_result


# ------------------------------------------------------------------ #
# Model store — unchanged loading logic                               #
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
# Serve-wrapper registry — generic dispatch by task type             #
# ------------------------------------------------------------------ #
# A servable task is turned into a wrapper by a registered factory. New
# servable kinds are added by registering a factory, not by editing ModelStore.
# A factory is (task, task_name, cfg) -> wrapper instance or None (None = can't
# serve yet, e.g. an untrained classifier with no checkpoint).

_WRAPPER_FACTORIES: dict = {}


def register_wrapper(wrapper_type: str):
    def deco(fn):
        _WRAPPER_FACTORIES[wrapper_type] = fn
        return fn
    return deco


def resolve_wrapper_type(task, cfg: dict) -> str:
    """Explicit cfg['serve_wrapper'] wins; else infer from task_type/TASK_TYPE."""
    explicit = cfg.get("serve_wrapper")
    if explicit:
        return explicit
    if cfg.get("task_type") == "nonverbal_pretrained":
        return "pretrained_event"
    if getattr(task, "TASK_TYPE", "") == "transcription":
        return "transcription"
    return "classifier"


def _find_local_checkpoint(task_name: str, cfg: dict):
    """Look for a trained checkpoint in the usual places."""
    output_dir = cfg.get("output_dir", os.path.join("storage", task_name))
    candidates = [
        os.path.join(output_dir, "checkpoints", "best_model.pt"),
        os.path.join("storage", task_name, "checkpoints", "best_model.pt"),
        os.path.join(MODELS_DIR, task_name, "best_model.pt"),
        os.path.join(MODELS_DIR, f"{task_name}.pt"),
    ]
    return next((c for c in candidates if os.path.exists(c)), None)


def _download_checkpoint_from_storage(task_name: str, cfg: dict):
    """If object storage is enabled, pull the task's checkpoint locally."""
    if os.environ.get("STORAGE_BACKEND", "local").lower() not in ("minio", "s3"):
        return None
    try:
        from audio_ops_dagster.core.storage import S3StorageBackend, ArtifactType
        backend = S3StorageBackend.from_env()
        if not backend.enabled:
            return None
        dest_dir = os.path.join(MODELS_DIR, task_name)
        return str(backend.download_task_file(
            task_name, ArtifactType.MODELS, "best_model.pt", dest_dir))
    except Exception as exc:
        logger.warning("Storage checkpoint fetch failed for %s: %s", task_name, exc)
        return None


@register_wrapper("classifier")
def _make_classifier_wrapper(task, task_name: str, cfg: dict):
    """Generic: build the model from the TASK's own build_model(), load weights."""
    import torch
    ckpt = _find_local_checkpoint(task_name, cfg) or _download_checkpoint_from_storage(task_name, cfg)
    if not ckpt:
        logger.info("No checkpoint for '%s' yet — train it, then it will serve.", task_name)
        return None
    input_shape = cfg.get("input_shape")
    model = task.build_model(input_shape=input_shape)
    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = loaded.get("model_state_dict", loaded) if isinstance(loaded, dict) else loaded
    if isinstance(state, dict):
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning("State dict mismatch '%s' — missing:%s unexpected:%s",
                           task_name, list(missing)[:3], list(unexpected)[:3])
    model.eval()
    model.class_names = cfg.get("class_labels", getattr(model, "class_names", []))
    model.input_shape = input_shape
    return ModelWrapper(model, task_name)


@register_wrapper("transcription")
def _make_transcription_wrapper(task, task_name: str, cfg: dict):
    buffer_s = float(cfg.get("stream_buffer_seconds", 5.0))
    silence = float(cfg.get("silence_rms", 0.006))
    return STTModelWrapper(task, task_name, buffer_s, silence)


@register_wrapper("pretrained_event")
def _make_pretrained_event_wrapper(task, task_name: str, cfg: dict):
    buffer_s = float(os.environ.get("YAMNET_BUFFER_S", cfg.get("stream_buffer_seconds", 2.0)))
    return YamnetModelWrapper(task, task_name, buffer_s)


class ModelStore:
    def __init__(self):
        self._models: dict = {}

    def load_all(self) -> None:
        self._load_task_models()
        # Local-first: the template ships without MinIO, so checkpoints load
        # from MODELS_DIR by default. Opt into object storage with
        # STORAGE_BACKEND=minio (and the MINIO_* env vars).
        backend = os.environ.get("STORAGE_BACKEND", "local").lower()
        if backend in ("minio", "s3"):
            try:
                self._load_from_minio()
            except Exception as exc:
                logger.warning("MinIO unavailable (%s) — falling back to local dir", exc)
                self._load_from_local()
        else:
            self._load_from_local()
        if not self._models:
            logger.warning("No models loaded — inference will return empty results")
        else:
            logger.info("Models ready: %s", list(self._models))

    def _load_task_models(self) -> None:
        import yaml
        try:
            from audio_ops_dagster.registry import get_task, task_search_dirs
            from audio_ops_dagster.capabilities import resolve_capabilities, SERVE
        except ImportError:
            logger.warning("audio_ops_dagster not importable — task models will not load.")
            return

        # Scan the framework's built-in tasks + any external project task dirs
        # (AUDIO_OPS_TASKS_DIR), plus the legacy TASKS_DIR for back-compat.
        search_dirs = [str(d) for d in task_search_dirs()]
        if TASKS_DIR and os.path.isdir(TASKS_DIR) and TASKS_DIR not in search_dirs:
            search_dirs.append(TASKS_DIR)
        if not search_dirs:
            logger.warning("No task directories found — skipping task model load.")
            return

        seen = set()
        for tasks_dir in search_dirs:
            if not os.path.isdir(tasks_dir):
                continue
            for entry in sorted(os.scandir(tasks_dir), key=lambda e: e.name):
                if not entry.is_dir():
                    continue
                config_path = os.path.join(entry.path, "config.yaml")
                if not os.path.exists(config_path):
                    continue
                try:
                    with open(config_path) as f:
                        cfg = yaml.safe_load(f) or {}
                    task_name = cfg.get("task_name", entry.name)
                    if task_name in seen:
                        continue
                    # Only tasks that declare `serve` are loaded by the API.
                    try:
                        if SERVE not in resolve_capabilities(cfg):
                            continue
                    except Exception:
                        pass  # if capabilities can't resolve, fall through and try
                    seen.add(task_name)
                    config  = type("Cfg", (), {k: v for k, v in cfg.items()})()
                    task    = get_task(task_name, config)

                    wtype   = resolve_wrapper_type(task, cfg)
                    factory = _WRAPPER_FACTORIES.get(wtype)
                    if factory is None:
                        logger.warning("No serve wrapper '%s' for task '%s' — "
                                       "register one with @register_wrapper.", wtype, task_name)
                        continue
                    wrapper = factory(task, task_name, cfg)
                    if wrapper is not None:
                        self._models[task_name] = wrapper
                        logger.info("Serving '%s' via '%s' wrapper.", task_name, wtype)
                except Exception as exc:
                    logger.warning("Could not load task %s: %s", entry.name, exc)

    def _load_from_minio(self) -> None:
        # Unified on the boto3 S3StorageBackend (works with MinIO/S3/R2/B2) so
        # there's one storage client across training and serving.
        import torch
        from audio_ops_dagster.core.storage import S3StorageBackend
        backend = S3StorageBackend.from_env()
        if not backend.enabled:
            return
        backend.ensure_bucket()
        for key in backend.list_objects(""):
            if not key.endswith(".pt"):
                continue
            task = key.split("/")[0]
            if task in self._models:
                continue
            try:
                local = os.path.join(MODELS_DIR, task, os.path.basename(key))
                backend.download_file(key, local)
                loaded = torch.load(local, map_location="cpu", weights_only=False)
                model  = _reconstruct_model(loaded, task)
                model.eval()
                self._models[task] = ModelWrapper(model, task)
                logger.info("Loaded from storage: %s <- %s", task, key)
            except Exception as exc:
                logger.warning("Could not load %s: %s", key, exc)

    def _load_from_local(self) -> None:
        import glob, torch
        for path in glob.glob(os.path.join(MODELS_DIR, "**/*.pt"), recursive=True):
            task = os.path.splitext(os.path.basename(path))[0]
            task = task.replace("_best", "").replace("_model", "").replace("best", "")
            if task in self._models:
                continue
            try:
                loaded = torch.load(path, map_location="cpu", weights_only=False)
                model  = _reconstruct_model(loaded, task)
                model.eval()
                self._models[task] = ModelWrapper(model, task)
                logger.info("Loaded local: %s <- %s", task, path)
            except Exception as exc:
                logger.warning("Could not load %s: %s", path, exc)

    def get(self, task):
        return self._models.get(task)

    def all(self):
        return dict(self._models)

    def names(self):
        return list(self._models)


store:    ModelStore   = ModelStore()
influx:   InfluxWriter = InfluxWriter()
streams:  dict         = {}
_sse_all: list         = []


class StreamHandle:
    def __init__(self, stream_id: str, url: str = "", chunk_ms: int = CHUNK_MS,
                 source_mode: str = "rtsp"):
        self.stream_id     = stream_id
        self.url           = url
        self.chunk_ms      = chunk_ms
        self.chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
        self.active        = True
        self.frames        = 0
        self.errors        = 0
        self.dropped       = 0
        self.last_rms      = 0.0
        self.task          = None
        self.subscribers   = []
        self.source_mode   = source_mode      # 'rtsp' | 'ws' | 'pending'
        # --- the single media clock for this stream ---
        self.media_samples = 0                # total samples consumed
        self.fuser         = (TranscriptFuser(nonverbal_min_conf=NONVERBAL_MIN_CONF)
                              if _FUSION_OK else None)
        self._fusion_tick  = 0
        # --- decoupled processing: ingestion never blocks on inference ---
        self.proc_queue: asyncio.Queue = asyncio.Queue(maxsize=PROC_QUEUE_MAX)
        self.proc_task = None
        # --- speech-to-text runs in its OWN lane so a slow Whisper pass can
        #     never stall or drop the non-verbal / sentiment path ---
        self.stt_queue: asyncio.Queue = asyncio.Queue(maxsize=STT_QUEUE_MAX)
        self.stt_task = None

    @property
    def media_t(self) -> float:
        return self.media_samples / SAMPLE_RATE

    def submit(self, audio: np.ndarray) -> None:
        """
        Hand a chunk to the processor. Advances the media clock at *arrival* and
        tags the chunk with its media position, so timestamps stay aligned to the
        real audio/playhead even if we have to drop chunks to keep up. When the
        queue is full we drop the OLDEST queued chunk (stay live, not behind).
        """
        t0 = self.media_t
        self.media_samples += len(audio)
        item = (t0, audio)
        if self.proc_queue.full():
            try:
                self.proc_queue.get_nowait()
                self.dropped += 1
            except Exception:
                pass
        try:
            self.proc_queue.put_nowait(item)
        except Exception:
            pass

    def ensure_processor(self) -> None:
        if self.proc_task is None or self.proc_task.done():
            self.proc_task = asyncio.create_task(
                _processor_loop(self), name=f"proc_{self.stream_id}")
        if self.stt_task is None or self.stt_task.done():
            self.stt_task = asyncio.create_task(
                _stt_loop(self), name=f"stt_{self.stream_id}")

    def submit_stt(self, t0: float, audio: np.ndarray) -> None:
        """Copy a chunk into the STT lane. STT may drop under load WITHOUT
        affecting non-verbal/sentiment detection."""
        if self.stt_queue.full():
            try:
                self.stt_queue.get_nowait()
            except Exception:
                pass
        try:
            self.stt_queue.put_nowait((t0, audio))
        except Exception:
            pass


async def _processor_loop(handle: "StreamHandle") -> None:
    """Single consumer per stream: pulls queued chunks and runs inference."""
    logger.info("Processor started: %s", handle.stream_id)
    while True:
        try:
            t0, audio = await asyncio.wait_for(handle.proc_queue.get(), timeout=5)
        except asyncio.TimeoutError:
            if not handle.active and handle.proc_queue.empty():
                break
            continue
        except asyncio.CancelledError:
            break
        handle.submit_stt(t0, audio)          # hand a copy to the speech lane
        try:
            await _process_chunk(handle, audio, t0)
        except Exception as exc:
            handle.errors += 1
            logger.debug("Processor error [%s]: %s", handle.stream_id, exc)
    logger.info("Processor stopped: %s (dropped %d)", handle.stream_id, handle.dropped)


# ------------------------------------------------------------------ #
# Shared per-chunk processing — used by BOTH RTSP and WS paths        #
# ------------------------------------------------------------------ #

async def _process_chunk(handle: StreamHandle, audio: np.ndarray, media_t0: float) -> None:
    """
    Run every model on one chunk, write to InfluxDB, feed the fuser, and push
    SSE/WS payloads. `audio` is float32 mono at SAMPLE_RATE. `media_t0` is the
    chunk's position on the media clock (set at submit time).
    """
    rms             = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    handle.last_rms = rms
    handle.frames  += 1
    loop            = asyncio.get_event_loop()

    fuser = handle.fuser
    emitted = False

    # Run the models concurrently. torch/TF release the GIL during inference, so
    # YAMNet (CPU) overlaps Whisper/ResNet (GPU) instead of running back-to-back.
    async def _predict(name, model):
        r = await loop.run_in_executor(None, model.predict, audio, media_t0)
        return name, model, r

    predicted = await asyncio.gather(
        *[_predict(n, m) for n, m in store.all().items()
          if not isinstance(m, STTModelWrapper)], return_exceptions=True)

    for _res in predicted:
        if isinstance(_res, Exception):
            handle.errors += 1
            logger.debug("Inference error [%s]: %s", handle.stream_id, _res)
            continue
        task_name, model, result = _res
        try:
            payload = {
                "timestamp":          time.time(),
                "stream_id":          handle.stream_id,
                "task":               task_name,
                "rms":                rms,
                "media_ts_start":     result.get("media_ts_start", media_t0),
                "media_ts_end":       result.get("media_ts_end", media_t0 + len(audio) / SAMPLE_RATE),
                "media_now":          round(handle.media_t, 3),
                "label":              result.get("label"),
                "confidence":         result.get("confidence"),
                "scores":             result.get("scores", {}),
                "transcription":      result.get("transcription"),
                "word_count":         result.get("word_count"),
                "language":           result.get("language"),
                "words":              result.get("words"),
                "buffering":          result.get("buffering", False),
                "primary_event":      result.get("primary_event"),
                "primary_event_conf": result.get("primary_event_conf"),
                "event_media_ts":     result.get("event_media_ts"),
                "top_classes":        result.get("top_classes", []),
            }

            if isinstance(model, YamnetModelWrapper):
                if not result.get("buffering"):
                    influx.write_nonverbal(
                        task=task_name,
                        primary_event=result.get("primary_event", "unknown"),
                        primary_event_conf=result.get("primary_event_conf", 0.0),
                        curated_scores=result.get("scores", {}),
                        top_classes_json=json.dumps(result.get("top_classes", [])),
                        rms=rms, source=handle.source_mode, stream_id=handle.stream_id,
                        duration_s=result.get("duration_sec", 0.0),
                        inference_time_ms=result.get("inference_time_ms", 0.0),
                        media_ts=result.get("event_media_ts", media_t0),
                    )
                    # Don't gate on the PRIMARY event — speech/silence dominate
                    # almost always, so an applause/cry/pain at 20% would never
                    # show. Instead pick the best INTERESTING group above a lenient
                    # floor and surface that, even when speech is the top class.
                    if fuser is not None:
                        scores = result.get("scores", {}) or {}
                        interesting = {k: v for k, v in scores.items()
                                       if k not in NONVERBAL_BACKGROUND}
                        best_label, best_conf = (max(interesting.items(), key=lambda kv: kv[1])
                                                 if interesting else (None, 0.0))
                        # also let a strong primary (e.g. cough) through directly
                        pe, pec = result.get("primary_event"), result.get("primary_event_conf", 0.0)
                        if pe not in NONVERBAL_BACKGROUND and pec > best_conf:
                            best_label, best_conf = pe, pec
                        if best_label and best_conf >= NONVERBAL_MIN_CONF:
                            fuser.add_nonverbal(
                                label=best_label,
                                t_start=result.get("event_media_ts", result["media_ts_start"]),
                                t_end=result.get("media_ts_end"),
                                confidence=float(best_conf),
                                scores=scores,
                            )

            else:  # spectrogram classifier == sentiment
                influx.write_classifier(
                    task=task_name, label=result.get("label", ""),
                    confidence=result.get("confidence", 0.0),
                    scores=result.get("scores", {}),
                    rms=rms, source=handle.source_mode, stream_id=handle.stream_id,
                    media_ts=result.get("media_ts_start", media_t0))
                if fuser is not None and result.get("label"):
                    fuser.add_sentiment(
                        label=result.get("label"),
                        t_start=result.get("media_ts_start", media_t0),
                        t_end=result.get("media_ts_end", media_t0 + len(audio) / SAMPLE_RATE),
                        confidence=result.get("confidence", 0.0),
                        scores=result.get("scores", {}))

            _broadcast(handle, payload)
            emitted = True
        except Exception as exc:
            handle.errors += 1
            logger.debug("Inference error [%s/%s]: %s", handle.stream_id, task_name, exc)

    # --- emit a rolling fusion event ---
    if fuser is not None and emitted:
        handle._fusion_tick += 1
        if handle._fusion_tick % max(FUSION_EMIT_EVERY, 1) == 0:
            try:
                # window the LIVE event so the payload stays small on long calls
                # (the /timeline and /llm endpoints can still pull wider windows)
                fused = fuser.build_llm_payload(window_seconds=FUSION_EMIT_WINDOW_S)
                seg = fused["segments"]
                primary_sent = (seg[-1]["sentiment"]["label"]
                                if seg and seg[-1].get("sentiment") else "")
                _broadcast(handle, {
                    "timestamp":        time.time(),
                    "stream_id":        handle.stream_id,
                    "task":             "fusion",
                    "media_now":        round(handle.media_t, 3),
                    "fused_transcript": fused["fused_transcript"],
                    "segments":         fused["segments"],
                    "sentiment_track":  fused["sentiment_track"],
                    "nonverbal_track":  fused["nonverbal_track"],
                })
                influx.write_fusion(
                    stream_id=handle.stream_id, media_ts=handle.media_t,
                    transcript=fused["fused_transcript"], n_segments=len(seg),
                    primary_sentiment=primary_sent, n_nonverbal=len(fused["nonverbal_track"]))
            except Exception as exc:
                logger.debug("Fusion emit error [%s]: %s", handle.stream_id, exc)


def _broadcast(handle: StreamHandle, payload: dict) -> None:
    for q in handle.subscribers + _sse_all:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


async def _stt_loop(handle: "StreamHandle") -> None:
    """
    Speech-to-text in its own lane. Whisper is the slow model; running it here
    means a long transcription pass can lag (or drop, via the bounded stt_queue)
    WITHOUT ever stalling or dropping the non-verbal / sentiment path. That is
    what makes non-verbal detection consistent across replays.
    """
    stt_model = next((m for m in store.all().values()
                      if isinstance(m, STTModelWrapper)), None)
    if stt_model is None:
        return
    logger.info("STT lane started: %s", handle.stream_id)
    loop = asyncio.get_event_loop()
    fuser = handle.fuser
    while True:
        try:
            t0, audio = await asyncio.wait_for(handle.stt_queue.get(), timeout=5)
        except asyncio.TimeoutError:
            if not handle.active and handle.stt_queue.empty():
                break
            continue
        except asyncio.CancelledError:
            break
        try:
            result = await loop.run_in_executor(None, stt_model.predict, audio, t0)
            rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
            payload = {
                "timestamp": time.time(), "stream_id": handle.stream_id,
                "task": stt_model.task_name, "rms": rms,
                "media_ts_start": result.get("media_ts_start", t0),
                "media_ts_end": result.get("media_ts_end", t0 + len(audio) / SAMPLE_RATE),
                "media_now": round(handle.media_t, 3),
                "transcription": result.get("transcription"),
                "confidence": result.get("confidence"),
                "word_count": result.get("word_count"),
                "language": result.get("language"),
                "words": result.get("words"),
                "buffering": result.get("buffering", False),
            }
            if not result.get("buffering") and result.get("transcription"):
                buffer_s = getattr(stt_model, "_buffer_seconds", 5.0)
                influx.write_stt(
                    task=stt_model.task_name, transcription=result.get("transcription", ""),
                    confidence=result.get("confidence", 0.0),
                    word_count=result.get("word_count", 0),
                    language=result.get("language", ""),
                    rms=rms, source=handle.source_mode,
                    stream_id=handle.stream_id, duration_s=buffer_s,
                    media_ts=result.get("media_ts_start", t0))
                if result.get("words"):
                    influx.write_stt_words(
                        task=stt_model.task_name, words=result["words"],
                        stream_id=handle.stream_id, language=result.get("language", ""))
                if fuser is not None:
                    if result.get("words"):
                        fuser.add_words(result["words"], language=result.get("language", ""))
                    else:
                        fuser.add_utterance(
                            result.get("transcription", ""),
                            t_start=result.get("media_ts_start", t0),
                            t_end=result.get("media_ts_end", t0 + buffer_s),
                            confidence=result.get("confidence", 0.0),
                            language=result.get("language", ""))
            _broadcast(handle, payload)
        except Exception as exc:
            handle.errors += 1
            logger.debug("STT lane error [%s]: %s", handle.stream_id, exc)
    logger.info("STT lane stopped: %s", handle.stream_id)


# ------------------------------------------------------------------ #
# RTSP worker (server-side file/stream source)                        #
# ------------------------------------------------------------------ #

async def _stream_worker(handle: StreamHandle) -> None:
    chunk_bytes = handle.chunk_samples * 2

    async def _spawn():
        return await asyncio.create_subprocess_exec(
            "ffmpeg", "-loglevel", "error", "-rtsp_transport", "tcp",
            "-i", handle.url, "-vn", "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "pipe:1",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)

    async def _read_exact(stream, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = await stream.read(n - len(buf))
            if not chunk:
                return bytes(buf)
            buf.extend(chunk)
        return bytes(buf)

    logger.info("Worker started: %s -> %s", handle.stream_id, handle.url)
    handle.ensure_processor()
    while handle.active:
        proc = None
        try:
            proc = await _spawn()
            while handle.active:
                raw = await asyncio.wait_for(_read_exact(proc.stdout, chunk_bytes), timeout=10)
                if not raw:
                    break
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                handle.submit(audio)
        except asyncio.CancelledError:
            break
        except asyncio.TimeoutError:
            handle.errors += 1
            logger.warning("Stream timeout: %s", handle.stream_id)
        except Exception as exc:
            handle.errors += 1
            logger.warning("Stream error [%s]: %s", handle.stream_id, exc)
        finally:
            if proc:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
        if handle.active:
            await asyncio.sleep(3)
    logger.info("Worker stopped: %s", handle.stream_id)


# ------------------------------------------------------------------ #
# App lifecycle                                                       #
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load_all()
    for entry in filter(None, os.environ.get("AUTO_RTSP_STREAMS", "").split(",")):
        parts = entry.strip().split("|")
        url   = parts[0]
        sid   = parts[1] if len(parts) > 1 else f"auto_{len(streams)}"
        await _add_stream(sid, url)
    heartbeat = asyncio.create_task(_heartbeat_loop())
    discover  = asyncio.create_task(_autodiscover_loop()) if AUTODISCOVER else None
    yield
    for h in streams.values():
        h.active = False
        if h.task:
            h.task.cancel()
        if h.proc_task:
            h.proc_task.cancel()
        if h.stt_task:
            h.stt_task.cancel()
    heartbeat.cancel()
    if discover:
        discover.cancel()


async def _heartbeat_loop() -> None:
    while True:
        await asyncio.sleep(15)


def _fetch_mediamtx_paths_sync() -> list:
    """Return the names of paths currently publishing to MediaMTX."""
    import urllib.request
    url = MEDIAMTX_API_URL.rstrip("/") + "/v3/paths/list"
    with urllib.request.urlopen(url, timeout=3) as resp:
        data = json.loads(resp.read().decode())
    # 'ready' means a publisher is active on that path
    return [it["name"] for it in data.get("items", []) if it.get("ready", True)]


async def _list_mediamtx_paths() -> list:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_mediamtx_paths_sync)


async def _autodiscover_loop() -> None:
    """Poll MediaMTX; run loaded models on whatever is publishing.

    Generic: the framework never names streams. A path appearing in MediaMTX
    becomes a stream (stream_id = path name); a path disappearing tears its
    stream down. Only auto-added streams are auto-removed — manual /streams
    registrations are left alone.
    """
    pattern = re.compile(AUTODISCOVER_FILTER) if AUTODISCOVER_FILTER else None
    logger.info("Stream auto-discovery ON — polling %s every %ss%s",
                MEDIAMTX_API_URL, AUTODISCOVER_INTERVAL,
                f" (filter: {AUTODISCOVER_FILTER})" if AUTODISCOVER_FILTER else "")
    while True:
        try:
            names = await _list_mediamtx_paths()
            present = set()
            for name in names:
                if pattern and not pattern.search(name):
                    continue
                present.add(name)
                if name not in streams or not streams[name].active:
                    url = f"{RTSP_BASE_URL}/{name}"
                    try:
                        await _add_stream(name, url)
                        _autodiscovered.add(name)
                        logger.info("Auto-discovered stream: %s <- %s", name, url)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Auto-add failed for %s: %s", name, exc)
            # tear down auto-added streams that are no longer publishing
            for sid in list(_autodiscovered):
                if sid not in present and sid in streams:
                    try:
                        await _remove_stream(sid)
                    except Exception:  # noqa: BLE001
                        pass
                    _autodiscovered.discard(sid)
                    logger.info("Auto-removed stream (no longer publishing): %s", sid)
        except Exception as exc:  # noqa: BLE001
            logger.debug("auto-discovery poll error: %s", exc)
        await asyncio.sleep(AUTODISCOVER_INTERVAL)
        for h in streams.values():
            influx.write_stream_health(
                stream_id=h.stream_id,
                active=h.active and (h.source_mode == "ws" or bool(h.task and not h.task.done())),
                frames=h.frames, errors=h.errors, rms=h.last_rms)


app = FastAPI(
    title="Audio Ops Inference", version="4.0.0",
    description="Audio inference over file/RTSP/URL/WebSocket with SSE output (optional fusion via ENABLE_FUSION)",
    lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _resample(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    except Exception:
        return audio


async def _run_all_models(audio: np.ndarray, sr: int, source: str) -> dict:
    """One-shot file/url inference (no streaming clock)."""
    audio      = _resample(audio, sr)
    duration_s = len(audio) / SAMPLE_RATE
    rms        = float(np.sqrt(np.mean(audio ** 2)))
    loop       = asyncio.get_event_loop()
    results    = {}
    for task_name, model in store.all().items():
        try:
            t0     = time.time()
            result = await loop.run_in_executor(None, model.predict, audio, 0.0)
            result.pop("_yamnet_result", None)
            result["latency_ms"] = round((time.time() - t0) * 1000, 1)
            results[task_name]   = result
        except Exception as exc:
            results[task_name] = {"error": str(exc)}
    return {"tasks": results, "duration_s": round(duration_s, 3), "rms": round(rms, 4)}


async def _add_stream(stream_id: str, url: str, chunk_ms: int = CHUNK_MS) -> StreamHandle:
    handle             = StreamHandle(stream_id=stream_id, url=url, chunk_ms=chunk_ms, source_mode="rtsp")
    streams[stream_id] = handle
    handle.task        = asyncio.create_task(_stream_worker(handle), name=f"worker_{stream_id}")
    logger.info("Stream registered: %s -> %s", stream_id, url)
    return handle


async def _remove_stream(stream_id: str) -> None:
    handle = streams.pop(stream_id, None)
    if handle:
        handle.active = False
        if handle.task:
            handle.task.cancel()
        if handle.proc_task:
            handle.proc_task.cancel()
        if handle.stt_task:
            handle.stt_task.cancel()
        logger.info("Stream removed: %s", stream_id)


# ------------------------------------------------------------------ #
# REST endpoints                                                      #
# ------------------------------------------------------------------ #

@app.get("/health")
async def health():
    return {"status": "ok", "models": store.names(), "streams": len(streams),
            "influxdb": influx.ping(), "fusion": _FUSION_OK,
            "llm": bool(_llm_client and _llm_client.config.enabled)}


@app.get("/models")
async def list_models():
    out = []
    for name, m in store.all().items():
        if isinstance(m, YamnetModelWrapper):
            out.append({"task": name, "type": "nonverbal_pretrained",
                        "backend": "tensorflow_hub_yamnet", "num_classes": 521})
        elif isinstance(m, STTModelWrapper):
            out.append({"task": name, "type": "transcription",
                        "model": getattr(getattr(m.task, "_backend", None), "model_name", "?")})
        else:
            out.append({"task": name, "type": "classifier",
                        "classes": m.class_names, "input_shape": m.input_shape})
    return {"models": out}


@app.get("/demo")
async def demo_audio():
    """Serve the demo WAV so the browser can play it (audible) and stream it."""
    if not os.path.exists(DEMO_AUDIO_PATH):
        raise HTTPException(404, f"Demo audio not found at {DEMO_AUDIO_PATH} — "
                                 f"mount /test_audio into the inference container "
                                 f"or set DEMO_AUDIO_PATH")
    return FileResponse(DEMO_AUDIO_PATH, media_type="audio/wav", filename="demo.wav")


@app.post("/infer/file")
async def infer_file(file: UploadFile = File(...)):
    data = await file.read()
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception as exc:
        raise HTTPException(422, f"Cannot decode audio: {exc}")
    return await _run_all_models(audio, sr, source="file")


class AddStreamRequest(BaseModel):
    url:               str
    stream_id:         Optional[str] = None
    chunk_duration_ms: int = Field(CHUNK_MS, ge=100, le=10000)


class BatchStreamItem(BaseModel):
    stream_id:         str
    url:               str
    chunk_duration_ms: int = Field(CHUNK_MS, ge=100, le=10000)


class BatchStreamRequest(BaseModel):
    streams: list[BatchStreamItem]


@app.post("/streams", status_code=201)
async def add_stream(req: AddStreamRequest):
    sid = req.stream_id or f"stream_{int(time.time())}"
    if sid in streams and streams[sid].active:
        raise HTTPException(409, f"Stream '{sid}' already registered")
    await _add_stream(sid, req.url, req.chunk_duration_ms)
    return {"stream_id": sid, "url": req.url, "status": "started"}


@app.delete("/streams/{stream_id}")
async def remove_stream(stream_id: str):
    if stream_id not in streams:
        raise HTTPException(404, f"Stream '{stream_id}' not found")
    await _remove_stream(stream_id)
    return {"stream_id": stream_id, "status": "removed"}


@app.get("/streams")
async def list_streams():
    return {"streams": [
        {"stream_id": sid, "url": h.url, "active": h.active, "mode": h.source_mode,
         "frames": h.frames, "errors": h.errors, "rms": round(h.last_rms, 4),
         "media_t": round(h.media_t, 2)}
        for sid, h in streams.items()]}


@app.post("/streams/batch", status_code=201)
async def add_streams_batch(req: "BatchStreamRequest"):
    """Register many streams at once. Callers (projects) supply the names + URLs;
    the framework stays neutral about what the streams represent."""
    results = {}
    for item in req.streams:
        if item.stream_id in streams and streams[item.stream_id].active:
            results[item.stream_id] = "already running"
        else:
            await _add_stream(item.stream_id, item.url, item.chunk_duration_ms)
            results[item.stream_id] = "started"
    return {"status": "ok", "streams": results}


@app.get("/discovery")
async def discovery_status():
    """Report auto-discovery state and what MediaMTX is currently publishing."""
    paths = []
    if AUTODISCOVER:
        try:
            paths = await _list_mediamtx_paths()
        except Exception as exc:  # noqa: BLE001
            return {"enabled": True, "error": str(exc), "mediamtx_api": MEDIAMTX_API_URL}
    return {
        "enabled": AUTODISCOVER,
        "mediamtx_api": MEDIAMTX_API_URL,
        "interval_s": AUTODISCOVER_INTERVAL,
        "filter": AUTODISCOVER_FILTER or None,
        "publishing_paths": paths,
        "auto_streams": sorted(_autodiscovered),
    }


# ------------------------------------------------------------------ #
# Fusion endpoints                                                    #
# ------------------------------------------------------------------ #

@app.get("/fusion/{stream_id}/timeline")
async def fusion_timeline(stream_id: str, window_s: Optional[float] = None):
    handle = streams.get(stream_id)
    if not handle or handle.fuser is None:
        raise HTTPException(404, f"No fusion timeline for '{stream_id}'")
    return handle.fuser.build_llm_payload(window_seconds=window_s)


class FusionLLMRequest(BaseModel):
    window_s:    float = FUSION_LLM_WINDOW_S
    instruction: Optional[str] = None
    system:      Optional[str] = None


@app.post("/fusion/{stream_id}/llm")
async def fusion_llm(stream_id: str, req: FusionLLMRequest):
    handle = streams.get(stream_id)
    if not handle or handle.fuser is None:
        raise HTTPException(404, f"No fusion timeline for '{stream_id}'")
    if not _llm_client:
        raise HTTPException(503, "LLM client unavailable")
    messages = handle.fuser.build_llm_messages(
        window_seconds=req.window_s, system_prompt=req.system, instruction=req.instruction)
    loop = asyncio.get_event_loop()
    reply = await loop.run_in_executor(None, _llm_client.chat, messages)
    if not reply:
        raise HTTPException(503, "LLM returned empty (is ENABLE_LLM=true and the model up?)")
    return {"stream_id": stream_id, "window_s": req.window_s,
            "summary": reply, "media_t": round(handle.media_t, 2)}


# ------------------------------------------------------------------ #
# WebSocket ingest — browser microphone OR a file the browser plays   #
# ------------------------------------------------------------------ #

@app.websocket("/ws/ingest/{stream_id}")
async def ws_ingest(ws: WebSocket, stream_id: str):
    """
    Protocol
    --------
    Client connects, then streams BINARY messages: little-endian Int16 PCM,
    mono, 16 kHz (any frame size; the server reslices to CHUNK_MS).
    TEXT messages: {"cmd":"reset"} clears the fuser, {"cmd":"close"} ends.
    The server pushes per-task + fusion JSON events back over the SAME socket,
    so the browser clock and the events stay bound together.

    Because the browser captures exactly what it plays/records, the media clock
    derived from received samples equals the browser's playhead — that is the
    alignment guarantee.
    """
    await ws.accept()
    existing = streams.get(stream_id)
    if existing and existing.source_mode == "rtsp" and existing.active:
        # Browser audio takes over: stop the RTSP worker so we never run both
        # an RTSP pull AND a WS push on the same stream at the same time.
        existing.active = False
        if existing.task:
            existing.task.cancel()
        if existing.proc_task:
            existing.proc_task.cancel()
        logger.info("WS takeover: stopped RTSP worker for %s", stream_id)

    handle = streams.get(stream_id)
    if handle is None or handle.source_mode in ("pending", "rtsp"):
        prev_subs = handle.subscribers if handle else []
        handle = StreamHandle(stream_id=stream_id, url="ws://browser", source_mode="ws")
        handle.subscribers = prev_subs            # keep any SSE viewers attached
        streams[stream_id] = handle
    handle.source_mode = "ws"
    handle.active = True
    if handle.fuser:
        handle.fuser.reset()
    handle.media_samples = 0
    handle.frames = 0
    # fresh processing queues + consumers for this session
    handle.proc_queue = asyncio.Queue(maxsize=PROC_QUEUE_MAX)
    handle.stt_queue = asyncio.Queue(maxsize=STT_QUEUE_MAX)
    handle.ensure_processor()

    q: asyncio.Queue = asyncio.Queue(maxsize=400)
    handle.subscribers.append(q)

    await ws.send_json({"event": "connected", "stream_id": stream_id, "mode": "ws",
                        "sample_rate": SAMPLE_RATE, "expected_chunk_ms": CHUNK_MS})

    async def _pump_events():
        # Resilient: a single transient send error must not silently stop the
        # event stream while audio keeps arriving. Only a real disconnect ends it.
        while True:
            try:
                payload = await q.get()
            except asyncio.CancelledError:
                break
            try:
                await ws.send_json(payload)
            except (WebSocketDisconnect, RuntimeError):
                break
            except asyncio.CancelledError:
                break
            except Exception:
                continue

    pump = asyncio.create_task(_pump_events())
    leftover = np.empty(0, dtype=np.float32)
    chunk_samples = handle.chunk_samples

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                try:
                    cmd = json.loads(msg["text"]).get("cmd")
                except Exception:
                    cmd = None
                if cmd == "reset":
                    if handle.fuser:
                        handle.fuser.reset()
                    handle.media_samples = 0
                    leftover = np.empty(0, dtype=np.float32)
                    while not handle.proc_queue.empty():
                        try:
                            handle.proc_queue.get_nowait()
                        except Exception:
                            break
                    while not handle.stt_queue.empty():
                        try:
                            handle.stt_queue.get_nowait()
                        except Exception:
                            break
                    await ws.send_json({"event": "reset", "stream_id": stream_id})
                elif cmd == "close":
                    break
                continue

            data = msg.get("bytes")
            if not data:
                continue
            pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            leftover = np.concatenate([leftover, pcm]) if leftover.size else pcm
            while leftover.size >= chunk_samples:
                block, leftover = leftover[:chunk_samples], leftover[chunk_samples:]
                handle.submit(block)        # non-blocking; processor consumes it
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("WS ingest error [%s]: %s", stream_id, exc)
    finally:
        pump.cancel()
        try:
            handle.subscribers.remove(q)
        except ValueError:
            pass
        handle.active = False     # keep handle+fuser for /fusion endpoints
        logger.info("WS ingest closed: %s", stream_id)


# ------------------------------------------------------------------ #
# SSE endpoints                                                       #
# ------------------------------------------------------------------ #

@app.get("/streams/{stream_id}/events")
async def stream_events(stream_id: str):
    handle = streams.get(stream_id)
    if not handle:
        # Allow subscribing before the stream exists (e.g. WS not open yet)
        handle = StreamHandle(stream_id=stream_id, source_mode="pending")
        handle.active = False
        streams[stream_id] = handle
    q: asyncio.Queue = asyncio.Queue(maxsize=400)
    handle.subscribers.append(q)

    async def gen():
        yield f"data: {json.dumps({'event': 'connected', 'stream_id': stream_id})}\n\n"
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=30)
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            try:
                handle.subscribers.remove(q)
            except ValueError:
                pass
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/events")
async def all_events():
    q: asyncio.Queue = asyncio.Queue(maxsize=500)
    _sse_all.append(q)

    async def gen():
        yield f"data: {json.dumps({'event': 'connected'})}\n\n"
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=30)
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            try:
                _sse_all.remove(q)
            except ValueError:
                pass
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})