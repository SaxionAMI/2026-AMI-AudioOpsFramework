"""
Audio Ops — streaming inference service
=======================================

A FastAPI daemon that serves every task declaring the ``serve`` capability
over several transports:

* ``POST /infer/file``           score an uploaded clip (segmented like training)
* ``POST /streams``              pull an RTSP/HTTP stream through ffmpeg
* ``WS   /ws/ingest/{stream}``   push 16 kHz mono PCM16 frames (browser mic/file)
* ``GET  /streams/{id}/events``  per-stream results as Server-Sent Events
* ``GET  /events``               all streams' results as SSE
* ``/fusion/...``                optional transcript+event+sentiment timeline
                                 (``ENABLE_FUSION=true``)

Single media clock
------------------
Every chunk carries ``media_t0`` (seconds since stream start) and every result
is stamped with ``media_ts_start``/``media_ts_end`` on that clock, so events
line up with the audio regardless of inference latency.

Isolation
---------
Each stream gets its own wrapper instances (shared model weights, private
buffers), so audio from one stream never leaks into another stream's
transcription or event window.

Security
--------
Set ``API_KEY`` to require ``X-API-Key: <key>`` on every endpoint except
``/health`` (browsers' EventSource/WebSocket may pass ``?api_key=<key>``).
``CORS_ALLOW_ORIGINS`` restricts cross-origin callers (default ``*``).
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np

from audio_ops_inference.settings import Settings, configure_model_caches

settings = Settings.from_env()
# Must happen before torch / transformers / tensorflow are imported.
configure_model_caches(settings.model_cache_dir)

import soundfile as sf  # noqa: E402
from fastapi import (  # noqa: E402
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from audio_ops_dagster import __version__  # noqa: E402
from audio_ops_dagster._log import get_logger  # noqa: E402
from audio_ops_inference.influx import InfluxWriter  # noqa: E402
from audio_ops_inference.store import ModelStore  # noqa: E402
from audio_ops_inference.wrappers import (  # noqa: E402,F401 — re-exported for back-compat
    SAMPLE_RATE,
    ModelWrapper,
    PretrainedEventWrapper,
    ServeWrapper,
    STTModelWrapper,
    YamnetModelWrapper,
    register_wrapper,
    resolve_wrapper_type,
)

logger = get_logger("audio_ops_inference.api")

# --- Fusion layer (OPTIONAL, off by default; enable with ENABLE_FUSION=true) --
TranscriptFuser = None
_FUSION_OK = False
if settings.fusion_enabled:
    try:
        from audio_ops_dagster.extras.fusion import TranscriptFuser  # type: ignore[no-redef]

        _FUSION_OK = True
    except Exception as _exc:  # pragma: no cover  # noqa: BLE001
        logger.warning("Fusion enabled but unavailable (%s) — running without fusion", _exc)

# --- LLM client (optional, opt-in via ENABLE_LLM) --------------------------
try:
    from audio_ops_dagster.core.llm_client import LLMClient

    _llm_client: Optional[LLMClient] = LLMClient.from_env()
except Exception as _exc:  # pragma: no cover  # noqa: BLE001
    logger.warning("LLMClient unavailable (%s)", _exc)
    _llm_client = None

CHUNK_MS = settings.chunk_ms

store: ModelStore = ModelStore()
influx: InfluxWriter = InfluxWriter(
    enabled=settings.influx_enabled, url=settings.influx_url, token=settings.influx_token,
    org=settings.influx_org, bucket=settings.influx_bucket,
)
streams: Dict[str, StreamHandle] = {}
_sse_all: List[asyncio.Queue] = []
_autodiscovered: set = set()


# ------------------------------------------------------------------ #
# Stream state                                                        #
# ------------------------------------------------------------------ #


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
        self.task: Optional[asyncio.Task] = None
        self.subscribers: List[asyncio.Queue] = []
        self.source_mode   = source_mode      # 'rtsp' | 'ws' | 'pending'
        self.created_at    = time.time()
        # Per-stream wrapper instances: shared weights, private buffers.
        self.models: Dict[str, ServeWrapper] = store.for_stream()
        # --- the single media clock for this stream ---
        self.media_samples = 0
        self.fuser = (TranscriptFuser(nonverbal_min_conf=settings.nonverbal_min_conf)
                      if _FUSION_OK and TranscriptFuser is not None else None)
        self._fusion_tick  = 0
        # --- decoupled processing: ingestion never blocks on inference ---
        self.proc_queue: asyncio.Queue = asyncio.Queue(maxsize=settings.proc_queue_max)
        self.proc_task: Optional[asyncio.Task] = None
        # --- speech-to-text runs in its OWN lane so a slow pass can never
        #     stall or drop the event / classifier path ---
        self.stt_queue: asyncio.Queue = asyncio.Queue(maxsize=settings.stt_queue_max)
        self.stt_task: Optional[asyncio.Task] = None

    @property
    def media_t(self) -> float:
        return self.media_samples / SAMPLE_RATE

    def fast_models(self) -> Dict[str, ServeWrapper]:
        return {n: m for n, m in self.models.items() if m.kind != "transcription"}

    def stt_models(self) -> Dict[str, ServeWrapper]:
        return {n: m for n, m in self.models.items() if m.kind == "transcription"}

    def reset_models(self) -> None:
        self.models = store.for_stream()

    def submit(self, audio: np.ndarray) -> None:
        """
        Hand a chunk to the processor. Advances the media clock at *arrival* and
        tags the chunk with its media position, so timestamps stay aligned to the
        real audio even if chunks are dropped to keep up. When the queue is full
        the OLDEST queued chunk is dropped (stay live, not behind).
        """
        t0 = self.media_t
        self.media_samples += len(audio)
        if self.proc_queue.full():
            try:
                self.proc_queue.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:
                pass
        try:
            self.proc_queue.put_nowait((t0, audio))
        except asyncio.QueueFull:
            self.dropped += 1

    def ensure_processor(self) -> None:
        if self.proc_task is None or self.proc_task.done():
            self.proc_task = asyncio.create_task(
                _processor_loop(self), name=f"proc_{self.stream_id}")
        if self.stt_models() and (self.stt_task is None or self.stt_task.done()):
            self.stt_task = asyncio.create_task(
                _stt_loop(self), name=f"stt_{self.stream_id}")

    def submit_stt(self, t0: float, audio: np.ndarray) -> None:
        """Copy a chunk into the STT lane (which may drop under load on its own)."""
        if not self.stt_models():
            return
        if self.stt_queue.full():
            try:
                self.stt_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self.stt_queue.put_nowait((t0, audio))
        except asyncio.QueueFull:
            pass

    def cancel_tasks(self) -> None:
        for t in (self.task, self.proc_task, self.stt_task):
            if t is not None:
                t.cancel()

    def summary(self) -> dict:
        return {"stream_id": self.stream_id, "url": self.url, "active": self.active,
                "mode": self.source_mode, "frames": self.frames, "errors": self.errors,
                "dropped": self.dropped, "rms": round(self.last_rms, 4),
                "media_t": round(self.media_t, 2), "models": sorted(self.models)}


async def _processor_loop(handle: StreamHandle) -> None:
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
        handle.submit_stt(t0, audio)
        try:
            await _process_chunk(handle, audio, t0)
        except Exception as exc:  # noqa: BLE001
            handle.errors += 1
            logger.debug("Processor error [%s]: %s", handle.stream_id, exc)
    logger.info("Processor stopped: %s (dropped %d)", handle.stream_id, handle.dropped)


# ------------------------------------------------------------------ #
# Shared per-chunk processing — used by BOTH RTSP and WS paths        #
# ------------------------------------------------------------------ #


def _event_payload(handle: StreamHandle, task_name: str, result: dict, rms: float,
                   media_t0: float, n_samples: int) -> dict:
    payload = {
        "timestamp":      time.time(),
        "stream_id":      handle.stream_id,
        "task":           task_name,
        "rms":            rms,
        "media_ts_start": result.get("media_ts_start", media_t0),
        "media_ts_end":   result.get("media_ts_end", media_t0 + n_samples / SAMPLE_RATE),
        "media_now":      round(handle.media_t, 3),
        "buffering":      result.get("buffering", False),
    }
    for key in ("label", "confidence", "scores", "value", "labels", "transcription",
                "word_count", "language", "words", "primary_event", "primary_event_conf",
                "event_media_ts", "top_classes", "error"):
        if key in result:
            payload[key] = result[key]
    return payload


async def _process_chunk(handle: StreamHandle, audio: np.ndarray, media_t0: float) -> None:
    """
    Run every non-STT model on one chunk, write metrics, feed the fuser, and
    push SSE/WS payloads. ``audio`` is float32 mono at SAMPLE_RATE.
    """
    rms             = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    handle.last_rms = rms
    handle.frames  += 1
    loop            = asyncio.get_running_loop()
    fuser           = handle.fuser
    emitted         = False

    # Run the models concurrently; torch/TF release the GIL during inference.
    async def _predict(name, model):
        r = await loop.run_in_executor(None, model.predict, audio, media_t0)
        return name, model, r

    predicted = await asyncio.gather(
        *[_predict(n, m) for n, m in handle.fast_models().items()], return_exceptions=True)

    for res in predicted:
        if isinstance(res, BaseException):
            handle.errors += 1
            logger.debug("Inference error [%s]: %s", handle.stream_id, res)
            continue
        task_name, model, result = res
        try:
            payload = _event_payload(handle, task_name, result, rms, media_t0, len(audio))

            if model.kind == "event":
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
                    # Surface the best *interesting* group above a lenient floor,
                    # even when speech/silence dominate the primary class.
                    if fuser is not None:
                        scores = result.get("scores", {}) or {}
                        background = set(settings.nonverbal_background)
                        interesting = {k: v for k, v in scores.items() if k not in background}
                        best_label, best_conf = (max(interesting.items(), key=lambda kv: kv[1])
                                                 if interesting else (None, 0.0))
                        pe, pec = result.get("primary_event"), result.get("primary_event_conf", 0.0)
                        if pe not in background and pec > best_conf:
                            best_label, best_conf = pe, pec
                        if best_label and best_conf >= settings.nonverbal_min_conf:
                            fuser.add_nonverbal(
                                label=best_label,
                                t_start=result.get("event_media_ts", result["media_ts_start"]),
                                t_end=result.get("media_ts_end"),
                                confidence=float(best_conf),
                                scores=scores,
                            )
            else:
                influx.write_classifier(
                    task=task_name, label=result.get("label", result.get("value", "")),
                    confidence=result.get("confidence", 0.0),
                    scores=result.get("scores", {}),
                    rms=rms, source=handle.source_mode, stream_id=handle.stream_id,
                    duration_s=len(audio) / SAMPLE_RATE,
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
        except Exception as exc:  # noqa: BLE001
            handle.errors += 1
            logger.debug("Inference error [%s/%s]: %s", handle.stream_id, task_name, exc)

    # --- emit a rolling fusion event ---
    if fuser is not None and emitted:
        handle._fusion_tick += 1
        if handle._fusion_tick % max(settings.fusion_emit_every, 1) == 0:
            try:
                fused = fuser.build_llm_payload(window_seconds=settings.fusion_emit_window_s)
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
            except Exception as exc:  # noqa: BLE001
                logger.debug("Fusion emit error [%s]: %s", handle.stream_id, exc)


def _broadcast(handle: StreamHandle, payload: dict) -> None:
    for q in handle.subscribers + _sse_all:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


async def _stt_loop(handle: StreamHandle) -> None:
    """
    Speech-to-text in its own lane. A long transcription pass can lag (or drop,
    via the bounded stt_queue) WITHOUT stalling the event / classifier path.
    """
    if not handle.stt_models():
        return
    logger.info("STT lane started: %s", handle.stream_id)
    loop = asyncio.get_running_loop()
    while True:
        try:
            t0, audio = await asyncio.wait_for(handle.stt_queue.get(), timeout=5)
        except asyncio.TimeoutError:
            if not handle.active and handle.stt_queue.empty():
                break
            continue
        except asyncio.CancelledError:
            break
        rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
        for task_name, stt_model in handle.stt_models().items():
            try:
                result = await loop.run_in_executor(None, stt_model.predict, audio, t0)
                payload = _event_payload(handle, task_name, result, rms, t0, len(audio))
                if not result.get("buffering") and result.get("transcription"):
                    buffer_s = getattr(stt_model, "_buffer_seconds", 5.0)
                    influx.write_stt(
                        task=task_name, transcription=result.get("transcription", ""),
                        confidence=result.get("confidence", 0.0),
                        word_count=result.get("word_count", 0),
                        language=result.get("language", ""),
                        rms=rms, source=handle.source_mode,
                        stream_id=handle.stream_id, duration_s=buffer_s,
                        media_ts=result.get("media_ts_start", t0))
                    if result.get("words"):
                        influx.write_stt_words(task=task_name, words=result["words"],
                                               stream_id=handle.stream_id,
                                               language=result.get("language", ""))
                    if handle.fuser is not None:
                        if result.get("words"):
                            handle.fuser.add_words(result["words"],
                                                   language=result.get("language", ""))
                        else:
                            handle.fuser.add_utterance(
                                result.get("transcription", ""),
                                t_start=result.get("media_ts_start", t0),
                                t_end=result.get("media_ts_end", t0 + buffer_s),
                                confidence=result.get("confidence", 0.0),
                                language=result.get("language", ""))
                _broadcast(handle, payload)
            except Exception as exc:  # noqa: BLE001
                handle.errors += 1
                logger.debug("STT lane error [%s/%s]: %s", handle.stream_id, task_name, exc)
    logger.info("STT lane stopped: %s", handle.stream_id)


# ------------------------------------------------------------------ #
# RTSP worker (server-side file/stream source)                        #
# ------------------------------------------------------------------ #


async def _stream_worker(handle: StreamHandle) -> None:
    chunk_bytes = handle.chunk_samples * 2

    async def _spawn():
        args = ["ffmpeg", "-loglevel", "error"]
        if handle.url.startswith("rtsp"):
            args += ["-rtsp_transport", "tcp"]
        args += ["-i", handle.url, "-vn", "-acodec", "pcm_s16le",
                 "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "pipe:1"]
        return await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)

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
        except FileNotFoundError:
            handle.errors += 1
            logger.error("ffmpeg not found on PATH — cannot pull %s", handle.url)
            handle.active = False
        except Exception as exc:  # noqa: BLE001
            handle.errors += 1
            logger.warning("Stream error [%s]: %s", handle.stream_id, exc)
        finally:
            if proc:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:  # noqa: BLE001
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
    for entry in settings.auto_rtsp_streams:
        parts = entry.split("|")
        url = parts[0]
        sid = parts[1] if len(parts) > 1 else f"auto_{len(streams)}"
        await _add_stream(sid, url)
    heartbeat = asyncio.create_task(_heartbeat_loop())
    discover = asyncio.create_task(_autodiscover_loop()) if settings.autodiscover else None
    yield
    for h in list(streams.values()):
        h.active = False
        h.cancel_tasks()
    heartbeat.cancel()
    if discover:
        discover.cancel()


async def _heartbeat_loop() -> None:
    """Periodic stream-health metrics (no-op unless InfluxDB is enabled)."""
    while True:
        await asyncio.sleep(15)
        for h in list(streams.values()):
            influx.write_stream_health(
                stream_id=h.stream_id,
                active=h.active and (h.source_mode == "ws" or bool(h.task and not h.task.done())),
                frames=h.frames, errors=h.errors, rms=h.last_rms)


def _fetch_mediamtx_paths_sync() -> list:
    """Names of paths currently publishing to MediaMTX."""
    import urllib.request

    url = settings.mediamtx_api_url.rstrip("/") + "/v3/paths/list"
    with urllib.request.urlopen(url, timeout=3) as resp:  # noqa: S310 — configured URL
        data = json.loads(resp.read().decode())
    return [it["name"] for it in data.get("items", []) if it.get("ready", True)]


async def _list_mediamtx_paths() -> list:
    return await asyncio.get_running_loop().run_in_executor(None, _fetch_mediamtx_paths_sync)


async def _autodiscover_loop() -> None:
    """Poll MediaMTX; run the loaded models on whatever is publishing.

    The framework never names streams: a path appearing in MediaMTX becomes a
    stream (stream_id = path name); a path disappearing tears it down. Only
    auto-added streams are auto-removed.
    """
    pattern = re.compile(settings.autodiscover_filter) if settings.autodiscover_filter else None
    logger.info("Stream auto-discovery ON — polling %s every %ss%s",
                settings.mediamtx_api_url, settings.autodiscover_interval,
                f" (filter: {settings.autodiscover_filter})" if pattern else "")
    while True:
        try:
            present = set()
            for name in await _list_mediamtx_paths():
                if pattern and not pattern.search(name):
                    continue
                present.add(name)
                if name not in streams or not streams[name].active:
                    url = f"{settings.rtsp_base_url}/{name}"
                    try:
                        await _add_stream(name, url)
                        _autodiscovered.add(name)
                        logger.info("Auto-discovered stream: %s <- %s", name, url)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Auto-add failed for %s: %s", name, exc)
            for sid in list(_autodiscovered):
                if sid not in present and sid in streams:
                    await _remove_stream(sid)
                    _autodiscovered.discard(sid)
                    logger.info("Auto-removed stream (no longer publishing): %s", sid)
        except Exception as exc:  # noqa: BLE001
            logger.debug("auto-discovery poll error: %s", exc)
        await asyncio.sleep(settings.autodiscover_interval)


app = FastAPI(
    title="Audio Ops Inference",
    version=__version__,
    description="Audio inference over file / RTSP / WebSocket with SSE output "
                "(optional fusion timeline via ENABLE_FUSION).",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_allow_origins,
                   allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ #
# Auth                                                                #
# ------------------------------------------------------------------ #


def _key_ok(provided: Optional[str]) -> bool:
    import hmac

    return bool(provided) and hmac.compare_digest(str(provided), str(settings.api_key))


async def require_api_key(request: Request) -> None:
    """Enforce ``API_KEY`` when it is set (header ``X-API-Key`` or ``?api_key=``)."""
    if not settings.api_key:
        return
    provided = request.headers.get("x-api-key") or request.query_params.get("api_key")
    if not _key_ok(provided):
        raise HTTPException(401, "Missing or invalid API key (X-API-Key header).")


protected = [Depends(require_api_key)]


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _resample(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return audio
    import librosa

    return librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)


async def _run_all_models(audio: np.ndarray, sr: int, source: str,
                          tasks: Optional[List[str]] = None) -> dict:
    """One-shot clip inference (no streaming clock)."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = _resample(audio.astype(np.float32), sr)
    duration_s = len(audio) / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    loop = asyncio.get_running_loop()
    results = {}
    for task_name, model in store.all().items():
        if tasks and task_name not in tasks:
            continue
        try:
            t0 = time.time()
            result = await loop.run_in_executor(None, model.predict_clip, audio)
            result["latency_ms"] = round((time.time() - t0) * 1000, 1)
            results[task_name] = result
        except Exception as exc:  # noqa: BLE001
            logger.warning("Clip inference failed for %s: %s", task_name, exc)
            results[task_name] = {"error": str(exc)}
    return {"tasks": results, "duration_s": round(duration_s, 3), "rms": round(rms, 4),
            "source": source}


async def _add_stream(stream_id: str, url: str, chunk_ms: int = CHUNK_MS) -> StreamHandle:
    handle = StreamHandle(stream_id=stream_id, url=url, chunk_ms=chunk_ms, source_mode="rtsp")
    old = streams.get(stream_id)
    if old is not None:
        handle.subscribers = old.subscribers  # keep SSE viewers attached
        old.active = False
        old.cancel_tasks()
    streams[stream_id] = handle
    handle.task = asyncio.create_task(_stream_worker(handle), name=f"worker_{stream_id}")
    logger.info("Stream registered: %s -> %s", stream_id, url)
    return handle


async def _remove_stream(stream_id: str) -> None:
    handle = streams.pop(stream_id, None)
    if handle:
        handle.active = False
        handle.cancel_tasks()
        logger.info("Stream removed: %s", stream_id)


# ------------------------------------------------------------------ #
# REST endpoints                                                      #
# ------------------------------------------------------------------ #


@app.get("/health")
async def health():
    return {"status": "ok", "version": __version__, "models": store.names(),
            "streams": len(streams), "influxdb": influx.ping(), "fusion": _FUSION_OK,
            "llm": bool(_llm_client and _llm_client.config.enabled),
            "auth": bool(settings.api_key)}


@app.get("/models", dependencies=protected)
async def list_models():
    return {"models": [m.describe() for m in store.all().values()],
            "not_loaded": store.skipped()}


@app.post("/models/reload", dependencies=protected)
async def reload_models():
    """Re-discover tasks and reload checkpoints (e.g. after training finished).

    Streams started after the reload use the new models; running streams keep
    the models they started with until they reconnect.
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, store.load_all)
    return {"models": store.names(), "not_loaded": store.skipped()}


@app.get("/demo", dependencies=protected)
async def demo_audio():
    """Serve the demo WAV so a browser can play it and stream it for analysis."""
    import os

    if not os.path.exists(settings.demo_audio_path):
        raise HTTPException(404, f"Demo audio not found at {settings.demo_audio_path} — "
                                 f"run `audio-ops demo-data` or set DEMO_AUDIO_PATH.")
    return FileResponse(settings.demo_audio_path, media_type="audio/wav", filename="demo.wav")


@app.post("/infer/file", dependencies=protected)
async def infer_file(file: UploadFile = File(...), tasks: Optional[str] = None):
    """Score an audio file with every loaded model (or ``?tasks=a,b``).

    Model tasks return per-segment results (segmented exactly like training)
    plus an aggregate; transcription/event tasks process the whole clip.
    """
    limit = int(settings.max_upload_mb * 1024 * 1024)
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File larger than MAX_UPLOAD_MB={settings.max_upload_mb:g}.")
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"Cannot decode audio: {exc}") from None
    wanted = [t.strip() for t in tasks.split(",")] if tasks else None
    return await _run_all_models(audio, sr, source="file", tasks=wanted)


class AddStreamRequest(BaseModel):
    url:               str
    stream_id:         Optional[str] = None
    chunk_duration_ms: int = Field(CHUNK_MS, ge=100, le=10000)


class BatchStreamItem(BaseModel):
    stream_id:         str
    url:               str
    chunk_duration_ms: int = Field(CHUNK_MS, ge=100, le=10000)


class BatchStreamRequest(BaseModel):
    streams: List[BatchStreamItem]


@app.post("/streams", status_code=201, dependencies=protected)
async def add_stream(req: AddStreamRequest):
    sid = req.stream_id or f"stream_{int(time.time() * 1000)}"
    if sid in streams and streams[sid].active:
        raise HTTPException(409, f"Stream '{sid}' already registered")
    await _add_stream(sid, req.url, req.chunk_duration_ms)
    return {"stream_id": sid, "url": req.url, "status": "started"}


@app.delete("/streams/{stream_id}", dependencies=protected)
async def remove_stream(stream_id: str):
    if stream_id not in streams:
        raise HTTPException(404, f"Stream '{stream_id}' not found")
    await _remove_stream(stream_id)
    return {"stream_id": stream_id, "status": "removed"}


@app.get("/streams", dependencies=protected)
async def list_streams():
    return {"streams": [h.summary() for h in streams.values()]}


@app.post("/streams/batch", status_code=201, dependencies=protected)
async def add_streams_batch(req: BatchStreamRequest):
    """Register many streams at once (callers choose names + URLs)."""
    results = {}
    for item in req.streams:
        if item.stream_id in streams and streams[item.stream_id].active:
            results[item.stream_id] = "already running"
        else:
            await _add_stream(item.stream_id, item.url, item.chunk_duration_ms)
            results[item.stream_id] = "started"
    return {"status": "ok", "streams": results}


@app.get("/discovery", dependencies=protected)
async def discovery_status():
    """Auto-discovery state and what MediaMTX is currently publishing."""
    paths: list = []
    if settings.autodiscover:
        try:
            paths = await _list_mediamtx_paths()
        except Exception as exc:  # noqa: BLE001
            return {"enabled": True, "error": str(exc), "mediamtx_api": settings.mediamtx_api_url}
    return {
        "enabled": settings.autodiscover,
        "mediamtx_api": settings.mediamtx_api_url,
        "interval_s": settings.autodiscover_interval,
        "filter": settings.autodiscover_filter or None,
        "publishing_paths": paths,
        "auto_streams": sorted(_autodiscovered),
    }


# ------------------------------------------------------------------ #
# Fusion endpoints                                                    #
# ------------------------------------------------------------------ #


@app.get("/fusion/{stream_id}/timeline", dependencies=protected)
async def fusion_timeline(stream_id: str, window_s: Optional[float] = None):
    handle = streams.get(stream_id)
    if not handle or handle.fuser is None:
        raise HTTPException(404, f"No fusion timeline for '{stream_id}' (ENABLE_FUSION=true?)")
    return handle.fuser.build_llm_payload(window_seconds=window_s)


class FusionLLMRequest(BaseModel):
    window_s:    float = settings.fusion_llm_window_s
    instruction: Optional[str] = None
    system:      Optional[str] = None


@app.post("/fusion/{stream_id}/llm", dependencies=protected)
async def fusion_llm(stream_id: str, req: FusionLLMRequest):
    handle = streams.get(stream_id)
    if not handle or handle.fuser is None:
        raise HTTPException(404, f"No fusion timeline for '{stream_id}' (ENABLE_FUSION=true?)")
    if not _llm_client or not _llm_client.config.enabled:
        raise HTTPException(503, "LLM client disabled (set ENABLE_LLM=true and LLM_*).")
    messages = handle.fuser.build_llm_messages(
        window_seconds=req.window_s, system_prompt=req.system, instruction=req.instruction)
    reply = await asyncio.get_running_loop().run_in_executor(None, _llm_client.chat, messages)
    if not reply:
        raise HTTPException(503, "LLM returned an empty reply (is the model server up?)")
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
    Connect (with ``?api_key=`` if the server requires a key), then stream
    BINARY messages: little-endian Int16 PCM, mono, 16 kHz (any frame size; the
    server reslices to CHUNK_MS). TEXT messages: ``{"cmd":"reset"}`` clears the
    timeline, ``{"cmd":"close"}`` ends. The server pushes per-task + fusion JSON
    events back over the SAME socket, so the browser clock and the events stay
    bound together.
    """
    if settings.api_key and not _key_ok(ws.query_params.get("api_key")
                                        or ws.headers.get("x-api-key")):
        await ws.close(code=4401)
        return
    await ws.accept()
    existing = streams.get(stream_id)
    if existing and existing.source_mode == "rtsp" and existing.active:
        # Browser audio takes over: never run an RTSP pull AND a WS push at once.
        existing.active = False
        existing.cancel_tasks()
        logger.info("WS takeover: stopped RTSP worker for %s", stream_id)

    handle = streams.get(stream_id)
    if handle is None or handle.source_mode in ("pending", "rtsp"):
        prev_subs = handle.subscribers if handle else []
        handle = StreamHandle(stream_id=stream_id, url="ws://client", source_mode="ws")
        handle.subscribers = prev_subs            # keep any SSE viewers attached
        streams[stream_id] = handle
    handle.source_mode = "ws"
    handle.active = True
    handle.reset_models()
    if handle.fuser:
        handle.fuser.reset()
    handle.media_samples = 0
    handle.frames = 0
    handle.proc_queue = asyncio.Queue(maxsize=settings.proc_queue_max)
    handle.stt_queue = asyncio.Queue(maxsize=settings.stt_queue_max)
    handle.ensure_processor()

    q: asyncio.Queue = asyncio.Queue(maxsize=400)
    handle.subscribers.append(q)

    await ws.send_json({"event": "connected", "stream_id": stream_id, "mode": "ws",
                        "sample_rate": SAMPLE_RATE, "expected_chunk_ms": CHUNK_MS,
                        "models": sorted(handle.models)})

    async def _pump_events():
        # A transient send error must not stop the event stream; only a real
        # disconnect ends it.
        while True:
            try:
                payload = await q.get()
            except asyncio.CancelledError:
                break
            try:
                await ws.send_json(payload)
            except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
                break
            except Exception:  # noqa: BLE001
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
                except Exception:  # noqa: BLE001
                    cmd = None
                if cmd == "reset":
                    if handle.fuser:
                        handle.fuser.reset()
                    handle.reset_models()
                    handle.media_samples = 0
                    leftover = np.empty(0, dtype=np.float32)
                    for queue in (handle.proc_queue, handle.stt_queue):
                        while not queue.empty():
                            try:
                                queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                    await ws.send_json({"event": "reset", "stream_id": stream_id})
                elif cmd == "close":
                    break
                continue

            data = msg.get("bytes")
            if not data:
                continue
            if len(data) % 2:
                data = data[:-1]   # ignore a dangling odd byte rather than crashing
            pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
            leftover = np.concatenate([leftover, pcm]) if leftover.size else pcm
            while leftover.size >= chunk_samples:
                block, leftover = leftover[:chunk_samples], leftover[chunk_samples:]
                handle.submit(block)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
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


def _sse(q: asyncio.Queue, hello: dict, on_close):
    async def gen():
        yield f"data: {json.dumps(hello)}\n\n"
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=30)
                    yield f"data: {json.dumps(payload, default=str)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            on_close()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/streams/{stream_id}/events", dependencies=protected)
async def stream_events(stream_id: str):
    handle = streams.get(stream_id)
    if not handle:
        # Allow subscribing before the stream exists (e.g. WS not open yet).
        handle = StreamHandle(stream_id=stream_id, source_mode="pending")
        handle.active = False
        streams[stream_id] = handle
    q: asyncio.Queue = asyncio.Queue(maxsize=400)
    handle.subscribers.append(q)

    def _close():
        try:
            handle.subscribers.remove(q)
        except ValueError:
            pass
        # Drop placeholder handles nobody listens to anymore.
        if handle.source_mode == "pending" and not handle.subscribers:
            streams.pop(stream_id, None)

    return _sse(q, {"event": "connected", "stream_id": stream_id}, _close)


@app.get("/events", dependencies=protected)
async def all_events():
    q: asyncio.Queue = asyncio.Queue(maxsize=500)
    _sse_all.append(q)

    def _close():
        try:
            _sse_all.remove(q)
        except ValueError:
            pass

    return _sse(q, {"event": "connected"}, _close)
