"""
Audio Inference API
===================
FastAPI application that exposes:

  POST /infer/file           — infer on an uploaded audio file
  POST /infer/url            — infer on a remote audio URL
  POST /streams              — register an RTSP stream for continuous inference
  DELETE /streams/{id}       — remove a stream
  GET  /streams              — list active streams
  GET  /streams/{id}/events  — SSE: real-time results from a stream
  GET  /tasks                — list available tasks / models
  GET  /health               — readiness probe

All results are written to InfluxDB automatically.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from audio_ops_dagster.monitoring.influx_writer import InfluxWriter, Metrics, get_writer
from audio_ops_dagster.streaming.rtsp_consumer import push_file_to_rtsp
from audio_ops_dagster.streaming.stream_inference_engine import (
    StreamInferenceEngine,
    StreamResult,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Model registry (lazy-loaded)                                        #
# ------------------------------------------------------------------ #

class ModelRegistry:
    """
    Loads and caches task models from disk / MLflow on first use.
    Each task must implement predict(audio: np.ndarray) → dict.
    """

    def __init__(self):
        self._models: dict[str, Any] = {}
        self._model_meta: dict[str, dict] = {}
        self._loaded = False

    def load(self) -> None:
        """Discover and load all available task models."""
        import glob, importlib

        models_dir = os.environ.get("MODELS_DIR", "/app/models")
        logger.info("Loading models from %s", models_dir)

        # Try MLflow model registry
        try:
            self._load_from_mlflow()
        except Exception as exc:
            logger.warning("MLflow model load failed: %s — trying local models", exc)
            self._load_from_disk(models_dir)

        self._loaded = True
        logger.info("Loaded %d models: %s", len(self._models), list(self._models))

    def _load_from_mlflow(self):
        import mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        client = mlflow.tracking.MlflowClient()

        for rm in client.search_registered_models():
            task = rm.name
            try:
                latest = client.get_latest_versions(task, stages=["Production", "None"])
                if not latest:
                    continue
                version = latest[0]
                model = mlflow.pytorch.load_model(version.source)
                wrapper = _PyTorchModelWrapper(model, task)
                self._models[task] = wrapper
                self._model_meta[task] = {
                    "name": task,
                    "version": version.version,
                    "source": "mlflow",
                    "run_id": version.run_id,
                }
                logger.info("Loaded MLflow model: %s v%s", task, version.version)
            except Exception as exc:
                logger.warning("Could not load MLflow model %s: %s", task, exc)

    def _load_from_disk(self, models_dir: str):
        import glob
        import torch

        for path in glob.glob(os.path.join(models_dir, "**/*.pt"), recursive=True):
            task = os.path.splitext(os.path.basename(path))[0].replace("_model", "")
            try:
                model = torch.load(path, map_location="cpu")
                model.eval()
                wrapper = _PyTorchModelWrapper(model, task)
                self._models[task] = wrapper
                self._model_meta[task] = {
                    "name": task,
                    "version": "local",
                    "source": path,
                }
                logger.info("Loaded local model: %s from %s", task, path)
            except Exception as exc:
                logger.warning("Could not load model %s: %s", path, exc)

    def get(self, task: str) -> Optional[Any]:
        return self._models.get(task)

    def all(self) -> dict[str, Any]:
        return dict(self._models)

    def meta(self) -> list[dict]:
        return [{"task": t, **m} for t, m in self._model_meta.items()]

    def task_names(self) -> list[str]:
        return list(self._models.keys())


class _PyTorchModelWrapper:
    """Wraps a PyTorch model with a standardised predict() interface."""

    SAMPLE_RATE = 16_000

    def __init__(self, model, task_name: str):
        self.model = model
        self.task_name = task_name
        self.model_name = task_name

    def predict(self, audio: np.ndarray) -> dict:
        import torch

        # Preprocess
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        tensor = torch.from_numpy(audio).float().unsqueeze(0)

        with torch.no_grad():
            output = self.model(tensor)

        # Handle different output shapes
        if hasattr(output, "logits"):
            logits = output.logits
        elif isinstance(output, torch.Tensor):
            logits = output
        else:
            return {"raw": str(output)}

        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        label_idx = int(np.argmax(probs))
        confidence = float(np.max(probs))

        # Try to get class names from model
        class_names = getattr(self.model, "class_names", None) or [
            str(i) for i in range(len(probs))
        ]

        scores = {name: float(p) for name, p in zip(class_names, probs)}

        return {
            "label": class_names[label_idx] if label_idx < len(class_names) else str(label_idx),
            "confidence": confidence,
            "scores": scores,
        }


# ------------------------------------------------------------------ #
# Global singletons                                                   #
# ------------------------------------------------------------------ #

registry = ModelRegistry()
stream_engine: Optional[StreamInferenceEngine] = None
influx: Optional[InfluxWriter] = None

# Per-stream SSE subscriber queues: stream_id → list[asyncio.Queue]
_sse_subscribers: dict[str, list[asyncio.Queue]] = {}


# ------------------------------------------------------------------ #
# Lifespan                                                             #
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global stream_engine, influx

    influx = get_writer()
    registry.load()

    stream_engine = StreamInferenceEngine(
        model_registry=registry.all(),
        influx_writer=influx,
    )
    await stream_engine.start()

    # Background task: fan out stream results to SSE subscribers
    fan_out_task = asyncio.create_task(_fan_out_results())

    # Auto-register any RTSP streams from env
    auto_streams = os.environ.get("AUTO_RTSP_STREAMS", "")
    for entry in auto_streams.split(","):
        entry = entry.strip()
        if entry:
            parts = entry.split("|")
            url = parts[0]
            sid = parts[1] if len(parts) > 1 else None
            try:
                await stream_engine.add_stream(url, stream_id=sid)
                logger.info("Auto-registered stream: %s", url)
            except Exception as exc:
                logger.warning("Failed to auto-register stream %s: %s", url, exc)

    yield

    fan_out_task.cancel()
    await stream_engine.stop()
    influx.close()


# ------------------------------------------------------------------ #
# App                                                                 #
# ------------------------------------------------------------------ #

app = FastAPI(
    title="Audio Ops Inference API",
    version="2.0.0",
    description="Real-time audio inference — file upload, URL, and continuous RTSP streams",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
# Request / response models                                           #
# ------------------------------------------------------------------ #

class InferUrlRequest(BaseModel):
    url: str
    tasks: Optional[list[str]] = None  # None = all loaded tasks

class AddStreamRequest(BaseModel):
    url: str = Field(..., description="RTSP URL, e.g. rtsp://localhost:8554/microphone")
    stream_id: Optional[str] = None
    chunk_duration_ms: int = Field(1000, ge=100, le=10000)
    overlap_duration_ms: int = Field(0, ge=0, le=5000)
    tasks: Optional[list[str]] = None   # subset of tasks to run on this stream

class PushFileRequest(BaseModel):
    file_path: str
    rtsp_path: str = "file_stream"
    loop: bool = False


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

async def _load_audio(data: bytes) -> tuple[np.ndarray, int]:
    """Load raw audio bytes into a float32 numpy array."""
    with io.BytesIO(data) as buf:
        audio, sr = sf.read(buf, dtype="float32", always_2d=False)
    return audio, sr


def _resample_if_needed(audio: np.ndarray, sr: int, target_sr: int = 16_000) -> np.ndarray:
    if sr == target_sr:
        return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    except ImportError:
        logger.warning("librosa not available — skipping resample")
        return audio


async def _run_tasks(
    audio: np.ndarray,
    sr: int,
    tasks: Optional[list[str]],
    source_type: str = "file",
    stream_id: str = "",
) -> dict:
    """Run selected (or all) task models on audio and return results dict."""
    loop = asyncio.get_event_loop()
    audio_16k = _resample_if_needed(audio, sr)
    duration_s = len(audio_16k) / 16_000
    rms = float(np.sqrt(np.mean(audio_16k ** 2)))

    task_names = tasks or registry.task_names()
    results = {}

    for task in task_names:
        model = registry.get(task)
        if model is None:
            results[task] = {"error": "model not found"}
            continue
        try:
            t0 = time.time()
            result = await loop.run_in_executor(None, model.predict, audio_16k)
            latency_ms = (time.time() - t0) * 1000
            result["latency_ms"] = round(latency_ms, 2)
            results[task] = result

            # Write to InfluxDB
            if task == "speech_to_text":
                metric = Metrics.speech_to_text(
                    model=task,
                    transcript=result.get("transcript", ""),
                    word_count=result.get("word_count", 0),
                    confidence=result.get("confidence", 0.0),
                    source_type=source_type,
                    stream_id=stream_id,
                    audio_duration_s=duration_s,
                )
            else:
                metric = Metrics.inference_result(
                    task=task,
                    model=task,
                    label=result.get("label", ""),
                    confidence=result.get("confidence", 0.0),
                    scores=result.get("scores"),
                    source_type=source_type,
                    stream_id=stream_id,
                    audio_duration_s=duration_s,
                    rms=rms,
                )
            if influx:
                influx.write(metric)

        except Exception as exc:
            logger.warning("Task %s failed: %s", task, exc)
            results[task] = {"error": str(exc)}

    return {
        "tasks": results,
        "audio_duration_s": round(duration_s, 3),
        "rms": round(rms, 4),
        "timestamp": time.time(),
    }


# ------------------------------------------------------------------ #
# Endpoints                                                           #
# ------------------------------------------------------------------ #

@app.get("/health")
async def health():
    writer_ok = influx.ping() if influx else False
    engine_stats = stream_engine.get_stats() if stream_engine else {}
    return {
        "status": "ok",
        "models": registry.task_names(),
        "influxdb": writer_ok,
        "engine": engine_stats,
    }


@app.get("/tasks")
async def list_tasks():
    return {"tasks": registry.meta()}


@app.post("/infer/file")
async def infer_file(
    file: UploadFile = File(...),
    tasks: Optional[str] = None,   # comma-separated
):
    """Run inference on an uploaded audio file."""
    data = await file.read()
    try:
        audio, sr = await _load_audio(data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode audio: {exc}")

    task_list = [t.strip() for t in tasks.split(",")] if tasks else None
    result = await _run_tasks(audio, sr, task_list, source_type="file")
    return result


@app.post("/infer/url")
async def infer_url(req: InferUrlRequest):
    """Download audio from a URL and run inference."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(req.url)
            response.raise_for_status()
            data = response.content
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot fetch URL: {exc}")

    try:
        audio, sr = await _load_audio(data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode audio: {exc}")

    result = await _run_tasks(audio, sr, req.tasks, source_type="url")
    return result


# ------------------------------------------------------------------ #
# Stream management                                                   #
# ------------------------------------------------------------------ #

@app.post("/streams", status_code=201)
async def add_stream(req: AddStreamRequest):
    """Register an RTSP stream for continuous inference."""
    if stream_engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")

    stream_id = await stream_engine.add_stream(
        url=req.url,
        stream_id=req.stream_id,
        chunk_duration_ms=req.chunk_duration_ms,
        overlap_duration_ms=req.overlap_duration_ms,
    )
    return {"stream_id": stream_id, "url": req.url, "status": "started"}


@app.delete("/streams/{stream_id}")
async def remove_stream(stream_id: str):
    if stream_engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")
    await stream_engine.remove_stream(stream_id)
    return {"stream_id": stream_id, "status": "removed"}


@app.get("/streams")
async def list_streams():
    if stream_engine is None:
        return {"streams": []}
    return {"streams": stream_engine.list_streams()}


@app.post("/streams/push-file")
async def push_file_as_stream(req: PushFileRequest):
    """Push a local audio file to MediaMTX as an RTSP stream (for testing)."""
    rtsp_url = f"rtsp://mediamtx:8554/{req.rtsp_path}"
    asyncio.create_task(
        push_file_to_rtsp(req.file_path, rtsp_url, loop=req.loop)
    )
    return {"rtsp_url": rtsp_url, "status": "pushing"}


# ------------------------------------------------------------------ #
# Server-Sent Events for real-time results                            #
# ------------------------------------------------------------------ #

@app.get("/streams/{stream_id}/events")
async def stream_events(stream_id: str):
    """
    SSE endpoint — subscribe to real-time inference results for one stream.
    Connect with EventSource in the browser or curl:

        curl -N http://localhost:8000/streams/mic_1/events
    """
    queue: asyncio.Queue[StreamResult] = asyncio.Queue(maxsize=200)
    _sse_subscribers.setdefault(stream_id, []).append(queue)

    async def event_generator():
        try:
            yield f"data: {json.dumps({'event': 'connected', 'stream_id': stream_id})}\n\n"
            while True:
                try:
                    result = await asyncio.wait_for(queue.get(), timeout=30)
                    payload = {
                        "timestamp": result.timestamp,
                        "task": result.task_name,
                        "label": result.label,
                        "confidence": result.confidence,
                        "rms": result.rms,
                        "result": result.result,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    # Keep-alive comment
                    yield ": keep-alive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            try:
                _sse_subscribers[stream_id].remove(queue)
            except (KeyError, ValueError):
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/events")
async def all_events():
    """SSE endpoint — subscribe to results from ALL active streams."""
    queue: asyncio.Queue[StreamResult] = asyncio.Queue(maxsize=500)
    _sse_subscribers.setdefault("__all__", []).append(queue)

    async def event_generator():
        try:
            yield f"data: {json.dumps({'event': 'connected'})}\n\n"
            while True:
                try:
                    result = await asyncio.wait_for(queue.get(), timeout=30)
                    payload = {
                        "timestamp": result.timestamp,
                        "stream_id": result.stream_id,
                        "task": result.task_name,
                        "label": result.label,
                        "confidence": result.confidence,
                        "rms": result.rms,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            try:
                _sse_subscribers["__all__"].remove(queue)
            except (KeyError, ValueError):
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------ #
# Background: fan out StreamInferenceEngine results to SSE queues     #
# ------------------------------------------------------------------ #

async def _fan_out_results():
    """Continuously drain the engine's result queue and push to SSE subscribers."""
    while True:
        result = await stream_engine.get_result(timeout=1.0)
        if result is None:
            continue

        # Push to stream-specific subscribers
        for q in _sse_subscribers.get(result.stream_id, []):
            try:
                q.put_nowait(result)
            except asyncio.QueueFull:
                pass

        # Push to "__all__" subscribers
        for q in _sse_subscribers.get("__all__", []):
            try:
                q.put_nowait(result)
            except asyncio.QueueFull:
                pass