"""
Audio Ops — Standalone Inference Service
=========================================
Completely independent from Dagster, MLflow, and the training stack.

Endpoints:
  GET  /health                  — readiness probe
  GET  /models                  — list loaded models + classes
  POST /infer/file              — infer on uploaded audio file
  POST /infer/url               — infer on remote audio URL
  POST /streams                 — register RTSP stream for continuous inference
  DELETE /streams/{id}          — remove stream
  GET  /streams                 — list active streams
  GET  /streams/{id}/events     — SSE real-time results from one stream
  GET  /events                  — SSE real-time results from all streams

Model loading:
  New checkpoints (from updated model_trainer): contain full model object.
    → torch.load() → done, no reconstruction.
  Legacy checkpoints (plain state dict or old wrapped format):
    → reconstruct from config.yaml as fallback.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("inference")

# ------------------------------------------------------------------ #
# Config from env                                                      #
# ------------------------------------------------------------------ #

MINIO_ENDPOINT   = os.environ.get("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET     = os.environ.get("MINIO_MODELS_BUCKET", "audioops-bucket")
MODELS_DIR       = os.environ.get("MODELS_DIR", "/app/models")
TASKS_DIR        = os.environ.get("TASKS_DIR", "/app/tasks")

INFLUXDB_URL    = os.environ.get("INFLUXDB_URL", "http://influxdb:8086")
INFLUXDB_TOKEN  = os.environ.get("INFLUXDB_TOKEN", "")
INFLUXDB_ORG    = os.environ.get("INFLUXDB_ORG", "audioops")
INFLUXDB_BUCKET = os.environ.get("INFLUXDB_BUCKET", "audio_metrics")

SAMPLE_RATE = 16_000
CHUNK_MS    = int(os.environ.get("CHUNK_MS", "1000"))


# ------------------------------------------------------------------ #
# Task config loader (fallback for legacy checkpoints)                #
# ------------------------------------------------------------------ #

def _load_task_config(task: str) -> dict:
    """Read tasks/{task}/config.yaml - returns {} if not found."""
    import yaml
    path = os.path.join(TASKS_DIR, task, "config.yaml")
    if os.path.exists(path):
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        logger.info("Task config: %s - %d classes, arch=%s, input_shape=%s",
                    task, cfg.get("num_classes", 0),
                    cfg.get("model_architecture", "?"),
                    cfg.get("input_shape", "?"))
        return cfg
    logger.warning("No config.yaml for task '%s' at %s", task, path)
    return {}


# ------------------------------------------------------------------ #
# Legacy: reconstruct model from state dict                           #
# ------------------------------------------------------------------ #

def _build_architecture(architecture: str, num_classes: int, input_channels: int = 3):
    """
    Build a torchvision model by name, adapt input channels and head.
    Only used for legacy checkpoints that don't contain a full model object.
    """
    import torch.nn as nn
    from torchvision.models import get_model

    model = get_model(architecture, weights=None)

    if input_channels != 3:
        if hasattr(model, "conv1"):
            old = model.conv1
            model.conv1 = nn.Conv2d(
                input_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=old.bias is not None,
            )
        elif hasattr(model, "features") and hasattr(model.features[0], "__getitem__"):
            old = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
                input_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=old.bias is not None,
            )

    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif hasattr(model, "head"):
        model.head = nn.Linear(model.head.in_features, num_classes)

    return model


def _reconstruct_model(loaded, task: str):
    """
    Load a checkpoint into a ready-to-use nn.Module.

    Priority:
      1. Full model object embedded in checkpoint  → torch.load() is enough
      2. Wrapped dict with "model" key             → extract and return
      3. Wrapped dict with "model_state_dict" only → rebuild from config.yaml
      4. Plain state dict (old format)             → rebuild from config.yaml
    """
    import torch.nn as nn

    # Already a full model (e.g. old torch.save(model) format)
    if isinstance(loaded, nn.Module):
        return loaded

    if not isinstance(loaded, dict):
        raise ValueError(f"Unknown checkpoint type: {type(loaded)}")

    # New format: full model object embedded — just use it
    if "model" in loaded and isinstance(loaded["model"], nn.Module):
        model = loaded["model"]
        model.class_names  = loaded.get("class_names",        getattr(model, "class_names", []))
        model.input_shape  = loaded.get("input_shape",        getattr(model, "input_shape", [1, 128, 87]))
        model.architecture = loaded.get("model_architecture", getattr(model, "architecture", ""))
        model.task_name    = loaded.get("task_name",          task)
        model.feature_config = loaded.get("feature_config", {})
        logger.info("Loaded full model object for task '%s'", task)
        return model

    # Legacy format: state dict only — reconstruct architecture from config.yaml
    logger.warning(
        "Checkpoint for '%s' has no embedded model object — "
        "falling back to architecture reconstruction from config.yaml", task
    )

    if "model_state_dict" in loaded:
        state_dict   = loaded["model_state_dict"]
        class_names  = loaded.get("class_names", [])
        num_classes  = loaded.get("num_classes", len(class_names))
        architecture = loaded.get("model_architecture", "")
        input_shape  = loaded.get("input_shape", None)
    else:
        # Bare state dict
        state_dict   = loaded
        class_names  = []
        num_classes  = 0
        architecture = ""
        input_shape  = None

    cfg = _load_task_config(task)
    if not num_classes:
        num_classes = cfg.get("num_classes", 2)
    if not class_names:
        class_names = cfg.get("class_labels", [str(i) for i in range(num_classes)])
    if not architecture:
        architecture = cfg.get("model_architecture", "")
    if input_shape is None:
        input_shape = cfg.get("input_shape", [3, 224, 224])

    if not architecture:
        raise ValueError(
            f"No model_architecture for task '{task}' and no full model in checkpoint. "
            f"Add model_architecture to tasks/{task}/config.yaml"
        )

    input_channels = input_shape[0] if input_shape else 3
    model = _build_architecture(architecture, num_classes, input_channels)

    # Use strict=False to handle custom head mismatches gracefully
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(
            "State dict mismatch for '%s' — missing: %s  unexpected: %s. "
            "Model may have a custom head not matching standard architecture. "
            "Re-save with updated model_trainer to embed the full model.",
            task, missing[:3], unexpected[:3]
        )

    model.class_names  = class_names
    model.input_shape  = input_shape
    model.architecture = architecture
    return model


# ------------------------------------------------------------------ #
# Model store                                                          #
# ------------------------------------------------------------------ #

class ModelWrapper:
    def __init__(self, model, task_name: str):
        self.model       = model
        self.task_name   = task_name
        self.class_names: list[str] = getattr(model, "class_names", [])
        self.input_shape: list[int] = getattr(model, "input_shape", [1, 128, 87])
        self.feature_cfg  = getattr(model, "feature_config", {
            "feature_type": "spectrogram",
            "sample_rate": 16000,
            "n_mels": 128,
            "n_fft": 512,
            "hop_length": 160,
            "clip_duration_s": 3.0,
        })
        
    def _extract_features(self, audio: np.ndarray) -> "torch.Tensor":
        import torch, librosa
        cfg = self.feature_cfg
        ft  = cfg.get("feature_type", "mel_spectrogram")

        if ft == "mel_spectrogram":
            mel     = librosa.feature.melspectrogram(
                y=audio, sr=cfg["sample_rate"],
                n_mels=cfg["n_mels"], n_fft=cfg["n_fft"],
                hop_length=cfg["hop_length"],
            )
            features = librosa.power_to_db(mel, ref=np.max)
            # Pad/trim to expected width from input_shape [C, H, W]
            target_w = self.input_shape[2] if len(self.input_shape) == 3 else 87
            if features.shape[1] < target_w:
                features = np.pad(features, ((0,0),(0, target_w - features.shape[1])))
            else:
                features = features[:, :target_w]
            tensor = torch.from_numpy(features).float().unsqueeze(0).unsqueeze(0)

        elif ft == "waveform":
            tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)

        elif ft == "spectrogram":
            import torch
            wave   = torch.from_numpy(audio).float()
            spec   = torch.stft(wave, n_fft=cfg.get("n_fft", 512),
                                hop_length=cfg.get("hop_length", 160),
                                return_complex=True)
            features = spec.abs().numpy()
            target_w = self.input_shape[2] if len(self.input_shape) == 3 else 87
            if features.shape[1] < target_w:
                features = np.pad(features, ((0,0),(0, target_w - features.shape[1])))
            else:
                features = features[:, :target_w]
            tensor = torch.from_numpy(features).float().unsqueeze(0).unsqueeze(0)

        else:
            raise ValueError(f"Unknown feature_type '{ft}' for task '{self.task_name}'")

        return tensor

    def predict(self, audio: np.ndarray) -> dict:
        import torch

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
     
        tensor = self._extract_features(audio)
        
        with torch.no_grad():
            out = self.model(tensor)

        logits = out.logits if hasattr(out, "logits") else out
        probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        idx    = int(np.argmax(probs))
        labels = self.class_names or [str(i) for i in range(len(probs))]

        return {
            "label":      labels[idx] if idx < len(labels) else str(idx),
            "confidence": float(np.max(probs)),
            "scores":     {lbl: float(p) for lbl, p in zip(labels, probs)},
        }


class ModelStore:
    def __init__(self):
        self._models: dict[str, ModelWrapper] = {}

    def load_all(self) -> None:
        try:
            self._load_from_minio()
        except Exception as exc:
            logger.warning("MinIO unavailable (%s) - trying local dir", exc)
            self._load_from_local()

        if not self._models:
            logger.warning("No models loaded - inference endpoints will return empty results")
        else:
            logger.info("Models ready: %s", list(self._models))

    def _load_from_minio(self) -> None:
        import torch
        from minio import Minio

        client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY,
                       secret_key=MINIO_SECRET_KEY, secure=False)

        if not client.bucket_exists(MINIO_BUCKET):
            raise RuntimeError(f"Bucket '{MINIO_BUCKET}' not found")

        for obj in client.list_objects(MINIO_BUCKET, recursive=True):
            if not obj.object_name.endswith(".pt"):
                continue
            task = obj.object_name.split("/")[0]
            try:
                data   = client.get_object(MINIO_BUCKET, obj.object_name).read()
                loaded = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)

                model  = _reconstruct_model(loaded, task)
                model.eval()
                self._models[task] = ModelWrapper(model, task)
                logger.info("Loaded: %s <- %s", task, obj.object_name)
            except Exception as exc:
                logger.warning("Could not load %s: %s", obj.object_name, exc)

    def _load_from_local(self) -> None:
        import glob
        import torch

        for path in glob.glob(os.path.join(MODELS_DIR, "**/*.pt"), recursive=True):
            task = os.path.splitext(os.path.basename(path))[0]
            task = task.replace("_best", "").replace("_model", "").replace("best", "")
            try:
                loaded = torch.load(path, map_location="cpu", weights_only=False)

                model  = _reconstruct_model(loaded, task)
                model.eval()
                self._models[task] = ModelWrapper(model, task)
                logger.info("Loaded local: %s <- %s", task, path)
            except Exception as exc:
                logger.warning("Could not load %s: %s", path, exc)

    def get(self, task: str) -> Optional[ModelWrapper]:
        return self._models.get(task)

    def all(self) -> dict[str, ModelWrapper]:
        return dict(self._models)

    def names(self) -> list[str]:
        return list(self._models)


# ------------------------------------------------------------------ #
# InfluxDB writer                                                      #
# ------------------------------------------------------------------ #

class InfluxWriter:
    def __init__(self):
        self._ok = False
        if not INFLUXDB_TOKEN:
            logger.warning("No INFLUXDB_TOKEN set - metrics will not be written")
            return
        try:
            from influxdb_client import InfluxDBClient
            from influxdb_client.client.write_api import SYNCHRONOUS
            self._client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
            self._write  = self._client.write_api(write_options=SYNCHRONOUS)
            self._ok     = True
            logger.info("InfluxDB connected: %s", INFLUXDB_URL)
        except Exception as exc:
            logger.warning("InfluxDB init failed: %s", exc)

    def write_inference(self, task: str, label: str, confidence: float,
                        scores: dict, rms: float, source: str,
                        stream_id: str = "", duration_s: float = 0.0) -> None:
        if not self._ok:
            return
        try:
            from influxdb_client import Point
            p = (Point(f"inference_{task}")
                 .tag("task", task)
                 .tag("source_type", source)
                 .tag("stream_id", stream_id or "none")
                 .field("label", label)
                 .field("confidence", confidence)
                 .field("rms", rms)
                 .field("audio_duration_s", duration_s))
            for cls, score in scores.items():
                p = p.field(f"score_{cls}", score)
            self._write.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=p)
        except Exception as exc:
            logger.debug("InfluxDB write error: %s", exc)

    def write_stream_health(self, stream_id: str, active: bool,
                            frames: int, errors: int, rms: float) -> None:
        if not self._ok:
            return
        try:
            from influxdb_client import Point
            p = (Point("stream_health")
                 .tag("stream_id", stream_id)
                 .field("active", int(active))
                 .field("frames_processed", frames)
                 .field("error_count", errors)
                 .field("rms", rms))
            self._write.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=p)
        except Exception as exc:
            logger.debug("InfluxDB write error: %s", exc)


# ------------------------------------------------------------------ #
# RTSP stream handle + worker                                          #
# ------------------------------------------------------------------ #

class StreamHandle:
    def __init__(self, stream_id: str, url: str, chunk_ms: int = CHUNK_MS):
        self.stream_id     = stream_id
        self.url           = url
        self.chunk_ms      = chunk_ms
        self.chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
        self.active        = True
        self.frames        = 0
        self.errors        = 0
        self.last_rms      = 0.0
        self.task: Optional[asyncio.Task] = None
        self.subscribers: list[asyncio.Queue] = []


async def _stream_worker(handle: StreamHandle) -> None:
    chunk_bytes = handle.chunk_samples * 2

    async def _spawn():
        return await asyncio.create_subprocess_exec(
            "ffmpeg", "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-i", handle.url,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE), "-ac", "1",
            "-f", "s16le", "pipe:1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def _read_exact(stream, n) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = await stream.read(n - len(buf))
            if not chunk:
                return bytes(buf)
            buf.extend(chunk)
        return bytes(buf)

    logger.info("Worker started: %s -> %s", handle.stream_id, handle.url)

    while handle.active:
        proc = None
        try:
            proc = await _spawn()
            while handle.active:
                raw = await asyncio.wait_for(
                    _read_exact(proc.stdout, chunk_bytes), timeout=10
                )
                if not raw:
                    break

                audio           = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                rms             = float(np.sqrt(np.mean(audio ** 2)))
                handle.last_rms = rms
                handle.frames  += 1

                loop = asyncio.get_event_loop()
                for task_name, model in store.all().items():
                    try:
                        result = await loop.run_in_executor(None, model.predict, audio)
                        payload = {
                            "timestamp":  time.time(),
                            "stream_id":  handle.stream_id,
                            "task":       task_name,
                            "label":      result.get("label"),
                            "confidence": result.get("confidence"),
                            "rms":        rms,
                            "scores":     result.get("scores", {}),
                        }
                        influx.write_inference(
                            task=task_name,
                            label=result.get("label", ""),
                            confidence=result.get("confidence", 0.0),
                            scores=result.get("scores", {}),
                            rms=rms, source="stream",
                            stream_id=handle.stream_id,
                        )
                        for q in handle.subscribers + _sse_all:
                            try:
                                q.put_nowait(payload)
                            except asyncio.QueueFull:
                                pass
                    except Exception as exc:
                        handle.errors += 1
                        logger.debug("Inference error [%s/%s]: %s",
                                     handle.stream_id, task_name, exc)

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
# Globals                                                              #
# ------------------------------------------------------------------ #

store    = ModelStore()
influx   = InfluxWriter()
streams: dict[str, StreamHandle] = {}
_sse_all: list[asyncio.Queue]    = []


# ------------------------------------------------------------------ #
# Lifespan                                                             #
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
    yield
    for h in streams.values():
        h.active = False
        if h.task:
            h.task.cancel()
    heartbeat.cancel()


async def _heartbeat_loop() -> None:
    while True:
        await asyncio.sleep(15)
        for h in streams.values():
            influx.write_stream_health(
                stream_id=h.stream_id,
                active=h.active and bool(h.task and not h.task.done()),
                frames=h.frames, errors=h.errors, rms=h.last_rms,
            )


# ------------------------------------------------------------------ #
# App                                                                  #
# ------------------------------------------------------------------ #

app = FastAPI(
    title="Audio Ops Inference",
    version="2.0.0",
    description="Standalone audio inference - no Dagster required",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _resample(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    except Exception:
        return audio


async def _run_all_models(audio: np.ndarray, sr: int,
                           source: str, stream_id: str = "") -> dict:
    audio      = _resample(audio, sr)
    duration_s = len(audio) / SAMPLE_RATE
    rms        = float(np.sqrt(np.mean(audio ** 2)))
    loop       = asyncio.get_event_loop()
    results    = {}

    for task_name, model in store.all().items():
        try:
            t0     = time.time()
            result = await loop.run_in_executor(None, model.predict, audio)
            result["latency_ms"] = round((time.time() - t0) * 1000, 1)
            results[task_name]   = result
            influx.write_inference(
                task=task_name, label=result.get("label", ""),
                confidence=result.get("confidence", 0.0),
                scores=result.get("scores", {}),
                rms=rms, source=source,
                stream_id=stream_id, duration_s=duration_s,
            )
        except Exception as exc:
            results[task_name] = {"error": str(exc)}

    return {"tasks": results, "duration_s": round(duration_s, 3), "rms": round(rms, 4)}


async def _add_stream(stream_id: str, url: str, chunk_ms: int = CHUNK_MS) -> StreamHandle:
    handle             = StreamHandle(stream_id=stream_id, url=url, chunk_ms=chunk_ms)
    streams[stream_id] = handle
    handle.task        = asyncio.create_task(
        _stream_worker(handle), name=f"worker_{stream_id}"
    )
    logger.info("Stream registered: %s -> %s", stream_id, url)
    return handle


# ------------------------------------------------------------------ #
# Endpoints                                                            #
# ------------------------------------------------------------------ #

@app.get("/health")
async def health():
    return {
        "status":   "ok",
        "models":   store.names(),
        "streams":  len(streams),
        "influxdb": INFLUXDB_URL,
    }


@app.get("/models")
async def list_models():
    return {"models": [
        {"task": name, "classes": m.class_names, "input_shape": m.input_shape}
        for name, m in store.all().items()
    ]}


@app.post("/infer/file")
async def infer_file(file: UploadFile = File(...)):
    data = await file.read()
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception as exc:
        raise HTTPException(422, f"Cannot decode audio: {exc}")
    return await _run_all_models(audio, sr, source="file")


@app.post("/infer/url")
async def infer_url(url: str):
    import httpx
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url)
            r.raise_for_status()
        audio, sr = sf.read(io.BytesIO(r.content), dtype="float32", always_2d=False)
    except Exception as exc:
        raise HTTPException(422, str(exc))
    return await _run_all_models(audio, sr, source="url")


class AddStreamRequest(BaseModel):
    url: str
    stream_id: Optional[str] = None
    chunk_duration_ms: int = Field(CHUNK_MS, ge=100, le=10000)


@app.post("/streams", status_code=201)
async def add_stream(req: AddStreamRequest):
    sid = req.stream_id or f"stream_{int(time.time())}"
    if sid in streams:
        raise HTTPException(409, f"Stream '{sid}' already registered")
    await _add_stream(sid, req.url, req.chunk_duration_ms)
    return {"stream_id": sid, "url": req.url, "status": "started"}


@app.delete("/streams/{stream_id}")
async def remove_stream(stream_id: str):
    handle = streams.pop(stream_id, None)
    if not handle:
        raise HTTPException(404, f"Stream '{stream_id}' not found")
    handle.active = False
    if handle.task:
        handle.task.cancel()
    return {"stream_id": stream_id, "status": "removed"}


@app.get("/streams")
async def list_streams():
    return {"streams": [
        {
            "stream_id": sid, "url": h.url, "active": h.active,
            "frames": h.frames, "errors": h.errors, "rms": round(h.last_rms, 4),
        }
        for sid, h in streams.items()
    ]}


@app.get("/streams/{stream_id}/events")
async def stream_events(stream_id: str):
    handle = streams.get(stream_id)
    if not handle:
        raise HTTPException(404, f"Stream '{stream_id}' not found")

    q: asyncio.Queue = asyncio.Queue(maxsize=200)
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
    
@app.post("/simulation/start")
async def start_simulation():
    """Register all stadium streams for inference."""
    streams_to_add = [
        ("stadium_north", "rtsp://mediamtx:8554/stadium_north"),
        ("stadium_south", "rtsp://mediamtx:8554/stadium_south"),
        ("stadium_east",  "rtsp://mediamtx:8554/stadium_east"),
        ("stadium_west",  "rtsp://mediamtx:8554/stadium_west"),
    ]
    results = {}
    for sid, url in streams_to_add:
        if sid not in streams:
            await _add_stream(sid, url)
            results[sid] = "started"
        else:
            results[sid] = "already running"
    return {"status": "ok", "streams": results}

@app.post("/simulation/stop")
async def stop_simulation():
    """Remove all stadium streams."""
    removed = []
    for sid in ["stadium_north", "stadium_south", "stadium_east", "stadium_west"]:
        if sid in streams:
            handle = streams.pop(sid)
            handle.active = False
            if handle.task:
                handle.task.cancel()
            removed.append(sid)
    return {"status": "ok", "removed": removed}