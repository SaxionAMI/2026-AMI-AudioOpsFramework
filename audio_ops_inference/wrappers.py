"""
wrappers.py — how a task is served.

A *serve wrapper* turns a task into something the streaming service can feed
audio chunks to. Wrappers are created by registered factories keyed on a
wrapper type, so new servable kinds plug in without editing the API::

    from audio_ops_inference.wrappers import register_wrapper, ServeWrapper

    @register_wrapper("my_kind")
    def make(task, task_name, cfg):          # cfg: TaskConfig
        return MyWrapper(task, task_name, cfg)

Built-in types:
  ``classifier``        any task with ``build_model`` + a trained checkpoint
                        (classification, regression, detection …) — features
                        are computed exactly as in training and the output goes
                        through the task's own ``postprocess``.
  ``transcription``     tasks exposing ``predict(audio, t_offset=...)`` (STT)
  ``pretrained_event``  pretrained event taggers exposing
                        ``predict(audio, sample_rate=..., source=...)`` (YAMNet-style)

Every stream gets its own wrapper instance (:meth:`ServeWrapper.for_stream`)
so buffered audio from one stream can never leak into another; the heavy model
object is shared.
"""

from __future__ import annotations

import collections
import copy
import inspect
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from audio_ops_dagster._log import get_logger

logger = get_logger("audio_ops_inference.wrappers")

SAMPLE_RATE = 16_000

WrapperFactory = Callable[[Any, str, Any], Optional["ServeWrapper"]]
_WRAPPER_FACTORIES: Dict[str, WrapperFactory] = {}


def register_wrapper(wrapper_type: str):
    """Register a ``(task, task_name, cfg) -> ServeWrapper | None`` factory."""

    def deco(fn: WrapperFactory) -> WrapperFactory:
        _WRAPPER_FACTORIES[wrapper_type] = fn
        return fn

    return deco


def get_wrapper_factory(wrapper_type: str) -> Optional[WrapperFactory]:
    return _WRAPPER_FACTORIES.get(wrapper_type)


def list_wrapper_types() -> List[str]:
    return sorted(_WRAPPER_FACTORIES)


def resolve_wrapper_type(task, cfg) -> str:
    """Explicit ``serve_wrapper`` wins; else infer from ``task_type``/``TASK_TYPE``."""
    explicit = getattr(cfg, "serve_wrapper", None)
    if explicit:
        return str(explicit)
    if getattr(cfg, "task_type", None) == "nonverbal_pretrained":
        return "pretrained_event"
    task_type = getattr(task, "task_type", None) or getattr(task, "TASK_TYPE", "")
    if task_type == "transcription":
        return "transcription"
    return "classifier"


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0


def _mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    return audio.mean(axis=1) if audio.ndim > 1 else audio


class ServeWrapper:
    """Base class for serve wrappers.

    Subclasses implement :meth:`predict` (streaming, may buffer) and may
    override :meth:`predict_clip` (one-shot, whole clip) and :meth:`describe`.
    ``kind`` tells the API how to route results: ``classifier`` | ``transcription``
    | ``event``.
    """

    kind = "classifier"

    def __init__(self, task, task_name: str, cfg=None):
        self.task = task
        self.task_name = task_name
        self.cfg = cfg

    # streaming
    def predict(self, audio: np.ndarray, media_t0: float = 0.0) -> dict:
        raise NotImplementedError

    def predict_clip(self, audio: np.ndarray) -> dict:
        """Score a whole clip at once (file upload). Default: one streaming call."""
        return self.for_stream().predict(audio, 0.0)

    def reset(self) -> None:
        """Clear per-stream state."""

    def for_stream(self) -> ServeWrapper:
        """A fresh instance for one stream: shares the model, owns its buffers."""
        clone = copy.copy(self)
        clone.reset()
        return clone

    def describe(self) -> dict:
        return {"task": self.task_name, "type": self.kind}


# ── Classifier / generic model wrapper ───────────────────────────────────────


def _normalize(pred: Any) -> Dict[str, Any]:
    if isinstance(pred, list) and pred:
        pred = pred[0]
    if not isinstance(pred, dict):
        return {"result": pred if isinstance(pred, (str, int, float, bool)) else str(pred)}
    out = dict(pred)
    if "scores" not in out and isinstance(out.get("probabilities"), dict):
        out["scores"] = out["probabilities"]
    return out


class ModelWrapper(ServeWrapper):
    """Serves any ``build_model`` task with a checkpoint.

    Features are produced by the same extractor, sample rate and window length
    the task was trained with (``feature_type``, ``segment_duration``). In a
    stream, a rolling window of the last ``segment_duration`` seconds is scored
    on every chunk; a file upload is segmented exactly like training and each
    segment is scored, plus an aggregate.
    """

    kind = "classifier"

    def __init__(self, model, task_name: str, task=None, cfg=None, device=None):
        super().__init__(task, task_name, cfg)
        from audio_ops_dagster.core.audio_processor import AudioProcessor

        self.model = model
        self.device = device
        self.class_names = list(getattr(cfg, "class_labels", None) or [])
        self.input_shape = getattr(cfg, "input_shape", None)
        self.processor = AudioProcessor(
            sample_rate=int(getattr(cfg, "sample_rate", SAMPLE_RATE)),
            segment_duration=float(getattr(cfg, "segment_duration", 1.0)),
            overlap=float(getattr(cfg, "segment_overlap", 0.25)),
            feature_type=getattr(cfg, "feature_type", "spectrogram"),
        )
        self._window = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self._window = np.zeros(0, dtype=np.float32)

    def _resample(self, audio: np.ndarray) -> np.ndarray:
        sr = self.processor.sample_rate
        if sr == SAMPLE_RATE:
            return audio
        import librosa

        return librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=sr)

    def _score(self, window: np.ndarray) -> Dict[str, Any]:
        import torch

        from audio_ops_dagster.core.runtime import to_model_input

        seg_n = self.processor.segment_samples
        if len(window) < seg_n:
            window = np.pad(window, (0, seg_n - len(window)))
        feature = self.processor.extract(window[:seg_n])
        x = to_model_input(feature, device=self.device)
        with torch.no_grad():
            out = self.model(x)
        out = out.logits if hasattr(out, "logits") else out
        if self.task is not None:
            return _normalize(self.task.postprocess(out))
        probs = torch.softmax(out, dim=-1).flatten().cpu().numpy()
        labels = self.class_names or [str(i) for i in range(len(probs))]
        idx = int(np.argmax(probs))
        return {"label": labels[idx] if idx < len(labels) else str(idx),
                "confidence": float(probs[idx]),
                "scores": {lbl: float(p) for lbl, p in zip(labels, probs)}}

    def predict(self, audio: np.ndarray, media_t0: float = 0.0) -> dict:
        audio = self._resample(_mono(audio))
        sr = self.processor.sample_rate
        seg_n = self.processor.segment_samples
        self._window = np.concatenate([self._window, audio])[-seg_n:]
        result = self._score(self._window)
        end = media_t0 + len(audio) / sr
        result["media_ts_start"] = round(max(0.0, end - len(self._window) / sr), 3)
        result["media_ts_end"] = round(end, 3)
        return result

    def predict_clip(self, audio: np.ndarray) -> dict:
        audio = self._resample(_mono(audio))
        hop = self.processor.hop_seconds
        segments = []
        for i, seg in enumerate(self.processor.segments(audio)):
            r = self._score(seg)
            r["start_sec"] = round(i * hop, 3)
            r["end_sec"] = round(i * hop + self.processor.segment_duration, 3)
            segments.append(r)
        summary: Dict[str, Any] = dict(segments[0]) if segments else {}
        score_dicts = [s.get("scores") for s in segments if isinstance(s.get("scores"), dict)]
        if score_dicts and len(score_dicts) == len(segments):
            keys = list(score_dicts[0])
            mean = {k: float(np.mean([d.get(k, 0.0) for d in score_dicts])) for k in keys}
            best = max(mean, key=mean.get)
            summary = {"label": best, "confidence": mean[best], "scores": mean}
        summary.pop("start_sec", None)
        summary.pop("end_sec", None)
        summary["segments"] = segments
        summary["aggregation"] = "mean_scores" if score_dicts else "first_segment"
        return summary

    def describe(self) -> dict:
        return {"task": self.task_name, "type": "classifier",
                "task_type": getattr(self.task, "task_type", "classification"),
                "classes": self.class_names, "input_shape": self.input_shape,
                "feature_type": self.processor.feature_type,
                "segment_duration": self.processor.segment_duration}


# ── Transcription (STT) ──────────────────────────────────────────────────────

# Whisper-family models hallucinate stock phrases on silence/noise (subtitle
# credits from their training data). Matched case-insensitively after stripping
# punctuation. Extend with STT_HALLUCINATION_PHRASES (comma-separated).
DEFAULT_STT_HALLUCINATIONS = {
    # English
    "thank you for watching", "thanks for watching", "please subscribe",
    "subtitles by the amara.org community", "amara.org",
    # Dutch
    "bedankt voor het kijken", "bedankt voor het kijken en tot de volgende keer",
    "ondertiteld door de amara.org gemeenschap", "ondertiteling door de amara.org gemeenschap",
    "ondertitels ingediend door de amara.org gemeenschap",
    "tot de volgende keer", "tot de volgende video", "abonneer je", "abonneer op het kanaal",
}


def _norm_text(t: str) -> str:
    return (t or "").strip().lower().strip(" .!?-…")


def is_hallucination(text: str, extra: Optional[set] = None) -> bool:
    t = _norm_text(text)
    if not t:
        return True
    if t in DEFAULT_STT_HALLUCINATIONS or (extra and t in extra):
        return True
    words = t.split()
    # degenerate repetition like "ja ja ja ja" or a single token echoed
    return len(words) >= 4 and len(set(words)) <= 2


class STTModelWrapper(ServeWrapper):
    """Buffers chunks to ``buffer_seconds``, then transcribes. Word ts on the media clock."""

    kind = "transcription"

    def __init__(self, task, task_name: str, buffer_seconds: float = 5.0,
                 silence_rms: float = 0.006, use_prompt: bool = False,
                 extra_hallucinations: Optional[List[str]] = None, cfg=None):
        super().__init__(task, task_name, cfg)
        self.class_names: list = []
        self.input_shape: list = []
        self._silence_rms = silence_rms
        self._buffer_seconds = buffer_seconds
        self._target_samples = int(SAMPLE_RATE * buffer_seconds)
        self._use_prompt = use_prompt
        self._extra = {_norm_text(p) for p in (extra_hallucinations or [])}
        self.reset()

    def reset(self) -> None:
        self._buffer: collections.deque = collections.deque()
        self._buffer_samples = 0
        self._pending_start: Optional[float] = None
        self._last_prompt = ""
        self._last_text = ""
        self._last_result = {
            "transcription": "", "confidence": 0.0, "word_count": 0,
            "words": [], "language": "", "buffering": True,
            "media_ts_start": 0.0, "media_ts_end": 0.0,
        }

    def _transcribe(self, audio: np.ndarray, t_offset: float, prompt: str = "") -> dict:
        fn = getattr(self.task, "predict", None)
        if fn is None:
            return self.task.transcribe(audio, t_offset=t_offset)
        if self._use_prompt and prompt and "initial_prompt" in inspect.signature(fn).parameters:
            return fn(audio, t_offset=t_offset, initial_prompt=prompt)
        return fn(audio, t_offset=t_offset)

    def predict(self, audio: np.ndarray, media_t0: float = 0.0) -> dict:
        audio = _mono(audio)
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

        # Silence gate: skip near-silence (kills hallucinations, saves a call).
        if _rms(combined) < self._silence_rms:
            self._last_prompt = ""
            self._last_text = ""
            self._last_result = {
                "transcription": "", "confidence": 0.0, "word_count": 0, "words": [],
                "language": self._last_result.get("language", ""), "buffering": False,
                "silence": True, "media_ts_start": round(buffer_start, 3),
                "media_ts_end": round(buffer_start + buffer_dur, 3),
            }
            return self._last_result

        try:
            # Feeding the previous transcript back as initial_prompt can make
            # Whisper loop on noisy audio, so it is OFF unless STT_USE_PROMPT=1.
            result = dict(self._transcribe(combined, buffer_start, self._last_prompt))
            new_text = result.get("transcription", "")
            if is_hallucination(new_text, self._extra) or _norm_text(new_text) == self._last_text:
                result.update({"transcription": "", "words": [], "word_count": 0})
            else:
                self._last_text = _norm_text(new_text)
                self._last_prompt = (self._last_prompt + " " + new_text).strip()[-200:]
            result["media_ts_start"] = round(buffer_start, 3)
            result["media_ts_end"] = round(buffer_start + buffer_dur, 3)
            self._last_result = {**result, "buffering": False}
        except Exception as exc:  # noqa: BLE001
            logger.warning("STT predict error [%s]: %s", self.task_name, exc)
            self._last_result = {**self._last_result, "buffering": False, "error": str(exc)}
        return self._last_result

    def predict_clip(self, audio: np.ndarray) -> dict:
        result = dict(self._transcribe(_mono(audio), 0.0))
        result.setdefault("media_ts_start", 0.0)
        result.setdefault("media_ts_end", round(len(audio) / SAMPLE_RATE, 3))
        return result

    def describe(self) -> dict:
        backend = getattr(self.task, "_backend", None)
        return {"task": self.task_name, "type": "transcription",
                "model": getattr(self.cfg, "model_name", None)
                or getattr(backend, "model_name", None),
                "buffer_seconds": self._buffer_seconds}


# ── Pretrained event detection (YAMNet-style) ────────────────────────────────


class PretrainedEventWrapper(ServeWrapper):
    """Buffers chunks to ``buffer_seconds``, then runs a pretrained event tagger.

    The task's ``predict(audio, sample_rate=..., source=...)`` should return a
    dict with ``label``/``confidence``/``scores`` and optionally
    ``primary_event``/``primary_event_conf``/``top_classes``.
    """

    kind = "event"
    WRAPPER_TYPE = "nonverbal_pretrained"

    def __init__(self, task, task_name: str, buffer_seconds: float = 2.0, cfg=None):
        super().__init__(task, task_name, cfg)
        self.class_names: list = []
        self.input_shape: list = []
        self._buffer_seconds = buffer_seconds
        self._target_samples = int(SAMPLE_RATE * buffer_seconds)
        self.reset()

    def reset(self) -> None:
        self._buffer: collections.deque = collections.deque()
        self._buffer_samples = 0
        self._pending_start: Optional[float] = None
        self._last_result = {
            "label": "unknown", "confidence": 0.0, "scores": {},
            "primary_event": "unknown", "primary_event_conf": 0.0,
            "top_classes": [], "duration_sec": 0.0, "inference_time_ms": 0.0,
            "buffering": True, "media_ts_start": 0.0, "media_ts_end": 0.0,
            "event_media_ts": 0.0,
        }

    def _run(self, audio: np.ndarray, buffer_start: float) -> dict:
        t0 = time.time()
        result = dict(self.task.predict(audio, sample_rate=SAMPLE_RATE, source="stream"))
        event_offset = 0.0
        yr = result.pop("_yamnet_result", None)  # not JSON-serialisable
        if yr is not None and getattr(yr, "frames", None):
            best = max(yr.frames, key=lambda f: f.primary_event_conf)
            event_offset = float(best.offset_sec)
        result.setdefault("inference_time_ms", round((time.time() - t0) * 1000, 1))
        dur = len(audio) / SAMPLE_RATE
        result["media_ts_start"] = round(buffer_start, 3)
        result["media_ts_end"] = round(buffer_start + dur, 3)
        result["event_media_ts"] = round(buffer_start + event_offset, 3)
        return result

    def predict(self, audio: np.ndarray, media_t0: float = 0.0) -> dict:
        audio = _mono(audio)
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
        try:
            self._last_result = {**self._run(combined, buffer_start), "buffering": False}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Event predict error [%s]: %s", self.task_name, exc)
            self._last_result = {**self._last_result, "buffering": False, "error": str(exc)}
        return self._last_result

    def predict_clip(self, audio: np.ndarray) -> dict:
        return self._run(_mono(audio), 0.0)

    def describe(self) -> dict:
        return {"task": self.task_name, "type": "event",
                "buffer_seconds": self._buffer_seconds}


#: Backwards-compatible alias.
YamnetModelWrapper = PretrainedEventWrapper


# ── Built-in factories ───────────────────────────────────────────────────────


def find_local_checkpoint(task_name: str, cfg, models_dir: str = "models") -> Optional[str]:
    """Look for a trained checkpoint in the usual places (first hit wins)."""
    from audio_ops_dagster.core.runtime import default_checkpoint_path

    candidates = [
        default_checkpoint_path(cfg),
        os.path.join("storage", task_name, "checkpoints", "best_model.pt"),
        os.path.join(models_dir, task_name, "best_model.pt"),
        os.path.join(models_dir, f"{task_name}.pt"),
    ]
    return next((c for c in candidates if os.path.exists(c)), None)


def download_checkpoint(task_name: str, cfg, models_dir: str = "models") -> Optional[str]:
    """If object storage is enabled, pull the task's checkpoint into ``models_dir``."""
    if os.environ.get("STORAGE_BACKEND", "local").lower() not in ("minio", "s3"):
        return None
    try:
        from audio_ops_dagster.core.storage import ArtifactType, S3StorageBackend

        backend = S3StorageBackend.from_env()
        if not backend.enabled:
            return None
        prefix = cfg.effective_bucket_prefix() if hasattr(cfg, "effective_bucket_prefix") \
            else task_name
        return str(backend.download_task_file(prefix, ArtifactType.MODELS, "best_model.pt",
                                              os.path.join(models_dir, task_name)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Storage checkpoint fetch failed for %s: %s", task_name, exc)
        return None


@register_wrapper("classifier")
def _make_classifier_wrapper(task, task_name: str, cfg):
    """Build the network from the task's own ``build_model()`` and load weights."""
    from audio_ops_dagster.core.runtime import CheckpointError, load_task_model, resolve_device

    models_dir = os.environ.get("MODELS_DIR", "models")
    ckpt = find_local_checkpoint(task_name, cfg, models_dir) or download_checkpoint(
        task_name, cfg, models_dir)
    if not ckpt:
        logger.info("No checkpoint for '%s' yet — train it, then it will serve.", task_name)
        return None
    device = resolve_device()
    try:
        model, meta = load_task_model(task, ckpt, device=device)
    except CheckpointError as exc:
        logger.error("Not serving '%s': %s", task_name, exc)
        return None
    trained_feat = (meta.get("feature_config") or {}).get("feature_type")
    if trained_feat and trained_feat != getattr(cfg, "feature_type", trained_feat):
        logger.warning("'%s': checkpoint was trained with feature_type=%s but config says %s.",
                       task_name, trained_feat, cfg.feature_type)
    logger.info("Loaded checkpoint for '%s' from %s", task_name, ckpt)
    return ModelWrapper(model, task_name, task=task, cfg=cfg, device=device)


@register_wrapper("transcription")
def _make_transcription_wrapper(task, task_name: str, cfg):
    return STTModelWrapper(
        task, task_name,
        buffer_seconds=float(getattr(cfg, "stream_buffer_seconds", None) or 5.0),
        silence_rms=float(getattr(cfg, "silence_rms", None) or 0.006),
        use_prompt=os.environ.get("STT_USE_PROMPT", "0") == "1",
        extra_hallucinations=[p for p in os.environ.get("STT_HALLUCINATION_PHRASES", "").split(",")
                              if p.strip()],
        cfg=cfg,
    )


@register_wrapper("pretrained_event")
def _make_pretrained_event_wrapper(task, task_name: str, cfg):
    env = os.environ.get("YAMNET_BUFFER_S")
    buffer_s = float(env) if env else float(getattr(cfg, "stream_buffer_seconds", None) or 2.0)
    return PretrainedEventWrapper(task, task_name, buffer_s, cfg=cfg)
