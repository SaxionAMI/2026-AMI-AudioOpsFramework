"""
settings.py — every environment variable the inference service reads, in one place.

All values have working defaults; see ``env.example`` and the docs'
configuration reference. ``Settings.from_env()`` is evaluated once at import of
``audio_ops_inference.api``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


def _bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _list(name: str, default: str = "") -> List[str]:
    return [p.strip() for p in os.environ.get(name, default).split(",") if p.strip()]


@dataclass
class Settings:
    # ── Audio / streaming ────────────────────────────────────────────────────
    sample_rate: int = 16_000
    chunk_ms: int = 1000
    proc_queue_max: int = 5
    stt_queue_max: int = 12
    auto_rtsp_streams: List[str] = field(default_factory=list)

    # ── Models ───────────────────────────────────────────────────────────────
    models_dir: str = "models"
    model_cache_dir: str = "models"
    storage_backend: str = "local"

    # ── Security / limits ────────────────────────────────────────────────────
    api_key: Optional[str] = None
    cors_allow_origins: List[str] = field(default_factory=lambda: ["*"])
    max_upload_mb: float = 100.0

    # ── Optional features ────────────────────────────────────────────────────
    fusion_enabled: bool = False
    fusion_emit_every: int = 1
    fusion_emit_window_s: float = 120.0
    fusion_llm_window_s: float = 60.0
    nonverbal_min_conf: float = 0.05
    nonverbal_background: List[str] = field(
        default_factory=lambda: ["speech", "silence", "unknown"])
    demo_audio_path: str = "/test_audio/demo.wav"

    # ── Stream auto-discovery (MediaMTX) ─────────────────────────────────────
    autodiscover: bool = False
    mediamtx_api_url: str = "http://mediamtx:9997"
    rtsp_base_url: str = "rtsp://mediamtx:8554"
    autodiscover_interval: float = 5.0
    autodiscover_filter: str = ""

    # ── InfluxDB (optional) ──────────────────────────────────────────────────
    influx_enabled: bool = False
    influx_url: str = "http://influxdb:8086"
    influx_token: str = ""
    influx_org: str = "audioops"
    influx_bucket: str = "audio_metrics"

    # ── STT behaviour ────────────────────────────────────────────────────────
    stt_use_prompt: bool = False
    stt_extra_hallucinations: List[str] = field(default_factory=list)
    event_buffer_s: Optional[float] = None

    @classmethod
    def from_env(cls) -> Settings:
        cache = os.environ.get("MODEL_CACHE_DIR", "models")
        token = os.environ.get("INFLUXDB_TOKEN", "")
        influx_flag = os.environ.get("ENABLE_INFLUXDB")
        return cls(
            sample_rate=16_000,
            chunk_ms=int(os.environ.get("CHUNK_MS", "1000")),
            proc_queue_max=int(os.environ.get("PROC_QUEUE_MAX", "5")),
            stt_queue_max=int(os.environ.get("STT_QUEUE_MAX", "12")),
            auto_rtsp_streams=_list("AUTO_RTSP_STREAMS"),
            models_dir=os.environ.get("MODELS_DIR", "models"),
            model_cache_dir=cache,
            storage_backend=os.environ.get("STORAGE_BACKEND", "local").lower(),
            api_key=(os.environ.get("API_KEY") or os.environ.get("API_SECRET_KEY") or None),
            cors_allow_origins=_list("CORS_ALLOW_ORIGINS", "*") or ["*"],
            max_upload_mb=float(os.environ.get("MAX_UPLOAD_MB", "100")),
            fusion_enabled=_bool("ENABLE_FUSION"),
            fusion_emit_every=int(os.environ.get("FUSION_EMIT_EVERY", "1")),
            fusion_emit_window_s=float(os.environ.get("FUSION_EMIT_WINDOW_S", "120")),
            fusion_llm_window_s=float(os.environ.get("FUSION_LLM_WINDOW_S", "60")),
            nonverbal_min_conf=float(os.environ.get("NONVERBAL_MIN_CONF", "0.05")),
            nonverbal_background=_list("NONVERBAL_BACKGROUND", "speech,silence,unknown"),
            demo_audio_path=os.environ.get("DEMO_AUDIO_PATH", "/test_audio/demo.wav"),
            autodiscover=_bool("STREAM_AUTODISCOVER"),
            mediamtx_api_url=os.environ.get("MEDIAMTX_API_URL",
                                            os.environ.get("MEDIAMTX_API", "http://mediamtx:9997")),
            rtsp_base_url=os.environ.get("RTSP_BASE_URL", "rtsp://mediamtx:8554").rstrip("/"),
            autodiscover_interval=float(os.environ.get("STREAM_AUTODISCOVER_INTERVAL", "5")),
            autodiscover_filter=os.environ.get("STREAM_AUTODISCOVER_FILTER", ""),
            # Back-compat: a token alone used to switch writes on.
            influx_enabled=(_bool("ENABLE_INFLUXDB") if influx_flag is not None else bool(token)),
            influx_url=os.environ.get("INFLUXDB_URL", "http://influxdb:8086"),
            influx_token=token,
            influx_org=os.environ.get("INFLUXDB_ORG", "audioops"),
            influx_bucket=os.environ.get("INFLUXDB_BUCKET", "audio_metrics"),
            stt_use_prompt=os.environ.get("STT_USE_PROMPT", "0") == "1",
            stt_extra_hallucinations=_list("STT_HALLUCINATION_PHRASES"),
            event_buffer_s=(float(os.environ["YAMNET_BUFFER_S"])
                            if os.environ.get("YAMNET_BUFFER_S") else None),
        )


def configure_model_caches(cache_dir: str) -> None:
    """Point HuggingFace / TF-Hub caches at one persistent directory.

    Must run before torch/transformers/tensorflow are imported (they read these
    at import time). Mount ``MODEL_CACHE_DIR`` as a volume so downloaded weights
    survive restarts; after a one-time prefetch, set ``HF_HUB_OFFLINE=1`` to
    forbid runtime downloads.
    """
    os.environ.setdefault("HF_HOME", os.path.join(cache_dir, "hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(cache_dir, "hf"))
    os.environ.setdefault("TFHUB_CACHE_DIR", os.path.join(cache_dir, "tfhub"))
    for sub in ("hf", "tfhub"):
        try:
            os.makedirs(os.path.join(cache_dir, sub), exist_ok=True)
        except OSError:
            pass
