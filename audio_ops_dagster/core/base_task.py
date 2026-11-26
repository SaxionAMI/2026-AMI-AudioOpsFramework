"""
AudioTask — the task contract.

A task is a model plus four methods. The framework calls them; you never edit
the pipeline:

  ``build_model(input_shape)``            → an ``nn.Module``
  ``compute_loss(model, batch, device)``  → loss tensor, or ``(loss, outputs)``
  ``compute_metrics(y_true, y_pred)``     → ``{"metric": value}``
  ``postprocess(raw_output)``             → a JSON-friendly prediction dict

``compute_loss`` is used for **both** training and validation, so a
regression or multi-label task trains correctly with its own loss. Returning
``(loss, outputs)`` lets the trainer report accuracy without a second forward
pass; returning just the loss also works.

Optional hooks (all have working defaults):

  ``format_influx_record`` — time-series schema for batch-inference results
  ``grafana_panel_hints``  — dashboard hints for your own provisioning
  ``postprocess_with_llm`` — LLM refinement of a prediction
  ``predict(audio, ...)``  — serve-only / transcription tasks that wrap their
                             own runtime (faster-whisper, YAMNet, …)
  ``warmup()``             — load heavy weights before a batch run
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from audio_ops_dagster.core.runtime import resolve_device

TASK_TYPES = ("classification", "regression", "detection", "transcription")


def effective_task_type(task: Any) -> str:
    """The task's type: an explicit ``TASK_TYPE`` on the class wins, else config."""
    for klass in type(task).__mro__:
        if klass is AudioTask:
            break
        if "TASK_TYPE" in vars(klass):
            return str(vars(klass)["TASK_TYPE"])
    cfg = getattr(task, "config", None)
    return str(getattr(cfg, "task_type", None) or AudioTask.TASK_TYPE)


class AudioTask(ABC):
    """Abstract base class for audio tasks. See the module docstring."""

    #: "classification" | "regression" | "detection" | "transcription"
    TASK_TYPE = "classification"

    def __init__(self, config):
        self.config = config
        self.device = resolve_device()
        self._loss_fn: Optional[nn.Module] = None

    @property
    def task_type(self) -> str:
        return effective_task_type(self)

    # ── Required interface ───────────────────────────────────────────────────

    @abstractmethod
    def build_model(self, input_shape: Optional[Tuple[int, ...]]) -> nn.Module:
        """Return the task's ``nn.Module``."""

    @abstractmethod
    def compute_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Return a dict of metrics (accuracy, f1, mae, wer, …)."""

    @abstractmethod
    def postprocess(self, raw_output: torch.Tensor) -> Any:
        """Turn raw model output for one clip (a batch of one) into a JSON-friendly dict."""

    @abstractmethod
    def compute_loss(self, model: nn.Module, batch, device):
        """Loss for one batch: a tensor, or ``(loss, outputs)``."""

    # ── Loss / optimizer defaults (driven by config.yaml) ────────────────────

    def get_loss_function(self) -> nn.Module:
        """Loss module from ``criterion`` (+ ``class_weights``, ``label_smoothing``).

        Without a ``criterion`` the task type decides: cross-entropy for
        classification, MSE for regression, BCE-with-logits for detection.

        Cached, so calling it from ``compute_loss`` every batch is cheap.
        """
        if self._loss_fn is not None:
            return self._loss_fn
        criterion = getattr(self.config, "criterion", None) or {
            "regression": "mse", "detection": "bce"}.get(self.task_type, "cross_entropy")
        weights = getattr(self.config, "class_weights", None)
        weight_t = torch.tensor(weights, dtype=torch.float32) if weights else None
        if criterion == "cross_entropy":
            fn: nn.Module = nn.CrossEntropyLoss(
                weight=weight_t, label_smoothing=float(getattr(self.config, "label_smoothing", 0.0))
            )
        elif criterion == "mse":
            fn = nn.MSELoss()
        elif criterion in ("l1", "mae"):
            fn = nn.L1Loss()
        elif criterion in ("bce", "bce_with_logits"):
            fn = nn.BCEWithLogitsLoss(pos_weight=weight_t)
        elif criterion == "ctc":
            fn = nn.CTCLoss(zero_infinity=True)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        self._loss_fn = fn.to(self.device)
        return self._loss_fn

    def get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Optimizer from ``optimizer``, ``learning_rate`` and ``weight_decay``."""
        name = str(getattr(self.config, "optimizer", "adamw")).lower()
        lr = float(getattr(self.config, "learning_rate", 1e-3))
        wd = float(getattr(self.config, "weight_decay", 0.0) or 0.0)
        params = [p for p in model.parameters() if p.requires_grad]
        if name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        if name == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        raise ValueError(f"Unknown optimizer: {name}")

    # ── Batch / streaming hooks ──────────────────────────────────────────────

    def warmup(self) -> None:  # noqa: B027 — optional hook, intentionally empty
        """Load heavy weights before a batch run. Default: nothing to do."""

    def transcribe(self, audio, t_offset: float = 0.0) -> Dict[str, Any]:
        """Transcribe a waveform (transcription tasks).

        Delegates to ``self.predict(audio, t_offset=...)`` if the task defines
        it, else to a ``self._backend.transcribe`` runtime (older tasks).
        """
        if callable(getattr(self, "predict", None)):
            return self.predict(audio, t_offset=t_offset)  # type: ignore[attr-defined]
        backend = getattr(self, "_backend", None)
        if backend is not None and hasattr(backend, "transcribe"):
            return backend.transcribe(audio, t_offset=t_offset)
        raise NotImplementedError(
            f"{type(self).__name__} is a transcription task but defines neither "
            f"predict(audio, t_offset=...) nor a _backend.transcribe runtime."
        )

    # ── InfluxDB integration ─────────────────────────────────────────────────

    def format_influx_record(
        self,
        segment_id: str,
        prediction: Any,
        audio_offset_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Convert one inference prediction into ``{"fields": {...}, "tags": {...}}``.

        The default stores scalar prediction fields (or a ``result`` string).
        Override for a task-specific schema.
        """
        fields: Dict[str, Any] = {}
        if isinstance(prediction, dict):
            fields = {
                k: v for k, v in prediction.items() if isinstance(v, (str, int, float, bool))
            }
        if not fields:
            fields = {"result": str(prediction)}
        if audio_offset_sec is not None:
            fields["audio_offset_sec"] = audio_offset_sec
        return {"fields": fields, "tags": {}}

    def grafana_panel_hints(self) -> List[Dict[str, Any]]:
        """
        Hints for generating dashboard panels from the time-series schema.

        Each hint: ``panel_type`` (timeseries|stat|bar|table|gauge), ``title``,
        ``field``, ``description``, ``agg``. No dashboards ship with the
        framework; this is metadata for your own provisioning.
        """
        return [
            {
                "panel_type": "stat",
                "title": f"{self.config.task_name} — Total Inferences",
                "field": "result",
                "agg": "count",
                "description": f"Total inference runs for {self.config.task_name}",
            }
        ]

    # ── LLM post-processing hook ─────────────────────────────────────────────

    def postprocess_with_llm(
        self,
        raw_result: Any,
        context: Optional[Dict[str, Any]] = None,
        llm_client: Optional[Any] = None,
    ) -> Any:
        """Optional LLM refinement of a ``postprocess()`` result. Default: unchanged."""
        return raw_result

    # ── Inference helpers ────────────────────────────────────────────────────

    def preprocess_for_inference(self, audio_array, sample_rate: int = 16000):
        """Prepare raw audio for model input. Default: unchanged."""
        return audio_array
