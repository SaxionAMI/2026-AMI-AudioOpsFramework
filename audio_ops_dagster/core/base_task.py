"""
AudioTask — Abstract base for all audio processing tasks.

v2 additions over original:
  • influx_schema()       — each task declares what fields it writes to InfluxDB
  • format_influx_record()— turns a raw inference output into an InfluxDB-ready dict
  • postprocess_with_llm()— optional hook for LLM-based refinement
  • grafana_panel_hints() — tells the auto-provisioner what dashboard panels to create
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class AudioTask(ABC):
    """
    Abstract base class for audio processing tasks.

    Every task must implement:
      build_model, compute_metrics, postprocess, compute_loss

    Every task *should* implement (for full platform integration):
      format_influx_record  — how inference results become time-series points
      grafana_panel_hints   — what panels to auto-generate on the dashboard
    """

    TASK_TYPE = "classification"  # override in subclass: "classification", "transcription", "detection"

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Required interface ───────────────────────────────────────────────────

    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...]) -> nn.Module:
        """Return task-specific PyTorch nn.Module."""

    @abstractmethod
    def compute_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Return dict of metrics (e.g., accuracy, f1, wer)."""

    @abstractmethod
    def postprocess(self, raw_output: torch.Tensor) -> Any:
        """Format raw model output for end-users."""

    @abstractmethod
    def compute_loss(self, model: nn.Module, batch, device) -> torch.Tensor:
        """Compute loss for a single training batch."""

    # ── Loss / optimizer defaults ────────────────────────────────────────────

    def get_loss_function(self):
        if self.config.criterion == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif self.config.criterion == "mse":
            return nn.MSELoss()
        elif self.config.criterion == "ctc":
            return nn.CTCLoss(zero_infinity=True)
        raise ValueError(f"Unknown criterion: {self.config.criterion}")

    def get_optimizer(self, model: nn.Module):
        if self.config.optimizer == "adam":
            return torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(model.parameters(), lr=self.config.learning_rate, momentum=0.9)
        elif self.config.optimizer == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    # ── InfluxDB integration ─────────────────────────────────────────────────

    def format_influx_record(
        self,
        segment_id: str,
        prediction: Any,
        audio_offset_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Convert a single inference prediction into an InfluxDB-ready dict.

        Must return {"fields": {...}, "tags": {...}} where fields contain the
        actual data values and tags contain indexable metadata.

        Subclasses override this for task-specific schemas. The base
        implementation stores a generic 'result' string.
        """
        fields = {"result": str(prediction)}
        if audio_offset_sec is not None:
            fields["audio_offset_sec"] = audio_offset_sec
        return {"fields": fields, "tags": {}}

    def grafana_panel_hints(self) -> List[Dict[str, Any]]:
        """
        Return hints for auto-generating Grafana dashboard panels.

        Each hint is a dict with:
          panel_type:  "timeseries" | "stat" | "bar" | "table" | "gauge"
          title:       human-readable panel title
          field:       InfluxDB field name to query
          description: tooltip text
          agg:         aggregation function ("mean", "last", "count", etc.)

        Override in subclass for richer dashboards.
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
        """
        Optional LLM refinement of inference results.

        For STT: fix transcription errors, add punctuation, resolve ambiguity.
        For sentiment: provide natural-language explanation of the prediction.
        For nonverbal: interpret clinical significance.

        Parameters
        ----------
        raw_result  : the output of postprocess()
        context     : additional context (e.g. nonverbal detections, speaker info)
        llm_client  : an LLMClient instance (see llm_client.py)

        Returns the refined result, or raw_result unchanged if no LLM is configured.
        """
        return raw_result

    # ── Inference helpers ────────────────────────────────────────────────────

    def preprocess_for_inference(self, audio_array, sample_rate: int = 16000):
        """
        Prepare raw audio for model input. Override for tasks that need
        special preprocessing (e.g. Wav2Vec2 processor).

        Default: return audio as-is (assumes upstream feature extraction).
        """
        return audio_array
