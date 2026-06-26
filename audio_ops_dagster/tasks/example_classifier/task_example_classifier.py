"""
example_classifier — the reference task for this framework.

It shows the entire task contract end to end: build a model, compute a loss,
turn raw logits into a friendly prediction, and report metrics. Copy this
folder (or use `python scripts/new_task.py <name>`) to start a new task.

The model is a small from-scratch CNN so the example trains with only `torch`
installed — no torchvision weight download. Swap in anything you like.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from audio_ops_dagster.core.base_task import AudioTask
from audio_ops_dagster.registry import register_task


class _SmallAudioCNN(nn.Module):
    """Tiny conv net over a (1, n_mels, time) log-mel spectrogram."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),  # shape-agnostic: any input size works
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@register_task("example_classifier")
class ExampleClassifierTask(AudioTask):
    """Multi-class audio classification from log-mel spectrograms."""

    TASK_TYPE = "classification"

    # ── Required interface ───────────────────────────────────────────────────

    def build_model(self, input_shape: Tuple[int, ...]) -> nn.Module:
        return _SmallAudioCNN(num_classes=self.config.num_classes)

    def compute_loss(self, model, batch, device):
        x, y = batch
        x, y = x.to(device), y.to(device)
        return nn.CrossEntropyLoss()(model(x), y)

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score
        y_pred_labels = y_pred.argmax(axis=1) if y_pred.ndim == 2 else y_pred
        return {
            "accuracy": float(accuracy_score(y_true, y_pred_labels)),
            "f1_weighted": float(f1_score(y_true, y_pred_labels, average="weighted", zero_division=0)),
        }

    def postprocess(self, raw_output: torch.Tensor) -> Any:
        if raw_output.ndim == 1:
            raw_output = raw_output.unsqueeze(0)
        probs = torch.softmax(raw_output, dim=1)
        idx = probs.argmax(dim=1)
        labels = self.config.class_labels or [str(i) for i in range(probs.shape[1])]

        out = []
        for i, j in enumerate(idx):
            j = j.item()
            out.append({
                "label": labels[j] if j < len(labels) else str(j),
                "confidence": float(probs[i, j].item()),
                "probabilities": {
                    labels[k]: float(probs[i, k].item())
                    for k in range(probs.shape[1]) if k < len(labels)
                },
            })
        return out[0] if len(out) == 1 else out

    # ── Optional: time-series schema (no-op unless InfluxDB is enabled) ──────

    def format_influx_record(
        self,
        segment_id: str,
        prediction: Any,
        audio_offset_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        if isinstance(prediction, list):
            prediction = prediction[0]
        fields: Dict[str, Any] = {
            "label": prediction.get("label", "unknown"),
            "confidence": prediction.get("confidence", 0.0),
        }
        if audio_offset_sec is not None:
            fields["audio_offset_sec"] = audio_offset_sec
        return {"fields": fields, "tags": {"predicted_label": prediction.get("label", "unknown")}}
