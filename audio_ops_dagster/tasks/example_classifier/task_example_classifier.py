"""
example_classifier — the reference task.

Built on ``ClassificationTask``, which already implements the whole task
contract (loss from config.yaml, accuracy/F1 metrics, label/confidence/scores
output). This example overrides three things to show how customisation works:

* ``build_model``          — its own small CNN (any ``nn.Module`` works)
* ``compute_metrics``      — adds weighted F1 to the defaults
* ``format_influx_record`` — a custom time-series schema (used only if InfluxDB is on)

Simplest possible task: delete this file — a folder with only ``config.yaml``
uses ``ClassificationTask`` (with ``SmallAudioCNN``) as-is.
"""

from typing import Any, Dict, Optional

import torch.nn as nn

from audio_ops import ClassificationTask, register_task


class TinyCNN(nn.Module):
    """Two conv blocks over a (1, n_mels, time) spectrogram; any input size works."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 16, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@register_task("example_classifier")
class ExampleClassifierTask(ClassificationTask):
    def build_model(self, input_shape) -> nn.Module:
        return TinyCNN(num_classes=self.num_classes)

    def compute_metrics(self, y_true, y_pred) -> Dict[str, float]:
        from sklearn.metrics import f1_score

        metrics = super().compute_metrics(y_true, y_pred)        # accuracy, f1_macro
        metrics["f1_weighted"] = float(f1_score(y_true, y_pred.argmax(axis=1),
                                                average="weighted", zero_division=0))
        return metrics

    def format_influx_record(self, segment_id: str, prediction: Any,
                             audio_offset_sec: Optional[float] = None) -> Dict[str, Any]:
        fields = {"label": prediction.get("label", "unknown"),
                  "confidence": prediction.get("confidence", 0.0)}
        if audio_offset_sec is not None:
            fields["audio_offset_sec"] = audio_offset_sec
        return {"fields": fields, "tags": {"predicted_label": fields["label"]}}
