"""
Sentiment classification task — updated with InfluxDB + Grafana integration.
"""

from typing import Dict, Tuple, Any, List, Optional
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from audio_ops_dagster.core.base_task import AudioTask
from audio_ops_dagster.registry import register_task


@register_task("sentiment")
class SentimentTask(AudioTask):
    """Sentiment classification from audio spectrograms."""

    TASK_TYPE = "classification"

    def build_model(self, input_shape: Tuple[int, ...]) -> nn.Module:
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=model.conv1.bias is not None,
        )
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, self.config.num_classes),
        )
        return model

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_pred_labels = y_pred.argmax(axis=1) if y_pred.ndim == 2 else y_pred
        return {
            "accuracy": float(accuracy_score(y_true, y_pred_labels)),
            "f1_weighted": float(f1_score(y_true, y_pred_labels, average="weighted", zero_division=0)),
        }

    def postprocess(self, raw_output: torch.Tensor) -> Any:
        if raw_output.ndim == 1:
            raw_output = raw_output.unsqueeze(0)
        probs = torch.softmax(raw_output, dim=1)
        indices = probs.argmax(dim=1)

        results = []
        for i, idx in enumerate(indices):
            label = (
                self.config.class_labels[idx.item()]
                if self.config.class_labels and idx.item() < len(self.config.class_labels)
                else "unknown"
            )
            results.append({
                "label": label,
                "confidence": float(probs[i, idx].item()),
                "probabilities": {
                    self.config.class_labels[j]: float(probs[i, j].item())
                    for j in range(probs.shape[1])
                    if self.config.class_labels and j < len(self.config.class_labels)
                },
            })
        return results[0] if len(results) == 1 else results

    def get_loss_function(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def compute_loss(self, model, batch, device):
        x, y = batch
        x, y = x.to(device), y.to(device)
        return self.get_loss_function()(model(x), y)

    # ── InfluxDB integration ─────────────────────────────────────────────────

    def format_influx_record(
        self,
        segment_id: str,
        prediction: Any,
        audio_offset_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Sentiment → InfluxDB fields:
          label       (string)  — predicted class
          confidence  (float)   — max softmax probability
          prob_*      (float)   — per-class probabilities
        """
        if isinstance(prediction, list):
            prediction = prediction[0]

        fields: Dict[str, Any] = {
            "label": prediction.get("label", "unknown"),
            "confidence": prediction.get("confidence", 0.0),
        }
        # Add per-class probabilities as separate fields for Grafana
        for cls_name, prob in prediction.get("probabilities", {}).items():
            fields[f"prob_{cls_name}"] = prob

        if audio_offset_sec is not None:
            fields["audio_offset_sec"] = audio_offset_sec

        return {"fields": fields, "tags": {"predicted_label": prediction.get("label", "unknown")}}

    def grafana_panel_hints(self) -> List[Dict[str, Any]]:
        labels = self.config.class_labels or ["positive", "negative", "neutral"]
        panels = [
            {
                "panel_type": "timeseries",
                "title": "Sentiment Over Time",
                "field": "confidence",
                "group_by": "predicted_label",
                "description": "Per-segment sentiment confidence over the audio timeline",
                "agg": "mean",
            },
            {
                "panel_type": "piechart",
                "title": "Sentiment Distribution",
                "field": "label",
                "agg": "count",
                "description": "Distribution of predicted sentiment labels",
            },
            {
                "panel_type": "stat",
                "title": "Total Segments Analyzed",
                "field": "confidence",
                "agg": "count",
                "description": "Total audio segments processed",
            },
            {
                "panel_type": "gauge",
                "title": "Average Confidence",
                "field": "confidence",
                "agg": "mean",
                "description": "Mean prediction confidence across all segments",
            },
        ]
        # Per-class probability timeseries
        for label in labels:
            panels.append({
                "panel_type": "timeseries",
                "title": f"P({label}) Over Time",
                "field": f"prob_{label}",
                "agg": "mean",
                "description": f"Probability of '{label}' sentiment over time",
            })
        return panels

    # ── LLM post-processing ──────────────────────────────────────────────────

    def postprocess_with_llm(self, raw_result, context=None, llm_client=None):
        if llm_client is None:
            return raw_result
        # Add a natural-language explanation of the sentiment prediction
        pred = raw_result if isinstance(raw_result, dict) else raw_result[0]
        explanation = llm_client.complete(
            f"In one sentence, explain what a '{pred['label']}' sentiment "
            f"(confidence {pred['confidence']:.0%}) in audio means.",
            system="You are an audio analysis assistant. Be very concise.",
        )
        if explanation:
            pred["explanation"] = explanation
        return raw_result
