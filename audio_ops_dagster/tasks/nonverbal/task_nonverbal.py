"""
Non-Verbal Audio Event Detection Task.

Detects non-speech vocalizations and sounds in audio:
  • Coughs, clearing throat
  • Pain vocalizations (groans, moans, cries)
  • Laughter
  • Sighs, yawns
  • Breathing anomalies (heavy breathing, wheezing)
  • Silence / pauses

Uses a CNN classifier on mel spectrograms, similar to sentiment but with
event-specific class labels and temporal localization.

This is a multi-label detection task — a single audio segment can contain
multiple overlapping events (e.g. cough during heavy breathing).
"""

from typing import Dict, Tuple, Any, List, Optional
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from audio_ops_dagster.core.base_task import AudioTask
from audio_ops_dagster.registry import register_task


@register_task("nonverbal")
class NonVerbalTask(AudioTask):
    """
    Non-verbal audio event detection from spectrograms.

    Default classes: cough, pain, laughter, sigh, breathing, silence, speech, other
    Override via config.yaml class_labels.
    """

    TASK_TYPE = "detection"

    DEFAULT_CLASSES = [
        "cough",
        "pain",
        "laughter",
        "sigh",
        "breathing",
        "silence",
        "speech",
        "other",
    ]

    def __init__(self, config):
        super().__init__(config)
        if not self.config.class_labels:
            self.config.class_labels = self.DEFAULT_CLASSES
        if not self.config.num_classes:
            self.config.num_classes = len(self.config.class_labels)

    def build_model(self, input_shape: Tuple[int, ...]) -> nn.Module:
        """
        ResNet18 backbone with multi-label head.

        Uses sigmoid activation (not softmax) because a segment can contain
        multiple simultaneous events.
        """
        model = models.resnet18(weights=None)
        # Adapt for single-channel spectrograms
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
            nn.Dropout(p=0.4),
            nn.Linear(256, self.config.num_classes),
            # No sigmoid here — applied in loss (BCEWithLogitsLoss) and postprocess
        )
        return model

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Multi-label metrics using threshold of 0.5."""
        if y_pred.ndim == 2 and y_true.ndim == 1:
            # Single-label fallback
            y_pred_labels = y_pred.argmax(axis=1)
            return {
                "accuracy": float(accuracy_score(y_true, y_pred_labels)),
                "f1_weighted": float(
                    f1_score(y_true, y_pred_labels, average="weighted", zero_division=0)
                ),
            }

        # Multi-label: threshold sigmoid outputs
        y_pred_binary = (y_pred > 0.5).astype(int) if y_pred.max() <= 1.0 else (
            (1 / (1 + np.exp(-y_pred))) > 0.5
        ).astype(int)

        return {
            "f1_micro": float(f1_score(y_true, y_pred_binary, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred_binary, average="macro", zero_division=0)),
            "precision_micro": float(
                precision_score(y_true, y_pred_binary, average="micro", zero_division=0)
            ),
            "recall_micro": float(
                recall_score(y_true, y_pred_binary, average="micro", zero_division=0)
            ),
        }

    def postprocess(self, raw_output: torch.Tensor) -> Any:
        """
        Convert logits to detected events with confidence scores.

        Returns list of detected events (those above threshold).
        """
        if raw_output.ndim == 1:
            raw_output = raw_output.unsqueeze(0)

        probs = torch.sigmoid(raw_output)
        threshold = 0.5

        results = []
        for i in range(probs.shape[0]):
            events = []
            for j in range(probs.shape[1]):
                if probs[i, j] > threshold:
                    label = (
                        self.config.class_labels[j]
                        if j < len(self.config.class_labels)
                        else f"class_{j}"
                    )
                    events.append({
                        "event": label,
                        "confidence": float(probs[i, j].item()),
                    })
            # Sort by confidence descending
            events.sort(key=lambda e: e["confidence"], reverse=True)
            results.append({
                "detected_events": events,
                "primary_event": events[0]["event"] if events else "silence",
                "num_events": len(events),
            })

        return results[0] if len(results) == 1 else results

    def get_loss_function(self) -> nn.Module:
        """BCE with logits for multi-label classification."""
        return nn.BCEWithLogitsLoss()

    def compute_loss(self, model, batch, device):
        x, y = batch
        x = x.to(device)
        # For multi-label, y should be float tensor of shape (batch, num_classes)
        if isinstance(y, torch.Tensor) and y.dtype in (torch.int64, torch.long):
            # One-hot encode if single labels
            y_onehot = torch.zeros(y.size(0), self.config.num_classes, device=device)
            y_onehot.scatter_(1, y.unsqueeze(1).to(device), 1.0)
            y = y_onehot
        else:
            y = y.float().to(device)
        logits = model(x)
        return self.get_loss_function()(logits, y)

    # ── InfluxDB integration ─────────────────────────────────────────────────

    def format_influx_record(
        self,
        segment_id: str,
        prediction: Any,
        audio_offset_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Non-verbal → InfluxDB fields:
          primary_event  (string)  — highest-confidence detected event
          num_events     (int)     — number of events above threshold
          conf_*         (float)   — per-event confidence scores
        """
        if isinstance(prediction, list):
            prediction = prediction[0]

        fields: Dict[str, Any] = {
            "primary_event": prediction.get("primary_event", "silence"),
            "num_events": prediction.get("num_events", 0),
        }
        for evt in prediction.get("detected_events", []):
            fields[f"conf_{evt['event']}"] = evt["confidence"]

        if audio_offset_sec is not None:
            fields["audio_offset_sec"] = audio_offset_sec

        return {
            "fields": fields,
            "tags": {"primary_event": prediction.get("primary_event", "silence")},
        }

    def grafana_panel_hints(self) -> List[Dict[str, Any]]:
        panels = [
            {
                "panel_type": "timeseries",
                "title": "Non-Verbal Events Timeline",
                "field": "primary_event",
                "agg": "count",
                "group_by": "primary_event",
                "description": "Detected non-verbal events over the audio timeline",
            },
            {
                "panel_type": "piechart",
                "title": "Event Type Distribution",
                "field": "primary_event",
                "agg": "count",
                "description": "Distribution of detected non-verbal event types",
            },
            {
                "panel_type": "stat",
                "title": "Total Events Detected",
                "field": "num_events",
                "agg": "sum",
                "description": "Total non-verbal events detected",
            },
            {
                "panel_type": "table",
                "title": "Recent Detections",
                "field": "primary_event",
                "agg": "last",
                "description": "Most recent non-verbal event detections with timestamps",
            },
        ]
        # Per-event confidence gauges
        for label in (self.config.class_labels or self.DEFAULT_CLASSES)[:5]:
            panels.append({
                "panel_type": "gauge",
                "title": f"{label.title()} Confidence",
                "field": f"conf_{label}",
                "agg": "mean",
                "description": f"Average detection confidence for '{label}' events",
            })
        return panels

    # ── LLM post-processing ──────────────────────────────────────────────────

    def postprocess_with_llm(self, raw_result, context=None, llm_client=None):
        """Interpret non-verbal detections in clinical/conversational context."""
        if llm_client is None:
            return raw_result

        pred = raw_result if isinstance(raw_result, dict) else raw_result[0]
        events = [e["event"] for e in pred.get("detected_events", [])]
        if events:
            interpretation = llm_client.complete(
                f"Briefly interpret these non-verbal audio events in a "
                f"healthcare monitoring context: {', '.join(events)}",
                system="You are a medical audio analysis assistant. Be concise and clinical.",
            )
            if interpretation:
                pred["clinical_interpretation"] = interpretation
        return raw_result
