"""
task_types.py — ready-made task bases for the common cases.

Each base implements the whole task contract with sensible defaults, so a task
only overrides what is different. The defaults use :class:`SmallAudioCNN`, the
config's ``criterion``/``class_weights``/``label_smoothing`` and the standard
metrics for the type::

    from audio_ops import ClassificationTask, register_task

    @register_task("wake_word")
    class WakeWord(ClassificationTask):
        pass                         # or override build_model / compute_metrics / …

A task folder with only a ``config.yaml`` (no Python file) gets one of these
bases automatically, chosen by ``task_type``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from audio_ops_dagster.core.base_task import AudioTask
from audio_ops_dagster.core.models import SmallAudioCNN


class _LabelledTask(AudioTask):
    @property
    def labels(self) -> List[str]:
        """Class names in output order (``class_labels``, else ``0..n-1``)."""
        if self.config.class_labels:
            return list(self.config.class_labels)
        return [str(i) for i in range(self.num_classes)]

    @property
    def num_classes(self) -> int:
        n = self.config.num_classes or (len(self.config.class_labels)
                                        if self.config.class_labels else None)
        if not n:
            raise ValueError(
                f"{self.config.task_name}: number of classes unknown. Set class_labels in "
                f"config.yaml (or train once — they are then read from the checkpoint).")
        return int(n)

    def build_model(self, input_shape) -> nn.Module:
        return SmallAudioCNN(out_dim=self.num_classes)


class ClassificationTask(_LabelledTask):
    """Single-label classification: one softmax over ``class_labels``."""

    TASK_TYPE = "classification"

    def compute_loss(self, model, batch, device):
        x, y = batch
        logits = model(x.to(device))
        return self.get_loss_function()(logits, y.to(device)), logits

    def compute_metrics(self, y_true, y_pred) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score

        pred = y_pred.argmax(axis=1) if np.ndim(y_pred) == 2 else np.asarray(y_pred)
        return {"accuracy": float(accuracy_score(y_true, pred)),
                "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0))}

    def postprocess(self, raw_output) -> Dict[str, Any]:
        probs = torch.softmax(raw_output.float(), dim=-1).reshape(-1).cpu()
        labels = self.labels
        idx = int(probs.argmax())
        return {"label": labels[idx] if idx < len(labels) else str(idx),
                "confidence": float(probs[idx]),
                "scores": {labels[i]: float(p) for i, p in enumerate(probs.tolist())
                           if i < len(labels)}}


class DetectionTask(_LabelledTask):
    """Multi-label detection: one sigmoid per class, threshold ``detection_threshold``
    (config, default 0.5). Folder labels give one positive class per clip."""

    TASK_TYPE = "detection"

    @property
    def threshold(self) -> float:
        return float(getattr(self.config, "detection_threshold", 0.5))

    def compute_loss(self, model, batch, device):
        x, y = batch
        logits = model(x.to(device))
        y = y.to(device)
        if y.dim() == 1:                                  # class index → multi-hot
            y = nn.functional.one_hot(y.long(), num_classes=logits.shape[-1])
        return self.get_loss_function()(logits, y.float()), None

    def compute_metrics(self, y_true, y_pred) -> Dict[str, float]:
        y_true = np.asarray(y_true)
        if y_true.ndim == 1:
            y_true = np.eye(y_pred.shape[1], dtype=int)[y_true.astype(int)]
        pred = (1 / (1 + np.exp(-y_pred)) > self.threshold).astype(int)
        return {"exact_match": float((pred == y_true).all(axis=1).mean()),
                "label_accuracy": float((pred == y_true).mean())}

    def postprocess(self, raw_output) -> Dict[str, Any]:
        probs = torch.sigmoid(raw_output.float()).reshape(-1).cpu().tolist()
        labels = self.labels
        return {"labels": [labels[i] for i, p in enumerate(probs) if p > self.threshold],
                "scores": {labels[i]: float(p) for i, p in enumerate(probs)}}


class RegressionTask(AudioTask):
    """Scalar regression. Targets come from numeric label folders
    (``audio_input/<task>/0.75/clip.wav``)."""

    TASK_TYPE = "regression"

    def build_model(self, input_shape) -> nn.Module:
        return SmallAudioCNN(out_dim=1)

    def compute_loss(self, model, batch, device):
        x, y = batch
        pred = model(x.to(device)).view(-1)
        return self.get_loss_function()(pred, y.to(device).float().view(-1)), pred

    def compute_metrics(self, y_true, y_pred) -> Dict[str, float]:
        yp = np.asarray(y_pred, dtype=float).reshape(-1)
        yt = np.asarray(y_true, dtype=float).reshape(-1)
        return {"mae": float(np.mean(np.abs(yp - yt))),
                "rmse": float(np.sqrt(np.mean((yp - yt) ** 2)))}

    def postprocess(self, raw_output) -> Dict[str, Any]:
        return {"value": float(raw_output.reshape(-1)[0])}


#: Base used for a config-only task folder, by ``task_type``.
BASES_BY_TYPE = {
    "classification": ClassificationTask,
    "detection": DetectionTask,
    "regression": RegressionTask,
}
