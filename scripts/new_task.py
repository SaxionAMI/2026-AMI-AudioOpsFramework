#!/usr/bin/env python3
"""
new_task.py — scaffold a new audio task of any type.

Usage:
    python scripts/new_task.py <name> [--type TYPE] [--archetype ARCH] [--feature FEAT]

    python scripts/new_task.py wake_word                          # classification
    python scripts/new_task.py loudness   --type regression
    python scripts/new_task.py tagging    --type detection --feature mfcc
    python scripts/new_task.py transcriber --type transcription   # serve-only

Types:      classification (default) | regression | detection | transcription
Archetypes: trainable_model (default) | pretrained_servable | pretrained_batch | eval_only
Features:   spectrogram (default) | mel | logmel | mfcc | raw  (or your own)

Auto-discovery picks the task up on the next `dagster dev` / API start.
See audio_ops_dagster/tasks/example_classifier/ for a complete reference.
"""

import argparse
import re
import sys
from pathlib import Path

ARCHETYPES = ("trainable_model", "pretrained_servable", "pretrained_batch", "eval_only")
TYPES = ("classification", "regression", "detection", "transcription")
TYPE_DEFAULTS = {
    "classification": ("trainable_model", "classification"),
    "regression":     ("trainable_model", "regression"),
    "detection":      ("trainable_model", "detection"),
    "transcription":  ("pretrained_servable", "transcription"),
}

# Module-level model, inserted verbatim (only for trainable types).
CNN_DEF = '''
class _SmallAudioCNN(nn.Module):
    """Shape-agnostic conv net over a (1, F, T) feature map."""
    def __init__(self, out_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 4 * 4, 64),
                                  nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, out_dim))

    def forward(self, x):
        return self.head(self.features(x))

'''

# Indented methods, inserted verbatim (normal braces — no .format()).
METHODS = {
    "classification": '''    def build_model(self, input_shape):
        return _SmallAudioCNN(out_dim=self.config.num_classes)

    def compute_loss(self, model, batch, device):
        x, y = batch; x, y = x.to(device), y.to(device)
        return nn.CrossEntropyLoss()(model(x), y)

    def compute_metrics(self, y_true, y_pred):
        from sklearn.metrics import accuracy_score
        yp = y_pred.argmax(axis=1) if y_pred.ndim == 2 else y_pred
        return {"accuracy": float(accuracy_score(y_true, yp))}

    def postprocess(self, raw_output):
        probs = torch.softmax(raw_output, dim=-1)
        idx = int(probs.argmax(dim=-1).flatten()[0].item())
        labels = self.config.class_labels or [str(i) for i in range(probs.shape[-1])]
        return {"label": labels[idx] if idx < len(labels) else str(idx),
                "confidence": float(probs.flatten()[idx].item())}
''',
    "regression": '''    def build_model(self, input_shape):
        return _SmallAudioCNN(out_dim=1)

    def compute_loss(self, model, batch, device):
        x, y = batch; x = x.to(device); y = y.to(device).float().view(-1, 1)
        return nn.MSELoss()(model(x), y)

    def compute_metrics(self, y_true, y_pred):
        import numpy as np
        yp = y_pred.reshape(-1); yt = y_true.reshape(-1)
        return {"mae": float(np.mean(np.abs(yp - yt))),
                "rmse": float(np.sqrt(np.mean((yp - yt) ** 2)))}

    def postprocess(self, raw_output):
        return {"value": float(raw_output.flatten()[0].item())}
''',
    "detection": '''    def build_model(self, input_shape):
        return _SmallAudioCNN(out_dim=self.config.num_classes)  # one logit per class

    def compute_loss(self, model, batch, device):
        x, y = batch; x = x.to(device); y = y.to(device).float()
        return nn.BCEWithLogitsLoss()(model(x), y)

    def compute_metrics(self, y_true, y_pred):
        import numpy as np
        preds = (1 / (1 + np.exp(-y_pred)) > 0.5).astype(int)
        return {"exact_match": float((preds == y_true).all(axis=1).mean())}

    def postprocess(self, raw_output):
        probs = torch.sigmoid(raw_output).flatten()
        labels = self.config.class_labels or [str(i) for i in range(probs.shape[0])]
        return {"labels": [labels[i] for i, p in enumerate(probs) if p.item() > 0.5],
                "scores": {labels[i]: float(p.item()) for i, p in enumerate(probs)}}
''',
    "transcription": '''    # Serve-only by default. Plug in your STT runtime (faster-whisper, wav2vec2).
    def build_model(self, input_shape):
        raise NotImplementedError("Transcription serving uses an STT runtime, not build_model.")

    def compute_loss(self, model, batch, device):
        raise NotImplementedError

    def compute_metrics(self, y_true, y_pred):
        return {}

    def postprocess(self, raw_output):
        return {"text": str(raw_output)}
''',
}

HEADER = '''"""
{title} task — scaffolded by scripts/new_task.py (type: {ttype}).

See audio_ops_dagster/tasks/example_classifier/ for a complete reference.
"""

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from audio_ops_dagster.core.base_task import AudioTask
from audio_ops_dagster.registry import register_task

__CNN__
@register_task("{name}")
class {cls}(AudioTask):
    TASK_TYPE = "{tasktype}"

__METHODS__'''

CONFIG_YAML = '''# {name} — scaffolded task ({ttype}).
archetype: {archetype}

task_name: {name}
task_type: {tasktype}

input_dir: "audio_input/{name}"
output_dir: "storage/{name}"
feature_type: {feature}
input_shape: [1, 128, 87]

{class_block}
split_strategy: group
batch_size: 16
epochs: 5
learning_rate: 0.001
use_remote_storage: false
'''


def camel(name: str) -> str:
    return "".join(p.capitalize() for p in re.split(r"[_\-]+", name)) + "Task"


def main() -> int:
    ap = argparse.ArgumentParser(description="Scaffold a new audio task.")
    ap.add_argument("task_name")
    ap.add_argument("--type", dest="ttype", default="classification", choices=TYPES)
    ap.add_argument("--archetype", default=None, choices=ARCHETYPES)
    ap.add_argument("--feature", default="spectrogram")
    args = ap.parse_args()

    name = args.task_name
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        print(f"Task name must be snake_case (got '{name}').", file=sys.stderr)
        return 1

    default_arch, tasktype = TYPE_DEFAULTS[args.ttype]
    archetype = args.archetype or default_arch

    tasks_root = Path(__file__).resolve().parent.parent / "audio_ops_dagster" / "tasks"
    task_dir = tasks_root / name
    if task_dir.exists():
        print(f"Task '{name}' already exists at {task_dir}", file=sys.stderr)
        return 1

    if args.ttype in ("classification", "detection"):
        class_block = "num_classes: 2\nclass_labels:\n  - class_a\n  - class_b"
    elif args.ttype == "regression":
        class_block = "# regression: single continuous output, no class_labels"
    else:
        class_block = "# transcription: no class_labels"

    header = HEADER.format(title=name.replace("_", " ").title(), ttype=args.ttype,
                           name=name, cls=camel(name), tasktype=tasktype)
    cnn = CNN_DEF if args.ttype != "transcription" else ""
    source = header.replace("__CNN__", cnn).replace("__METHODS__", METHODS[args.ttype])

    task_dir.mkdir(parents=True)
    (task_dir / "__init__.py").write_text("")
    (task_dir / "config.yaml").write_text(CONFIG_YAML.format(
        name=name, ttype=args.ttype, tasktype=tasktype, archetype=archetype,
        feature=args.feature, class_block=class_block))
    (task_dir / f"task_{name}.py").write_text(source)

    print(f"Created {args.ttype} task '{name}' ({archetype}, feature={args.feature}) at {task_dir}")
    if args.ttype != "transcription":
        print(f"Next: drop audio under audio_input/{name}/<label>/, then `dagster dev`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
