"""
scaffold.py — generate a new task (``audio-ops new-task``).

Types:      classification (default) | regression | detection | transcription
Archetypes: trainable_model | pretrained_servable | pretrained_batch | eval_only

Trainable types build on the ready-made bases (``ClassificationTask`` …), so
the generated Python file is a few lines that you extend only where you need
to. ``--config-only`` skips the Python file entirely.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

ARCHETYPES = ("trainable_model", "pretrained_servable", "pretrained_batch", "eval_only")
TYPES = ("classification", "regression", "detection", "transcription")
_BASE = {"classification": "ClassificationTask", "regression": "RegressionTask",
         "detection": "DetectionTask"}

TRAINABLE_TEMPLATE = '''"""
{title} — a {ttype} task (scaffolded by `audio-ops new-task`).

{base} already implements the whole task contract: SmallAudioCNN as the model,
the loss from config.yaml (criterion / class_weights / label_smoothing), the
standard metrics and the output format. Override only what you want to change.
"""

from audio_ops import {base}, SmallAudioCNN, register_task


@register_task("{name}")
class {cls}({base}):
    def build_model(self, input_shape):
        # Any torch.nn.Module works. Inputs are (batch, 1, freq, time) for 2-D
        # features, (batch, samples) for feature_type: raw.
        return SmallAudioCNN({out_dim})

    # Other hooks you can override (see the docs, "Writing tasks"):
    #   compute_loss(self, model, batch, device)  -> loss | (loss, outputs)
    #   compute_metrics(self, y_true, y_pred)     -> {{"metric": value}}
    #   postprocess(self, raw_output)             -> JSON-friendly dict
    #   get_optimizer(self, model)                -> torch optimizer
'''

TRANSCRIPTION_TEMPLATE = '''"""
{title} — a transcription task (scaffolded by `audio-ops new-task`).

Serve-only by default: wrap your speech-to-text runtime (faster-whisper,
wav2vec2, …) in predict(). The inference service calls it on buffered stream
audio; batch inference calls it on transcription_chunk_seconds windows.
"""

from typing import Any, Dict

import numpy as np

from audio_ops import AudioTask, register_task


@register_task("{name}")
class {cls}(AudioTask):
    TASK_TYPE = "transcription"

    def predict(self, audio: np.ndarray, t_offset: float = 0.0, **kwargs) -> Dict[str, Any]:
        """Transcribe 16 kHz mono float32 audio.

        Return at least {{"transcription": str}}. Optional: "words" — a list of
        {{"word", "t_start", "t_end", "confidence"}} on the media clock (add
        t_offset to your runtime's timestamps) — "language", "confidence".
        """
        raise NotImplementedError("Plug your speech-to-text runtime into predict().")

    # Not used for serve-only tasks, but part of the AudioTask contract:
    def build_model(self, input_shape):
        raise NotImplementedError

    def compute_loss(self, model, batch, device):
        raise NotImplementedError

    def compute_metrics(self, y_true, y_pred):
        return {{}}

    def postprocess(self, raw_output):
        return {{"transcription": str(raw_output)}}
'''

CONFIG_YAML = '''# {name} — {ttype} task. Check it with: audio-ops validate {name}
# Every key and default: docs → Reference → Task configuration.
archetype: {archetype}
task_type: {ttype}

# Data: audio under input_dir/<label>/*.wav — the folder name is the label.
input_dir: audio_input/{name}
feature_type: {feature}
segment_duration: 1.0      # seconds per training segment (also the serving window)
{labels}
split_strategy: group      # all segments of a recording stay in one split

# Training
epochs: 20
batch_size: 16
learning_rate: 0.001
'''


def camel(name: str) -> str:
    return "".join(p.capitalize() for p in re.split(r"[_\-]+", name)) + "Task"


def default_tasks_dir() -> Path:
    """Where `new-task` writes when no --dir is given.

    1. the first ``AUDIO_OPS_TASKS_DIR`` entry, 2. ``./audio_ops_dagster/tasks``
    when run from a framework checkout, 3. ``./tasks``.
    """
    env = os.environ.get("AUDIO_OPS_TASKS_DIR", "")
    for part in env.replace(",", os.pathsep).split(os.pathsep):
        if part.strip():
            return Path(part.strip()).expanduser()
    checkout = Path.cwd() / "audio_ops_dagster" / "tasks"
    if checkout.is_dir():
        return checkout
    return Path.cwd() / "tasks"


def create_task(name: str, ttype: str = "classification", archetype: Optional[str] = None,
                feature: str = "spectrogram", tasks_dir: Optional[Path] = None,
                classes: Optional[list] = None, config_only: bool = False) -> Path:
    """Create ``<tasks_dir>/<name>/`` and return its path."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError(f"Task name must be snake_case (got '{name}').")
    if ttype not in TYPES:
        raise ValueError(f"Unknown type '{ttype}'. Choose from {TYPES}.")
    if archetype is not None and archetype not in ARCHETYPES:
        raise ValueError(f"Unknown archetype '{archetype}'. Choose from {ARCHETYPES}.")
    if config_only and ttype == "transcription":
        raise ValueError("Transcription tasks need Python (predict); drop --config-only.")

    archetype = archetype or ("pretrained_servable" if ttype == "transcription"
                              else "trainable_model")
    root = Path(tasks_dir) if tasks_dir else default_tasks_dir()
    task_dir = root / name
    if task_dir.exists():
        raise FileExistsError(f"Task '{name}' already exists at {task_dir}")

    if ttype in ("classification", "detection"):
        classes = classes or ["class_a", "class_b"]
        labels = ("class_labels:            # output order; must match the folder names\n"
                  + "\n".join(f"  - {json.dumps(str(c))}" for c in classes))
    elif ttype == "regression":
        labels = "# Regression targets come from numeric folder names, e.g. 0.75/"
    else:
        labels = "stream_buffer_seconds: 5.0   # audio buffered per transcription call"

    task_dir.mkdir(parents=True)
    (task_dir / "config.yaml").write_text(CONFIG_YAML.format(
        name=name, ttype=ttype, archetype=archetype, feature=feature, labels=labels))
    if config_only:
        return task_dir

    (task_dir / "__init__.py").write_text("")
    fmt = dict(title=name.replace("_", " ").title(), name=name, cls=camel(name), ttype=ttype)
    if ttype == "transcription":
        source = TRANSCRIPTION_TEMPLATE.format(**fmt)
    else:
        out_dim = "1" if ttype == "regression" else "self.num_classes"
        source = TRAINABLE_TEMPLATE.format(base=_BASE[ttype], out_dim=out_dim, **fmt)
    (task_dir / f"task_{name}.py").write_text(source)
    return task_dir
