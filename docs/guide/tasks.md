# Writing tasks

A task is a folder; its **name is its identity** (asset, job and API names
derive from it). There are three levels. Use the lowest one that works.

| Level | You write | Good for |
|---|---|---|
| 1. Config only | `config.yaml` | standard classification, regression, multi-label detection |
| 2. Subclass a base | `config.yaml` + a few lines of Python | your own model, metrics or output |
| 3. Full contract | subclass `AudioTask` | anything else (speech-to-text, custom runtimes) |

Everything you need imports from one place:

```python
from audio_ops import (ClassificationTask, RegressionTask, DetectionTask, AudioTask,
                       SmallAudioCNN, register_task, register_feature, register_wrapper)
```

## Level 1: config only

```text
tasks/<name>/
└── config.yaml          # task_type: classification | regression | detection
```

The folder gets the base class for its `task_type`. Targets come from the
label folders under `input_dir`:

| `task_type` | Folder names | Output |
|---|---|---|
| `classification` | class names (`dog/`, `cat/`) | `label`, `confidence`, `scores` |
| `detection` | class names; one positive class per clip | `labels` above `detection_threshold` (0.5), `scores` |
| `regression` | numbers (`0.25/`, `0.75/`) | `value` |

`class_labels` fixes the class order (= output index); without it the sorted
folder names are used and stored in the checkpoint. `num_classes` is derived.
The loss follows `criterion` (default by type: cross-entropy, BCE, MSE) with
`class_weights` and `label_smoothing`; the optimizer follows `optimizer`,
`learning_rate`, `weight_decay`. Full list: [task configuration](../reference/task-config.md).

## Level 2: subclass a base

Add `task_<name>.py` (or `task.py`) and override only what differs:

```python
import torch.nn as nn
from audio_ops import ClassificationTask, register_task


@register_task("crowd_mood")
class CrowdMood(ClassificationTask):
    def build_model(self, input_shape):
        return MyNet(n_out=self.num_classes)

    def compute_metrics(self, y_true, y_pred):
        from sklearn.metrics import recall_score
        m = super().compute_metrics(y_true, y_pred)          # accuracy, f1_macro
        m["recall_angry"] = float(recall_score(y_true == 2, y_pred.argmax(1) == 2))
        return m
```

Useful members on every base: `self.config` (the [`TaskConfig`](../reference/task-config.md)
including your own custom keys, e.g. `self.config.my_threshold`),
`self.labels`, `self.num_classes`, `self.device`, `self.get_loss_function()`,
`self.get_optimizer(model)`.

## Level 3: the full interface

Subclass `AudioTask` and implement four methods:

| Method | Called by | Receives | Returns |
|---|---|---|---|
| `build_model(input_shape)` | trainer, evaluation, batch inference, API | `config.input_shape` (may be `None`) | an `nn.Module` |
| `compute_loss(model, batch, device)` | trainer (train **and** validation), evaluation | `(inputs, targets)` | a loss tensor, or `(loss, outputs)` |
| `compute_metrics(y_true, y_pred)` | evaluation | numpy targets and raw model outputs | `{"name": float}` |
| `postprocess(raw_output)` | batch inference, API | model output for a batch of one | JSON-friendly dict |

Returning `(loss, outputs)` from `compute_loss` lets the trainer report
accuracy without a second forward pass.

### What `postprocess` should return

Anything JSON-serialisable. Conventional keys get extra treatment:
`label`/`confidence` (logs, metrics, fusion sentiment), `scores` (per-class;
file inference averages them across segments), `value` (regression), `labels`
(multi-label). Scalar fields become CSV columns in batch inference.

### Optional hooks

```python
def format_influx_record(self, segment_id, prediction, audio_offset_sec=None):
    """Time-series schema for batch inference → {"fields": {...}, "tags": {...}}"""

def grafana_panel_hints(self):
    """Metadata for your own dashboard provisioning."""

def postprocess_with_llm(self, raw_result, context=None, llm_client=None):
    """Refine a prediction with an LLM (ENABLE_LLM=true)."""

def warmup(self):
    """Load heavy weights before a batch run."""
```

### Speech-to-text and other serve-only tasks

Tasks that wrap their own runtime implement `predict` instead of training:

```python
from audio_ops import AudioTask, register_task


@register_task("whisper_stt")
class WhisperTask(AudioTask):
    TASK_TYPE = "transcription"

    def predict(self, audio, t_offset=0.0, **kwargs):
        # audio: 16 kHz mono float32. Return at least {"transcription": str};
        # optional "words": [{"word", "t_start", "t_end", "confidence"}] with
        # t_offset added, "language", "confidence".
        ...
    # build_model / compute_loss / compute_metrics / postprocess: trivial stubs
    # (`audio-ops new-task whisper_stt --type transcription` writes them for you)
```

```yaml
archetype: pretrained_servable
stream_buffer_seconds: 5.0     # audio buffered per transcription call
silence_rms: 0.006             # skip near-silent buffers
model_name: large-v3           # your own key → self.config.model_name
```

Event taggers (YAMNet-style) implement `predict(audio, sample_rate=16000, source="stream")`
and set `task_type: nonverbal_pretrained` (or `serve_wrapper: pretrained_event`).
Runtimes such as `faster-whisper` or `tensorflow` are your project's
dependencies. Add them to the inference image with `EXTRA_PIP`.

## Splitting a task into modules

Task folders are imported as packages, so relative imports work:

```text
tasks/my_task/
├── __init__.py
├── config.yaml
├── model.py
└── task_my_task.py        # from .model import Net
```

## The reference task

`audio_ops_dagster/tasks/example_classifier/` subclasses `ClassificationTask`
with its own CNN, an extra metric and a custom time-series schema.
