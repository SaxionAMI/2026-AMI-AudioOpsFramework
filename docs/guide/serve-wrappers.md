# Custom serve wrappers

When a task can't be served by the built-in `classifier`, `transcription` or
`pretrained_event` wrappers, register a factory for a new wrapper type. Import
it from your task module so it registers during discovery:

```python
import numpy as np
from audio_ops import ServeWrapper, register_wrapper


class OnnxWrapper(ServeWrapper):
    kind = "classifier"            # "classifier" | "transcription" | "event"

    def __init__(self, task, task_name, cfg, session):
        super().__init__(task, task_name, cfg)
        self.session = session     # shared across streams
        self.reset()

    def reset(self):               # per-stream state lives here
        self._buf = np.zeros(0, dtype=np.float32)

    def predict(self, audio, media_t0=0.0):
        self._buf = np.concatenate([self._buf, audio])[-16000:]
        out = self.session.run(None, {"audio": self._buf[None]})[0]
        return {**self.task.postprocess(out),
                "media_ts_start": media_t0, "media_ts_end": media_t0 + len(audio) / 16000}

    def describe(self):
        return {"task": self.task_name, "type": "onnx"}


@register_wrapper("onnx")
def make_onnx(task, task_name, cfg):
    import onnxruntime as ort
    return OnnxWrapper(task, task_name, cfg, ort.InferenceSession(cfg.onnx_path))
```

```yaml
# config.yaml
archetype: pretrained_servable
serve_wrapper: onnx
onnx_path: models/my_model.onnx
```

## The interface

| Member | Purpose |
|---|---|
| `kind` | routes results: `classifier` (metrics + fusion sentiment), `event` (non-verbal track), `transcription` (separate STT lane) |
| `predict(audio, media_t0)` | one chunk (16 kHz mono float32); may buffer; return a dict stamped with `media_ts_start`/`media_ts_end` |
| `predict_clip(audio)` | one whole clip for `/infer/file`; default = a fresh instance's `predict` |
| `reset()` | clear per-stream state |
| `for_stream()` | default: shallow copy + `reset()`; override if state is more complex |
| `describe()` | what `GET /models` shows |

A factory returns `None` when the task can't be served yet (e.g. no
checkpoint); the store lists it under `not_loaded`.

!!! note
    Importing `register_wrapper` doesn't import FastAPI or start the app, so
    task modules can use it safely (it lives in `audio_ops_inference.wrappers`;
    the old `audio_ops_inference.api` import still works).
