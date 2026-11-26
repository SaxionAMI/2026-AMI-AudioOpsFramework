# Your first task

A task is a folder. Let's build a two-class "is someone speaking?" classifier,
first with **no Python at all**, then customised.

## 1. Config only

```bash
audio-ops new-task speech_presence --config-only --classes speech,silence
```

```text
tasks/speech_presence/
└── config.yaml
```

```yaml title="tasks/speech_presence/config.yaml (abridged)"
archetype: trainable_model     # train + evaluate + batch_infer + serve
task_type: classification
input_dir: audio_input/speech_presence
feature_type: spectrogram
segment_duration: 1.0
class_labels: ["speech", "silence"]
split_strategy: group
epochs: 20
```

That is a complete task. A folder without Python gets the ready-made
`ClassificationTask`: a small shape-agnostic CNN, cross-entropy loss,
accuracy/F1 metrics and `label`/`confidence`/`scores` output. (`regression` and
`detection` work the same way.)

`./tasks` is discovered automatically when you run `audio-ops` from the folder
that contains it. Elsewhere, point `AUDIO_OPS_TASKS_DIR` at it.

## 2. Add data, check, train

```text
audio_input/speech_presence/
├── speech/    *.wav
└── silence/   *.wav
```

```bash
audio-ops validate speech_presence     # config errors, typos, YAML pitfalls
audio-ops run speech_presence          # features → splits → train → evaluate → score
```

`run` prints the training curve and where everything went
(`storage/speech_presence/`). To use the Dagster UI instead, run `audio-ops dev`
and launch `speech_presence_full_training_job`.

## 3. Use it

```bash
audio-ops predict speech_presence meeting.wav --segments
audio-ops serve                        # HTTP / RTSP / WebSocket, see Streaming
```

## 4. Customise with Python

When the defaults aren't enough, add a Python file and override only what you
need. Scaffold it with `audio-ops new-task speech_presence` (without
`--config-only`), or write it yourself:

```python title="tasks/speech_presence/task_speech_presence.py"
from audio_ops import ClassificationTask, SmallAudioCNN, register_task


@register_task("speech_presence")
class SpeechPresence(ClassificationTask):
    def build_model(self, input_shape):
        # any torch.nn.Module; here the default CNN at double width
        return SmallAudioCNN(self.num_classes, channels=64)
```

The loss, metrics and output format still come from the base class.
See [Writing tasks](../guide/tasks.md) for every hook and for tasks that don't
fit a base (speech-to-text, custom runtimes).

!!! tip "Feature shape"
    The feature step logs the actual shape. With `segment_duration: 1.0` at
    16 kHz and the default `spectrogram` it is `(128, 32)`, so models see
    `(batch, 1, 128, 32)`. Set `input_shape` only if your model needs it.
