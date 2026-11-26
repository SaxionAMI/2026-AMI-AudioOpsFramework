# Audio Ops Framework

[![CI](https://github.com/SaxionAMI/2026-AMI-AudioOpsFramework/actions/workflows/ci.yml/badge.svg)](https://github.com/SaxionAMI/2026-AMI-AudioOpsFramework/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-zensical-blue)](https://saxionami.github.io/2026-AMI-AudioOpsFramework/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)

Audio Ops is a Python framework for audio machine learning. You describe a
task (for example crowd noise classification, wake word detection or speech
transcription) in a `config.yaml`, and the framework provides feature
extraction, training, evaluation, batch scoring and a streaming inference API
for it.

The offline pipeline runs on [Dagster](https://dagster.io). The inference API
is a FastAPI service that accepts file uploads, RTSP streams and WebSocket
audio. Both use the same task code and the same feature extraction. Everything
runs locally without extra services. Object storage, metrics, experiment
tracking, an LLM and the fusion timeline can be switched on with environment
variables.

```text
tasks/<name>/config.yaml  (+ optional task_<name>.py)    written by you
        |  capabilities: train, evaluate, batch_infer, serve
        v
 Dagster: raw audio > features > splits > trained model > evaluation
                                        > batch inference (JSON + CSV)
 Inference API: file upload, RTSP pull, WebSocket ingest > SSE results
```

Speech toolkits such as SpeechBrain and NeMo provide models, and tools such as
Dagster, MLflow and BentoML provide general pipelines and serving. Audio Ops
connects the two for audio: one task definition gives you both the training
pipeline and a service for live streams, with identical features in both. See
[Comparison with other tools](docs/concepts/why-audio-ops.md).

## Quickstart

Needs Linux, macOS or Windows with WSL2, Python 3.10 to 3.14 and git.

```bash
git clone https://github.com/SaxionAMI/2026-AMI-AudioOpsFramework
cd 2026-AMI-AudioOpsFramework
./scripts/setup.sh        # .venv, PyTorch for your hardware, the framework, .env, demo audio
source .venv/bin/activate

audio-ops run example_classifier          # features, splits, training, evaluation, scoring
audio-ops predict example_classifier audio_input/example_classifier/class_b/class_b_00.wav
audio-ops serve                           # inference API on http://localhost:8000/docs
audio-ops dev                             # the same pipeline in the Dagster UI, http://localhost:3000
```

The script is safe to run again and explains what to do when something is
missing; `./scripts/setup.sh --help` lists its options. With Docker:
`make setup && make up-all`. Details in
[Installation](docs/getting-started/installation.md).

To build your own project on the framework, copy
[`project-template/`](project-template/) and run its `make setup`. It installs
the framework as a package; see [Project repos](docs/guide/project-repos.md).

## Write a task

A standard task needs only a config file:

```bash
audio-ops new-task wake_word --config-only --classes wake,not_wake
```

```yaml
# tasks/wake_word/config.yaml
archetype: trainable_model        # train + evaluate + batch_infer + serve
task_type: classification         # or regression / detection
feature_type: spectrogram         # mel | mfcc | raw | ast | your own @register_feature
class_labels: [wake, not_wake]
split_strategy: group             # no recording leaks across splits
```

Put audio in `audio_input/wake_word/<label>/*.wav` and run
`audio-ops run wake_word`. Tasks in `./tasks` are found automatically. To use
your own model or metrics, subclass a base and override that method:

```python
# tasks/wake_word/task_wake_word.py
from audio_ops import ClassificationTask, register_task

@register_task("wake_word")
class WakeWord(ClassificationTask):
    def build_model(self, input_shape):
        return MyNet(self.num_classes)        # any torch.nn.Module
```

Tasks that don't fit these bases, such as speech-to-text, implement the full
[`AudioTask`](docs/guide/tasks.md) interface. A task that only serves adds
nothing to Dagster, and a task that is only trained never starts the API.

## Used in

- [GRASS](https://www.saxion.nl/onderzoek/overige-projecten/ambient-intelligence/grass---gastvrij-en-respectvol-gedrag-van-supporters-in-stadions)
  (Gastvrij en Respectvol gedrAg van Supporters in Stadions): measuring crowd
  sentiment in Dutch football stadiums, with Heracles Almelo, FC Twente,
  Go Ahead Eagles, the KNVB and others. The project code
  ([SaxionAMI/2025-RAAKMKB-GRASS](https://github.com/SaxionAMI/2025-RAAKMKB-GRASS))
  uses Audio Ops as a package.
- [AI-ZI](https://www.saxion.nl/binaries/content/assets/onderzoek/lectoraten/technology-health--care/magazine-deftechnology-health--care-mei-2026-1.pdf)
  (AI voor bemensbare Zorg en Industrie): explainable AI to support
  professionals in care and manufacturing.

Both are applied research projects at Saxion University of Applied Sciences
([research group Ambient Intelligence](https://www.saxion.edu/ami)). More on the
[Used in](docs/project/used-in.md) page. Audio Ops is not a medical device.

## Documentation

Full docs: **https://saxionami.github.io/2026-AMI-AudioOpsFramework/** (source in
[`docs/`](docs/), built with [Zensical](https://zensical.org): `make docs-serve`).

- [Quickstart](docs/getting-started/quickstart.md) · [Your first task](docs/getting-started/first-task.md) · [Writing tasks](docs/guide/tasks.md)
- [Comparison with other tools](docs/concepts/why-audio-ops.md) · [Capabilities](docs/concepts/capabilities.md) · [Architecture](docs/concepts/architecture.md)
- [Streaming inference](docs/guide/streaming.md) · [Deployment](docs/guide/deployment.md)
- Reference: [task config](docs/reference/task-config.md) · [environment](docs/reference/environment.md) · [CLI](docs/reference/cli.md) · [HTTP API](docs/reference/http-api.md)

## Contact

Questions, ideas or a project you would like to build on Audio Ops:
[e.m.vanwolfswinkel@saxion.nl](mailto:e.m.vanwolfswinkel@saxion.nl). Bugs and feature requests go in the
[issue tracker](https://github.com/SaxionAMI/2026-AMI-AudioOpsFramework/issues).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) (we use the
DCO sign-off, no CLA) and the [Code of Conduct](CODE_OF_CONDUCT.md). Report
security issues privately as described in [SECURITY.md](SECURITY.md).

## License

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) and
[THIRD_PARTY.md](THIRD_PARTY.md). Models your tasks download at runtime are
covered by their own licenses.
