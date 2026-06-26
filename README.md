# Audio Ops Framework

A framework for building **audio machine-learning pipelines** — feature
extraction, training, evaluation, and real-time inference — that you can reuse
across very different projects (stadium analytics, medical audio, call triage,
wake-word, …) by writing small **tasks** instead of rebuilding plumbing.

Dagster is the orchestration core. A task declares *what it is* and *what it can
do*; the framework assembles exactly the pipeline that task needs and nothing
more. It runs out of the box with no external services — object storage,
metrics, dashboards, and LLM features are optional and added only when a project
wants them.

---

## Table of contents
1. [The mental model](#1-the-mental-model)
2. [Architecture](#2-architecture)
3. [The capability system (the core logic)](#3-the-capability-system-the-core-logic)
4. [The training data flow](#4-the-training-data-flow)
5. [The inference / serving path](#5-the-inference--serving-path)
6. [Storage model](#6-storage-model)
7. [Anatomy of a task](#7-anatomy-of-a-task)
8. [Quickstart](#8-quickstart)
9. [Configuration reference](#9-configuration-reference)
10. [Optional integrations](#10-optional-integrations)
11. [Project layout](#11-project-layout)
12. [Further improvements](#12-further-improvements)

---

## 1. The mental model

Three ideas carry the whole framework:

**Tasks, not pipelines.** You never edit the pipeline. You write a *task* — a
model plus four methods (build / loss / metrics / postprocess) and a
`config.yaml`. Auto-discovery finds it; the framework builds its pipeline.

**Capabilities, not flags.** A task declares what it can do —
`train`, `evaluate`, `batch_infer`, `serve` — and the builder *adds* exactly
those stages. A serve-only model contributes zero Dagster assets; a
training-only project never starts the API. One framework, many shapes.

**Two meanings of "inference," kept separate.** Offline scoring of a dataset is
a Dagster asset (`batch_infer`). The long-running RTSP/WebSocket/SSE service is
a daemon (`serve`). They never get tangled.

---

## 2. Architecture

```
                       ┌──────────────────────────────────────────┐
                       │            tasks/<name>/                  │
                       │  config.yaml  +  task_<name>.py           │
                       │  (declares capabilities + the model)      │
                       └───────────────┬──────────────────────────┘
                      auto-discovery (registry.py)
                                       │
            ┌──────────────────────────┴───────────────────────────┐
            ▼                                                        ▼
   ┌───────────────────┐                                  ┌────────────────────┐
   │  DAGSTER (core)   │                                  │  INFERENCE API     │
   │  definitions.py   │  reads capabilities              │  (FastAPI)         │
   │  builds per task: │                                  │  serves tasks with │
   │   raw → features  │                                  │  the `serve` cap   │
   │   → splits        │   train/evaluate/batch_infer     │  • file upload     │
   │   → trained_model │                                  │  • RTSP pull       │
   │   → evaluation    │                                  │  • WS mic ingest   │
   │   → inference     │                                  │  • SSE out + fusion│
   └─────────┬─────────┘                                  └─────────┬──────────┘
             │ checkpoints / features / outputs                     │ loads checkpoints
             ▼                                                      ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  STORAGE: local filesystem by default (./storage),                        │
   │           or S3/MinIO when STORAGE_BACKEND=minio                          │
   └──────────────────────────────────────────────────────────────────────────┘

   Optional, off by default: InfluxDB metrics · MLflow server · LLM · MediaMTX
```

The two runtimes share one thing only: the **task contract** (registry +
`config.yaml` + the on-disk artifact layout). Dagster trains and writes
checkpoints; the API reads them. Neither imports the other's internals.

---

## 3. The capability system (the core logic)

Every task resolves to a set of capabilities (`capabilities.py`):

| Capability    | Meaning                                  | Dagster assets built                                   |
|---------------|------------------------------------------|--------------------------------------------------------|
| `train`       | produces a checkpoint                    | raw_audio → features → splits → **trained_model**      |
| `evaluate`    | scores a held-out split                  | …above… → **model_evaluation**                         |
| `batch_infer` | offline scoring of a dataset             | raw_audio → features → **inference_results**           |
| `serve`       | online serving via the API               | *(none — handled by the inference daemon)*             |

**Archetypes** are named bundles so configs read as intent:

```yaml
archetype: trainable_model        # train + evaluate + batch_infer + serve
archetype: pretrained_servable    # serve only
archetype: pretrained_batch       # batch_infer only
archetype: eval_only              # evaluate only
```

**Resolution order** (in `resolve_capabilities`): an explicit `capabilities:`
list wins; otherwise an `archetype:`; otherwise a legacy
`dagster.training_enabled` flag (back-compat) — defaulting to a full
`trainable_model`.

**How assembly works.** `definitions.py` loops over `tasks/`, resolves each
task's capabilities, and:
- registers the task's resource (always),
- calls `build_audio_assets(task, resource_key, caps)` — which returns only the
  assets the capabilities require, resolving prerequisites (e.g. `evaluate`
  pulls in `splits` and `trained_model`),
- calls `build_audio_jobs(task, caps)` — which only emits a job when its assets
  exist (no "full training job" for a serve-only task).

A serve-only task passes `produces_dagster_assets(caps) == False` and is skipped
entirely in Dagster — it exists only for the API to load. That single branch is
why a medical transcription project deploys *only* the API, while a stadium
classifier project deploys the full training stack.

---

## 4. The training data flow

For a task with `train`/`evaluate`, the assets run in this order (`assets.py`):

1. **raw_audio_files** — scans `input_dir/` for `.wav`/`.mp3`. Convention: audio
   lives under `input_dir/<label>/…`, so the folder name is the class label.
2. **segmented_features** — `AudioProcessor` slices each file into fixed windows
   (default 1 s) and writes one `.npy` log-mel spectrogram per segment to
   `output_dir/features/<label>/<recording>_seg_NNN.npy`.
3. **dataset_splits** — `AudioDataset.create_splits_csv` scans those features and
   writes `train/val/test.csv`. The **group** strategy groups segments by source
   recording (everything before `_seg`/`_chunk`) and does a *stratified* split on
   the groups, so segments from one clip never leak across splits and class
   balance is preserved. (Use `split_strategy: group` whenever segments are
   correlated; `random` otherwise.)
4. **trained_model** — builds the task's model, trains with `ModelTrainer`
   (MLflow-logged), writes `output_dir/checkpoints/best_model.pt`.
5. **model_evaluation** — loads the checkpoint, scores `test.csv` via the task's
   `compute_metrics`.
6. **inference_results** (`batch_infer`) — runs the model over all features and
   writes results JSON (and time-series points if metrics are enabled).
   Classifier and transcription paths differ: transcription concatenates
   segments into larger chunks for efficiency.

Every step optionally syncs its artifact to object storage **only if** the task
sets `use_remote_storage: true` *and* a storage backend is enabled — otherwise
everything stays on local disk.

---

### Running inference on a new batch (no scripts, stays in Dagster)

"Score this folder of audio with a trained task" is just the task's
`<task>_inference_job` — the `batch_infer` sub-graph (`raw_audio → features →
inference_results`), no training, no daemon. To point it at arbitrary data,
launch the job and set the run config **once** on the root asset:

```yaml
ops:
  example_classifier_raw_audio_files:
    config:
      input_dir: /data/new_batch          # audio to score
      output_dir: /data/new_batch_out      # features + results land here
```

The override threads through the whole inference path. Results are written to
`<output_dir>/outputs/` as both JSON and CSV. Leave the config unset for normal
pipeline runs (it uses the task's configured dirs). The trained checkpoint is
read from the task's `output_dir/checkpoints` (override with the
`INFERENCE_CHECKPOINT` env var). Streaming, by contrast, is **not** a job — it's
the long-running service in §5.

## 5. The inference / serving path

The FastAPI app (`audio_ops_inference/api.py`) is a standalone daemon. On
startup its `ModelStore`:
- discovers tasks from `tasks/` and loads any with a servable wrapper
  (classification, transcription, or a pretrained event model),
- loads trained checkpoints **local-first** from `MODELS_DIR` (set
  `STORAGE_BACKEND=minio` to pull from object storage instead).

How a task is served is decided by a **wrapper registry** (`@register_wrapper`)
keyed on task type — `classifier`, `transcription`, `pretrained_event` ship
built in, and you add new servable kinds by registering a factory rather than
editing the API. Classifier serving builds the network from the task's own
`build_model()` and loads the checkpoint (no architecture-name reconstruction).

It then exposes:
- **file upload** — score a clip,
- **RTSP pull** — consume a live stream (MediaMTX provides one for demos),
- **WebSocket ingest** — push browser/mic PCM frames on a shared media clock,
- **SSE output** — per-segment results streamed to clients,
- **fusion** *(optional)* — a project-specific add-on
  (`audio_ops_dagster/extras/fusion.py`) that merges transcript + event +
  sentiment tracks onto one timeline. It is **off by default**; enable with
  `ENABLE_FUSION=true`. Not part of the generic core.

The LLM client and fusion degrade gracefully: if disabled or unavailable, the
API logs a note and serves everything else.

---

## 6. Storage model

One taxonomy (`core/storage.py` → `ArtifactType`): `audio`, `features`,
`dataset`, `models`, `outputs`. Keys are `{task}/{artifact_type}/{file}`.

- **Default — local filesystem.** `MinIOStorageResource` is disabled; all
  artifacts live under `./storage/<task>/…`. No service required.
- **Opt-in — S3/MinIO** via the boto3-based `S3StorageBackend` (works with
  MinIO, AWS S3, R2, B2). Enable with `STORAGE_BACKEND=minio` + `MINIO_*`. Uploads
  still only happen for tasks with `use_remote_storage: true`.

---

## 7. Anatomy of a task

A task is a folder under `audio_ops_dagster/tasks/`:

```
tasks/example_classifier/
  __init__.py
  config.yaml                 # capabilities + data + hyperparameters
  task_example_classifier.py  # the model + 4 methods
```

The class subclasses `AudioTask` and implements:

```python
@register_task("example_classifier")
class ExampleClassifierTask(AudioTask):
    TASK_TYPE = "classification"          # or "transcription" / "detection"
    def build_model(self, input_shape): ...      # return an nn.Module
    def compute_loss(self, model, batch, device): ...
    def compute_metrics(self, y_true, y_pred): ...  # {"accuracy": ...}
    def postprocess(self, raw_output): ...       # logits -> friendly dict
```

Optional hooks: `format_influx_record` (time-series schema),
`grafana_panel_hints`, `postprocess_with_llm`. See
`tasks/example_classifier/` for the complete reference.

`TASK_TYPE` (`classification` | `regression` | `detection` | `transcription`)
tells the framework how to split labels and which serve wrapper to use.
`feature_type` selects a feature extractor from the registry (see
[Extending](#extending-the-framework)).

Create any of them with the scaffolder:

```bash
python scripts/new_task.py wake_word                          # classification
python scripts/new_task.py loudness    --type regression
python scripts/new_task.py tagging     --type detection --feature mfcc
python scripts/new_task.py transcriber --type transcription   # serve-only
```

---

## 8. Quickstart

```bash
make setup                      # runtime dirs + .env
# audio_input/example_classifier/<label>/*.wav   (folder name = label)

# Local:
poetry install
dagster dev                     # http://localhost:3000  → materialize the task

# Docker — core (Dagster) only:
docker compose up -d

# Docker — core + streaming inference (:8000) + media server:
docker compose -f docker-compose.yml -f docker-compose.inference.yml up -d
```

Training output lands in `storage/example_classifier/`; the API auto-loads the
checkpoint.

---

## 9. Configuration reference

Key `config.yaml` fields (per task):

| Field                | Purpose                                                        |
|----------------------|----------------------------------------------------------------|
| `archetype` / `capabilities` | what the task contributes (see §3)                     |
| `task_name`          | unique id; matches `@register_task` and the folder name        |
| `input_dir`          | where labelled audio lives (`<input_dir>/<label>/*.wav`)       |
| `output_dir`         | where features/splits/checkpoints/outputs are written          |
| `feature_type`       | extractor name: `spectrogram`/`mel`/`logmel`/`mfcc`/`raw`/custom |
| `task_type`          | `classification`/`regression`/`detection`/`transcription`       |
| `serve_wrapper`      | (optional) override the serving wrapper explicitly             |
| `input_shape`        | model input shape, e.g. `[1, 128, 87]`                         |
| `num_classes` / `class_labels` | classification head + label names                    |
| `split_strategy`     | `group` (leak-safe, stratified) or `random`                    |
| `batch_size` / `epochs` / `learning_rate` / `optimizer` / `criterion` | training |
| `use_remote_storage` | sync artifacts to object storage (needs a backend enabled)     |

For ad-hoc batch scoring, `input_dir`/`output_dir` can also be overridden at
launch time via the inference job's run config (see §4) instead of editing the
config file.

---

## 10. Optional integrations

All env-driven (`env.example`, `MIGRATION.md`); each is omitted by default and
the code runs without it:

- **Object storage** — `STORAGE_BACKEND=minio` + `MINIO_*`.
- **Metrics (InfluxDB)** — `ENABLE_INFLUXDB=true` + `INFLUXDB_*`; writes are
  no-ops otherwise (no dashboards bundled).
- **Experiment tracking** — `MLFLOW_TRACKING_URI` (defaults to local `./mlruns`).
- **LLM** — `ENABLE_LLM=true` + `LLM_*` (any OpenAI-compatible endpoint).
- **Fusion** (transcript+event+sentiment timeline) — `ENABLE_FUSION=true`; an
  optional add-on in `extras/`, not core.

---

## 11. Project layout

```
audio_ops_dagster/
  capabilities.py            capability model + archetypes + resolver
  registry.py                @register_task + auto-discovery
  definitions.py             Dagster entry point (capability-driven assembly)
  assets.py                  the 6 per-task assets
  jobs.py                    per-task jobs (gated by capability)
  core/
    base_task.py             AudioTask abstract base (the task contract)
    features.py              pluggable feature-extractor registry
    audio_processor.py       segments audio, runs the chosen extractor
    audio_dataset.py         CSV+npy dataset + stratified group splits
    task_config_schema.py    config validation (clear early errors)
    metrics.py               metrics sink façade (+ NullMetricsResource)
    model_trainer.py         training loop (MLflow, early stopping, AMP)
    storage.py               S3StorageBackend + ArtifactType
    storage_resource.py      Dagster resource wrapping storage (local-first)
    influxdb_resource.py     metrics resource (no-op unless enabled)
    llm_client.py            optional generic LLM client
  streaming/                 RTSP sensor + stream engine
  monitoring/                influx writer helpers
  extras/                    OPTIONAL add-ons (not imported by default)
    fusion.py                transcript/event/sentiment timeline (ENABLE_FUSION)
  tasks/example_classifier/  the reference task
audio_ops_inference/         the serving daemon (importable package)
scripts/new_task.py          task scaffolder
mediamtx/                    media server config
tests/                       pytest suite (capabilities, splits, config, features)
docker-compose.yml           CORE: Dagster (training + batch inference)
docker-compose.inference.yml STREAMING inference service + media server
docker-compose.gpu.yml       GPU overlay (apply on top of either)
project-template/            copy this to start a thin project repo
PACKAGING.md                 use the framework as a package (thin project repos)
MIGRATION.md                 what's omitted + how to re-enable
IMPROVEMENTS.md              known limitations + roadmap
```

---

## Deployment: two units, one repo

Training and streaming inference have different operational profiles, so they're
separate **deployables** (not separate repos):

- **Core (Dagster)** — `docker-compose.yml`. Bursty, GPU-when-a-job-runs,
  scale-to-zero fine, lives near your data. Runs training, evaluation, and
  **batch inference** (the `<task>_inference_job`).
- **Streaming inference** — `docker-compose.inference.yml`. Long-running,
  latency-bound, scaled horizontally behind a load balancer, lives near your
  media ingress. Holds models in memory; reads checkpoints from shared storage
  or object storage.

```bash
docker compose up -d                                   # core only
docker compose -f docker-compose.inference.yml up -d   # streaming only (own host/env)
docker compose -f docker-compose.yml \
               -f docker-compose.inference.yml up -d    # both together
# add  -f docker-compose.gpu.yml  for an NVIDIA runtime
```

They share only the task contract and the artifact layout — so the same image
set deploys to two environments with independent GPU/scaling configs, without
the cost of keeping two repos in sync.

**Separate project repos.** The framework is an installable package, so each
project (e.g. one for training, one for serving) can be its own thin repo that
depends on it and brings only its own tasks. See **[`PACKAGING.md`](PACKAGING.md)**
and `project-template/`.

---

## Extending the framework

Three registries make new behavior plug-in, not fork:

- **Tasks** — `@register_task("name")` on an `AudioTask` subclass. Auto-discovered.
- **Feature extractors** — `@register_feature("name")` on a
  `(waveform, sample_rate) -> np.ndarray` callable. Select it with `feature_type:`.
- **Serve wrappers** — `@register_wrapper("type")` on a
  `(task, task_name, cfg) -> wrapper` factory, for new servable model kinds.

Plus a pluggable **metrics sink**: subclass `NullMetricsResource` and wire it in
`definitions.py`. Each is declared in your own task module — you never edit core.

Config is validated at build time, so a typo'd capability, a label/`num_classes`
mismatch, or an unknown feature surfaces immediately with a clear message.

Run the tests with `make test`.

---

## 12. Further improvements

See **[`IMPROVEMENTS.md`](IMPROVEMENTS.md)** for a prioritized list of known
limitations and suggested next steps (label-index consistency across splits,
unifying the two S3 clients, the model-reconstruction coupling in the API, a
config schema validator, and tests). None block normal use; they're the honest
rough edges worth addressing as the framework grows.
