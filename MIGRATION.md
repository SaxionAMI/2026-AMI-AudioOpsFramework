# Migration & Known Gaps

This repo is the **clean framework template** derived from the original
project repo. This file explains what was removed, how to add any of it back,
and what to double-check.

## What changed vs. the original

**Removed (project-specific or heavy infra):**
- Tasks `sentiment`, `nonverbal` (YAMNet), `speech_to_text` — collapsed into one
  reference task, `example_classifier`. The serve-only / transcription patterns
  they demonstrated are described below and supported by the same code paths.
- Grafana dashboards, the dashboard generator, and `config_influxdb.yaml`.
- Docker services for MinIO, MLflow, InfluxDB, Grafana, Label Studio, and Ollama.
- Project secrets/env actuals, sample datasets, and one-off scripts
  (`smoke_test.sh`, `extract_inference.sh`, `push_mic.sh`, `secrets_manager.py`,
  `config_loader.py`).

**Changed (so the core runs with zero external services):**
- Storage is **local-first**. `MinIOStorageResource` is disabled unless
  `STORAGE_BACKEND=minio`; the inference API loads checkpoints from local disk
  first and only tries object storage when that backend is selected.
- InfluxDB metrics are **off by default** (`ENABLE_INFLUXDB=false`); every write
  becomes a no-op, so the instrumented asset code keeps working untouched.
- MLflow logs to a local `./mlruns` folder — no server required.
- `docker-compose.yml` keeps only `dagster`, `dagster-daemon`, `inference`,
  `mediamtx`, and drops the GPU `deploy:` blocks for portability.

**Made optional (this revision):**
- `fusion.py` (transcript+nonverbal+sentiment timeline) moved from `core/` to
  `audio_ops_dagster/extras/fusion.py`. It is off by default and enabled in the
  API only with `ENABLE_FUSION=true`. It was project-specific (multimodal
  triage, with Dutch summary text) and is not part of the generic core.

**Kept (core):** Dagster orchestration, the capability model, the task registry,
the full inference API including RTSP/WebSocket/SSE streaming and the fusion
timeline (`core/fusion.py`, `core/llm_client.py` — both degrade gracefully).

## Turning optional systems back on

| System            | How                                                                 |
|-------------------|---------------------------------------------------------------------|
| Object storage    | `STORAGE_BACKEND=minio` + `MINIO_*` in `.env`; run your own MinIO. Set `use_remote_storage: true` in a task config to upload its artifacts. |
| Metrics (Influx)  | `ENABLE_INFLUXDB=true` + `INFLUXDB_*`; point at your InfluxDB v2.    |
| MLflow server     | Set `MLFLOW_TRACKING_URI=http://<host>:5000` (otherwise local files).|
| LLM fusion        | `ENABLE_LLM=true` + `LLM_*` (any OpenAI-compatible endpoint).        |
| Grafana / LabelStudio | Not bundled — add as your own compose services pointing at the data sources above. |

Each is additive and env-driven; no code changes are needed to enable them.

## Patterns the removed tasks demonstrated

- **Serve-only (no training), e.g. a pretrained transcriber or tagger:** give the
  task `archetype: pretrained_servable`. It contributes no Dagster assets and is
  loaded by the inference API. For streaming transcription, set the task's
  `TASK_TYPE = "transcription"` and a `stream_buffer_seconds` in its config — the
  API's STT wrapper path handles buffering.
- **Pretrained event detection (e.g. YAMNet-style):** set
  `task_type: nonverbal_pretrained` in the config; the API loads it through its
  pretrained wrapper. (You supply the model.)

## Core modules: resolved

Three modules were missing from the original archive; the **real versions have
now been supplied and integrated**, replacing the earlier reconstructions:

- `core/storage.py` — the boto3-based `S3StorageBackend` + `ArtifactType`
  (works with MinIO, AWS S3, Cloudflare R2, Backblaze B2).
- `core/storage_resource.py` — the Dagster `MinIOStorageResource` wrapping that
  backend. **One reconciliation was applied** for this template: `from_env()` is
  now local-first — storage is disabled unless `STORAGE_BACKEND=minio`/`s3` or
  `ENABLE_MINIO=true`. (The original defaulted on via `ENABLE_MINIO`.) Behavior
  is otherwise identical, and uploads still require a task's
  `use_remote_storage: true`.
- `core/audio_dataset.py` — the real pandas/sklearn dataset with the stratified
  **group** split (groups by `_seg`/`_chunk` prefix). Verified against the
  `AudioProcessor` layout: no group leakage across splits, class balance
  preserved.

Note: `S3StorageBackend.from_minio_config()` references the old `config_loader`
module (removed in cleanup). That classmethod is optional and unused by the
pipeline; ignore it or delete it unless you reintroduce `config_loader`.

## Validation status

- All modules compile.
- Capability resolution tested across configs.
- `create_splits_csv` tested end-to-end on synthetic features (group leakage =
  none, stratified across classes, `path` resolves correctly).
- Not run: a full training pass (needs real audio + torch). Do a first
  `dagster dev` run on a small sample before trusting a full pipeline.

See `IMPROVEMENTS.md` for known limitations worth addressing next.

## Genericization revision

The framework was made generic so any audio task type is first-class — no
hardcoded "classification + mel-spectrogram + three serving wrappers". New:

- Pluggable feature extractors (`core/features.py`, `@register_feature`).
- Serve-wrapper registry in the API (`@register_wrapper`); classifier serving
  builds from the task's `build_model()` (no torchvision reconstruction).
- One boto3 S3 client across training and serving.
- Metrics sink façade (`core/metrics.py`) selected by `ENABLE_INFLUXDB`.
- Task config validation (`core/task_config_schema.py`) at build time.
- Scaffolder generates classification/regression/detection/transcription tasks.
- `tests/` suite (`make test`).

See IMPROVEMENTS.md for what each change addressed and what remains.

## Config & deployment cleanup

- Removed the unused `config/` directory (`config.yaml`, `config_inference.yaml`)
  — nothing read it; the API is env-driven and tasks read their own config.yaml.
- Split compose into deployable units: `docker-compose.yml` (core/Dagster),
  `docker-compose.inference.yml` (streaming service + media server), and
  `docker-compose.gpu.yml` (GPU overlay). Run them separately or merged with
  multiple `-f` flags.
- `env.example` reorganized into Core / Storage / Metrics / Tracking / LLM+Fusion
  / Inference sections; `.dockerignore` now excludes runtime dirs, caches, tests.

