# Architecture

```text
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

   Optional, off by default: InfluxDB metrics · MLflow server · LLM · MediaMTX · fusion
```

## Packages

| Package | Role |
|---|---|
| `audio_ops_dagster` | task contract, registry, capability model, features, trainer, Dagster assets/jobs, CLI |
| `audio_ops_inference` | the FastAPI service: settings, serve wrappers, model store, streaming |

`audio_ops_inference` imports the shared core (registry, features, runtime
helpers) but **never Dagster**, so the `serve` extra runs without it.

## Shared contract

The runtimes share three things only:

1. **Task discovery**: `registry.iter_task_dirs()` scans the built-in tasks
   and `AUDIO_OPS_TASKS_DIR`. The folder name is the task's identity.
2. **The task class**: the API rebuilds the network with the task's own
   `build_model()` and formats output with its own `postprocess()`.
3. **The artifact layout**: `<output_dir>/checkpoints/best_model.pt`
   (weights-only, with feature metadata), mirrored to
   `{task}/models/best_model.pt` in object storage.

Train/serve parity lives in shared code: `core/audio_processor.py`
(segmentation + feature extraction) and `core/runtime.py` (input shaping,
device selection, checkpoint loading) are used by training, batch inference and
the API alike.

## Two deployables, one repo

| | Core (Dagster) | Streaming inference |
|---|---|---|
| Compose file | `docker-compose.yml` | `docker-compose.inference.yml` |
| Profile | bursty, GPU while a job runs, scale-to-zero fine | long-running, latency-bound |
| Scales by | run launcher / bigger box | more containers behind a load balancer |
| Lives near | your data | your media ingress |

See [Deployment](../guide/deployment.md). Each *project* can also be its own
thin repo that depends on the framework. See [Project repos](../guide/project-repos.md).

## Code map

```text
audio_ops/                   the public API (lazy re-exports), import from here
audio_ops_dagster/
  capabilities.py            capability model + archetypes + resolver
  registry.py                @register_task, discovery (./tasks, AUDIO_OPS_TASKS_DIR), check_task
  definitions.py             Dagster entry point (capability-driven assembly)
  assets.py / jobs.py        the per-task assets and jobs
  cli.py / scaffold.py       the `audio-ops` command and task templates
  demo.py                    synthetic demo audio
  core/
    config.py                TaskConfig: every config key, defined once
    base_task.py             AudioTask: the task interface
    task_types.py            ClassificationTask / RegressionTask / DetectionTask
    models.py                SmallAudioCNN
    features.py              pluggable feature-extractor registry
    audio_processor.py       segmentation + feature extraction + cache manifest
    audio_dataset.py         CSV + .npy datasets, leak-safe splits
    runtime.py               devices, input shaping, segment names, checkpoints
    model_trainer.py         training loop (AMP, early stopping, resume)
    tracking.py              optional MLflow tracking
    task_config_schema.py    readable validation messages + extra checks
    metrics.py / influxdb_resource.py   metrics sinks (no-op / InfluxDB)
    storage.py / storage_resource.py    S3-compatible storage (local-first)
    llm_client.py            optional OpenAI-compatible / Ollama client
  extras/fusion.py           optional transcript + event + sentiment timeline
  tasks/example_classifier/  the reference task
audio_ops_inference/
  settings.py                every env var the service reads
  wrappers.py                serve-wrapper registry + built-in wrappers
  store.py                   discovers servable tasks, loads checkpoints
  api.py                     FastAPI app, streaming, SSE/WS
  influx.py                  optional metrics writer
```
