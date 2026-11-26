# Python API

The stable extension surface. Everything else is internal and may change in
minor releases.

## `audio_ops`: the public API

Everything a task, feature or serve wrapper needs:

```python
from audio_ops import (
    AudioTask, ClassificationTask, RegressionTask, DetectionTask, SmallAudioCNN,
    register_task, get_task, list_tasks,
    register_feature, list_features,
    register_wrapper, ServeWrapper,
    TaskConfig, validate_task_config, load_task_model, LLMClient,
)
```

Names load lazily (`import audio_ops` doesn't import torch). The modules below
are where they live; deeper imports keep working but may change in minor releases.

## Tasks: `audio_ops_dagster.registry`, `audio_ops_dagster.core.base_task`

| Name | |
|---|---|
| `AudioTask` | base class, see [Writing tasks](../guide/tasks.md) |
| `register_task(name)` | class decorator (or `register_task(name, cls)`) |
| `get_task(name, config)` | instantiate a registered task (`config`: `TaskConfig` or dict) |
| `list_tasks()` | registered names |
| `TaskConfig` / `TaskConfig.from_dict(d, task_name=)` | the validated config (pydantic) with defaults; unknown keys become attributes |
| `iter_task_dirs()` / `check_task(t)` | discovery and validation used by Dagster, the API and the CLI |
| `discover_tasks()` | import task modules (idempotent; honours `AUDIO_OPS_TASKS_DIR` at call time) |

## Capabilities: `audio_ops_dagster.capabilities`

`TRAIN`, `EVALUATE`, `BATCH_INFER`, `SERVE`, `ARCHETYPES`,
`resolve_capabilities(cfg)`, `serves(cfg)`, `produces_dagster_assets(caps)`,
`describe(caps)`, `CapabilityError`.

## Features: `audio_ops_dagster.core.features`

`register_feature(name)`, `get_feature_extractor(name)`, `list_features()`,
`has_feature(name)`, `FeatureError`.

## Audio helpers

| Name | |
|---|---|
| `core.audio_processor.AudioProcessor` | `segments(y)`, `extract(window)`, `process_directory(in, out)` |
| `core.audio_dataset.AudioDataset` | CSV dataset; `create_splits_csv(...)`, `scan_features(dir)` |
| `core.runtime` | `load_task_model`, `resolve_device`, `to_model_input`, `load_checkpoint_into`, `read_checkpoint`, `default_checkpoint_path`, `list_audio_files`, `segment_name`, `parse_segment_name` |
| `core.model_trainer.ModelTrainer(task, config=None, *, checkpoint_dir, metrics=None, run_id="", context=None)` | `.train(model, train_loader, val_loader)` → best checkpoint path; `.test(model, loader)` → `(loss, metrics)` |
| `core.task_types` | `ClassificationTask`, `RegressionTask`, `DetectionTask`, `BASES_BY_TYPE` |
| `core.models.SmallAudioCNN(out_dim, channels=32, dropout=0.3)` | default shape-agnostic CNN |
| `core.task_config_schema.validate_task_config(cfg)` | `(errors, warnings)` |
| `demo.write_dataset`, `demo.write_stream_demo` | synthetic audio |

## Dagster: `audio_ops_dagster.definitions`, `.assets`, `.jobs`

`build_definitions(loggers=None)` (re-reads the environment), `defs`,
`build_audio_assets(task, resource_key, caps)`, `build_audio_jobs(task, caps)`,
`AudioTaskResource`, `InferenceRunConfig`.

## Serving: `audio_ops_inference.wrappers`

`register_wrapper(type)`, `ServeWrapper`, `ModelWrapper`, `STTModelWrapper`,
`PretrainedEventWrapper` (alias `YamnetModelWrapper`), `resolve_wrapper_type`,
`get_wrapper_factory`, `list_wrapper_types`.

`audio_ops_inference.api.app` is the ASGI application
(`uvicorn audio_ops_inference.api:app`).

## Integrations

`core.storage.S3StorageBackend`, `core.storage.ArtifactType`,
`core.storage_resource.MinIOStorageResource`, `core.metrics.NullMetricsResource`,
`core.influxdb_resource.InfluxDBMetricsResource`, `core.llm_client.LLMClient`,
`core.tracking.Tracker`, `extras.fusion.TranscriptFuser`.
