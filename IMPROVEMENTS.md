# Improvements

What's been made generic, and what's left.

## Done — the framework no longer hardcodes these

- **Pluggable feature extraction.** Features are a registry
  (`core/features.py`): `spectrogram`, `mel`/`logmel`, `mfcc`, `raw` built in,
  and `@register_feature("name")` for your own. `AudioProcessor` just runs the
  one a task names in `feature_type`. No more mel-only assumption.
- **Serve-wrapper registry.** The inference API dispatches by a registry
  (`@register_wrapper`) keyed on task type, not three hardcoded `if` branches.
  Built-ins: `classifier`, `transcription`, `pretrained_event`. Add a new
  servable kind by registering a factory.
- **Generic checkpoint loading.** Classifier serving builds the network from the
  task's own `build_model()` and loads weights — the torchvision
  architecture-name reconstruction is gone for task-backed models.
- **One S3 client.** Training and serving both use the boto3 `S3StorageBackend`
  (MinIO/S3/R2/B2). The `minio`-library path in the API is removed.
- **Pluggable metrics sink.** Assets talk to a metrics resource, not InfluxDB.
  `definitions.py` injects `InfluxDBMetricsResource` or a `NullMetricsResource`
  based on `ENABLE_INFLUXDB`. Add a sink by subclassing the null one.
- **Config validation.** `core/task_config_schema.py` checks each task's
  `config.yaml` at build time (capabilities resolve, feature type known,
  num_classes/class_labels agree) and prints clear errors/warnings instead of
  failing deep in a run.
- **Any task type, scaffolded.** `scripts/new_task.py --type
  {classification,regression,detection,transcription} --feature {…}` generates a
  working, compilable stub for each.
- **`is_sequence` and label indices are generic** (from `TASK_TYPE` and
  `class_labels`, not task-name allowlists or per-CSV inference).
- **Naming collision resolved.** The trainer's directory dataset is now
  `DirectoryAudioDataset`, distinct from `core/audio_dataset.AudioDataset`.
- **Tests.** `tests/` covers capability resolution, the feature registry, config
  validation, and split-leakage/stratification. `make test`.

- **Batch inference is first-class in Dagster.** The `<task>_inference_job`
  runs only `raw_audio → features → inference_results` (no training, no daemon),
  and accepts run config (`input_dir`/`output_dir`) so you can score an arbitrary
  folder from the Launchpad without editing config or writing a script. Results
  are written as JSON **and** CSV under `<output_dir>/outputs/`.

- **Deployment split + GPU overlay.** Core (Dagster) and streaming inference are
  now separate compose files (`docker-compose.yml`, `docker-compose.inference.yml`),
  with a `docker-compose.gpu.yml` overlay — one repo, two deployables with
  independent GPU/scaling. Removed the dead `config/` dir; tidied env.example
  and .dockerignore.

## Remaining

### Full training/serving decoupling (artifact contract)
Serving still imports task modules (to call `build_model`). For a clean split
into separate training and serving repos, export an ONNX/TorchScript artifact +
sidecar config as the last training step and load *that* at serve time. Then the
serving image needs only a runtime, not the task code. (The torchvision coupling
is already gone; this is the remaining, optional step.)

### Lower priority
- **`InferenceDataset` determinism** — sort the `rglob` file list.
- **`eval_only` from a checkpoint** — `model_evaluation` depends on the
  `trained_model` asset; true external-checkpoint eval needs a path input.
- **API auth** — `API_SECRET_KEY` is referenced but not enforced on endpoints.
