# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.2.0] — 2026-11-26

Simplification release: less to learn, less to write, ~2,000 fewer lines.

### Added
- **Ready-made task bases**: `ClassificationTask`, `RegressionTask`,
  `DetectionTask` implement the whole contract (`SmallAudioCNN`, loss from
  config, standard metrics, output format). A task overrides only what differs.
- **Config-only tasks**: a folder with just `config.yaml` is a complete
  classification/regression/detection task (`audio-ops new-task X --config-only`).
- **`from audio_ops import …`**: one public import for tasks, features, serve
  wrappers and config (lazy; `import audio_ops` doesn't load torch).
- **`audio-ops run TASK`**: headless train / evaluate / batch-infer with compact
  logs and a summary of the artifacts; `--input-dir/--output-dir` overrides.
- **`audio-ops predict TASK FILE...`**: score files without a server
  (`--segments`, `--json`, `--checkpoint`).
- `./tasks` in the working directory is discovered automatically.
- `class_labels` may be omitted: sorted folder names are used and stored in the
  checkpoint; evaluation, batch inference and serving read them back.
- `core.runtime.load_task_model`: build a task's model and load its checkpoint
  in one call (used by Dagster, the CLI and the API).
- `scripts/setup.sh`: one-command setup of a checkout (virtual environment,
  PyTorch for the machine's hardware, the framework, `.env`, demo audio), with
  clear messages when something is missing. `make setup` runs it.
- `project-template/` now has its own `scripts/setup.sh` and `Makefile`. The
  script reads the framework dependency from `pyproject.toml` and gets it over
  HTTPS, over SSH (for private repositories) or from a checkout next to the
  project, whichever works.
- `group_pattern` (config): a regex that takes the recording id from the file
  name, so files cut from one long recording (`match3_001.wav`,
  `match3_002.wav`) stay in the same split.

### Changed
- **One `TaskConfig`** (pydantic, in `core/config.py`) replaces four parallel
  definitions (dataclass, validation schema, Dagster resource fields,
  `TrainConfig`). Validation errors and defaults now come from one place.
- `num_classes` is derived from `class_labels`; `criterion` defaults by task
  type (cross-entropy / BCE / MSE).
- `ModelTrainer(task, config=None, *, checkpoint_dir, metrics, run_id, context)`
  with `train(model, train_loader, val_loader)` / `test(model, loader)`.
- The Dagster task resource carries the full config as JSON (`config_json`).
- `audio-ops dev` keeps run history in `./dagster_home` unless `DAGSTER_HOME` is set.
- Batch inference logs progress every 10 % plus a prediction summary instead
  of one line per segment.
- The example task subclasses `ClassificationTask`; its output uses `scores`
  (not `probabilities`).

### Fixed
- Batch-inference points sent to InfluxDB were lost when the run process exited
  before the background writer flushed. The InfluxDB resource now flushes at
  the end of every run.
- The streaming service wrote `audio_duration_s=0` for classifier results.
- `audio-ops dev` failed with "Dagster is not installed" when the virtual
  environment wasn't activated (for example `.venv/bin/audio-ops dev`).

### Documentation
- New pages *Comparison with other tools* and *Used in* (the GRASS and AI-ZI
  projects at Saxion, with a notice that Audio Ops is not a medical device).
- README, docs home and concepts page rewritten in plainer language.
- Copyright holder is Saxion University of Applied Sciences; links point to
  `SaxionAMI/2026-AMI-AudioOpsFramework`.

### Removed
- The Dagster MediaMTX stream-sync sensor (`ENABLE_STREAM_SYNC`); the API's
  `STREAM_AUTODISCOVER` does the same with one setting.
- The unused `streaming.rtsp_consumer`, the third InfluxDB writer
  (`monitoring.influx_writer`), `TrainConfig`, `DirectoryAudioDataset`,
  `build_data_loaders`, the API's legacy `TASKS_DIR`, and the `requests` dependency.

## [1.1.0] — 2026-11-24

First public, open-source release.

### Added
- **`audio-ops` CLI**: `new-task`, `list`, `validate`, `demo-data`, `serve`,
  `dev`, `doctor`, `version`. `audio-ops-serve` and `scripts/new_task.py` keep
  working as thin wrappers.
- **Synthetic demo data** (`audio-ops demo-data`) so the quickstart runs with no
  dataset; also writes a streaming demo WAV for MediaMTX / `GET /demo`.
- **Real `eval_only`**: evaluates an existing checkpoint (`checkpoint_path`)
  without training; new `<task>_evaluation_job`.
- **Evaluation report** `outputs/evaluation_<task>.json`, plus metrics, paths and
  counts as Dagster asset metadata.
- **Inference API**: optional API-key auth (`API_KEY`), configurable CORS,
  upload size limit, `POST /models/reload`, `?tasks=` filter on `/infer/file`,
  skipped-task reasons in `/models`, version in `/health`.
- **File inference is segmented like training** and returns per-segment
  results with timestamps plus an aggregate.
- Config validation: enum/range checks, split ratios, duplicate or boolean
  labels, `class_weights` length, and *did-you-mean* hints for misspelled keys.
- Feature-shape check that warns when `input_shape` doesn't match the features.
- Feature cache manifest: changing `feature_type`/`segment_*`/`sample_rate`
  automatically recomputes features instead of silently reusing stale ones.
- External task folders are imported as packages (relative imports work);
  `AUDIO_OPS_DISABLE_BUILTIN_TASKS` hides the bundled example.
- Configurable `split_ratios`, `seed`, `num_workers`, `label_smoothing`,
  `mixed_precision`, `checkpoint_path`, `transcription_chunk_seconds`.
- Regression targets from numeric label folders; multi-label detection
  scaffold that trains out of the box.
- Fusion language packs (`FUSION_LANGUAGE=en|nl`, English default).
- `DEVICE=auto|cpu|cuda|cuda:N|mps` honoured by training, batch inference and serving.
- Open-source project files: Apache-2.0 `LICENSE`, `NOTICE`, `THIRD_PARTY.md`,
  `CONTRIBUTING.md` (DCO), `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`,
  issue/PR templates, Dependabot, pre-commit.
- CI (lint, tests on 3.10–3.13, serve-only install check, package build, docs
  build), PyPI release via Trusted Publishing, docs deployment to GitHub Pages.
- Documentation site built with Zensical (`docs/`, `zensical.toml`).
- Test suite grew from 12 to 75 tests, including end-to-end pipeline, eval-only,
  batch-inference override, API, WebSocket streaming and scaffold-and-train
  tests for every task type.

### Changed
- **The trainer uses the task contract**: loss from `task.compute_loss`
  (which may return `(loss, outputs)`), optimizer from `task.get_optimizer`.
  Regression and detection tasks now train with their own loss instead of
  cross-entropy. `optimizer`, `weight_decay`, `criterion`, `class_weights` and
  `label_smoothing` in `config.yaml` are honoured (previously ignored).
- Default `optimizer` is `adamw` (what the trainer always used); configs that
  set `optimizer: adam` now really get Adam.
- **Serving computes features exactly like training** (same extractor, sample
  rate and `segment_duration` window, same input rank) and passes model output
  through the task's own `postprocess`. Previously serving always used the
  `spectrogram` extractor, padded/cropped to `input_shape` and applied a softmax.
- **Each stream gets its own wrapper instances** (shared weights, private
  buffers): audio from different streams can no longer mix in STT or event buffers.
- Inference service split into `settings`, `wrappers`, `store`, `influx` and
  `api` modules. `from audio_ops_inference.wrappers import register_wrapper`
  works without starting the app; old imports from `api` still work.
- Batch inference: per-file segment offsets (`start_sec`/`end_sec` from the real
  hop), result keys are `<label>/<file>.npy` (no collisions across folders), and
  transcription transcribes each recording in `transcription_chunk_seconds`
  windows (no duplicated audio when segments overlap).
- Segmentation no longer emits a near-empty trailing segment.
- Group splits use only the trailing `_seg_NNN` suffix and are scoped per label;
  file order is sorted so splits are reproducible across machines.
- MLflow defaults to local SQLite (`sqlite:///mlruns/mlflow.db`; MLflow 3
  refuses the old `./mlruns` file store) and tracking failures never fail training.
- Checkpoints carry the real feature config (`feature_type`, `sample_rate`,
  `segment_duration`, `segment_overlap`) and are written atomically.
- One Dagster resource per task now carries **every** config key (unknown keys
  in `extra_json`), so task code sees the same config in Dagster and in the API.
- The MediaMTX → API stream-sync sensor is opt-in (`ENABLE_STREAM_SYNC=true`),
  starts stopped, sends the API key, and never removes manual/WebSocket streams.
- Packaging: PEP 621 metadata with a hatchling build backend (installable with
  pip, uv or Poetry); extras `train`, `serve`, `ast`, `s3`, `metrics`, `all`,
  `dev`, `docs`. `torch>=2.4` has no upper cap. `boto3` and `influxdb-client`
  moved to the `s3`/`metrics` extras; unused `structlog`, `python-json-logger`,
  `pydantic-settings`, `python-dotenv`, `minio`, `scipy` and `torchvision`
  dependencies removed.
- Docker: images build from the package with CPU PyTorch by default
  (`TORCH_INDEX_URL` build arg for CUDA) and `EXTRA_PIP` for task runtimes; the
  inference image runs as a non-root user; GPU overlays split per deployable;
  obsolete compose `version:` keys removed.
- Generic defaults instead of project-specific ones: demo paths, MediaMTX
  streams, STT hallucination filters (English + Dutch, extendable via
  `STT_HALLUCINATION_PHRASES`), fusion prompts.
- `print` replaced by logging (`AUDIO_OPS_LOG_LEVEL`).

### Fixed
- `input_shape` never reached tasks (it was not a `TaskConfig` field).
- `pretrained_batch` tasks failed to load in Dagster (dangling dependency on a
  non-existent `trained_model` asset).
- Default `feature_type` differed between validation (`spectrogram`), the
  resource (`mel`) and definitions (`raw_waveform`, which is not registered).
- `task_name` in config vs. folder name could disagree between Dagster and the
  API; the folder name is now the identity and mismatches are reported.
- RTSP consumer overlap truncated fresh audio.
- The "AST feature extraction" log message for every feature type; duplicate
  "Loaded checkpoint" logs; `asyncio.get_event_loop()` deprecation in the API.
- Stream health metrics were only written when auto-discovery was on.
- Placeholder SSE handles accumulated forever.

### Removed
- Legacy checkpoint loaders that globbed `MODELS_DIR`/the bucket for any `.pt`,
  rebuilt torchvision architectures and unpickled files with
  `weights_only=False`. Serving builds models from the task's `build_model()`
  only. Pickled checkpoints require `ALLOW_UNSAFE_CHECKPOINTS=true`.
- `S3StorageBackend.from_minio_config` (depended on a removed module), the empty
  `stream_inference_engine.py`, a stale nested `framework.zip`, `poetry.lock`,
  `setup.sh` (use `make setup`).
- Hard-coded default credentials (`minioadmin`, an InfluxDB token): unset
  credentials fall through to the client's default chain.

### Upgrade notes
- Feature directories without a manifest are recomputed once.
- If you relied on the trainer's implicit cross-entropy + label smoothing 0.1,
  set `label_smoothing: 0.1` and make sure `compute_loss` uses
  `self.get_loss_function()`.
- If you used `INFERENCE_CHECKPOINT` for one task among many, prefer
  `checkpoint_path` in that task's `config.yaml`.
- Batch-inference JSON/CSV keys are now relative paths (`class_a/x_seg_000.npy`).
- MLflow runs in an old `./mlruns` file store are not read automatically; use
  `mlflow migrate-filestore` or set `MLFLOW_TRACKING_URI` + `MLFLOW_ALLOW_FILE_STORE=true`.

## [1.0.0] — 2026-07

Internal release: capability model, task registry, pluggable features,
serve-wrapper registry, local-first storage, optional InfluxDB/MLflow/LLM/fusion,
split core/inference compose files.
