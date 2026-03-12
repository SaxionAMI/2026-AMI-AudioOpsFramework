# AMI Audio Ops Framework

A modular, task-pluggable audio ML platform. Drop in audio files, register a task, and get a fully orchestrated train → evaluate → serve pipeline — tracked in MLflow, stored in MinIO, visualized in Grafana.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        Your Audio Files                        │
│                    audio_input/<task>/*.wav                     │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                   Dagster  (port 3000)                         │
│                                                                │
│   raw_audio_files                                              │
│        │  glob .wav from input_dir                             │
│        ▼                                                       │
│   segmented_features                                           │
│        │  librosa → mel-spectrogram segments → .npy            │
│        ▼                                                       │
│   dataset_splits                                               │
│        │  group-aware stratified train/val/test CSVs           │
│        ▼                                                       │
│   trained_model  ──────────────────────────► MLflow (5000)     │
│        │  task.build_model() + ModelTrainer   experiments,     │
│        ▼                                      checkpoints      │
│   model_evaluation                                             │
│        │  test set metrics logged to MLflow                    │
│        ▼                                                       │
│    storage/<task>/checkpoints/best_model.pt                    │
└────────────────────────────────────────────────────────────────┘
                         │
                    checkpoint
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                 Inference Service (port 8000)                  │
│             FastAPI / uvicorn — loads from storage/            │
└────────────────────────────────────────────────────────────────┘

Supporting Services
───────────────────
MinIO   (9000/9001)  S3-compatible artifact & feature storage
InfluxDB    (8086)   Time-series audio and system metrics
Grafana     (3001)   Dashboards wired to InfluxDB
Label Studio(8080)   Audio annotation and labeling
```

---

## Services at a Glance

| Service       | Port(s)    | Purpose                              |
|---------------|------------|--------------------------------------|
| Dagster UI    | 3000       | Pipeline orchestration & scheduling  |
| MLflow        | 5000       | Experiment tracking, model registry  |
| Inference API | 8000       | FastAPI model serving endpoint       |
| MinIO API     | 9000       | S3-compatible object storage         |
| MinIO Console | 9001       | MinIO web UI                         |
| InfluxDB      | 8086       | Time-series metrics database         |
| Grafana       | 3001       | Dashboards                           |
| Label Studio  | 8080       | Audio data labeling                  |

---

## How It Works

### The Task Plugin System

Each audio task lives in its own folder under `audio_ops_dagster/tasks/`. Adding a new task requires two files:

```
audio_ops_dagster/tasks/
└── my_task/
    ├── config.yaml       ← task parameters (loaded by definitions.py)
    └── task_my_task.py   ← @register_task class implementing AudioTask
```

`definitions.py` auto-discovers every task folder that has a `config.yaml`, creates a typed `AudioTaskResource` from it, and generates the full 5-asset DAG plus two named jobs (`<task>_full_training_job`, `<task>_data_prep_job`) — all without touching any other file.

### Task Registration

```python
# audio_ops_dagster/tasks/my_task/task_my_task.py

from audio_ops_dagster.core.base_task import AudioTask
from audio_ops_dagster.registry import register_task

@register_task("my_task")
class MyTask(AudioTask):
    def build_model(self, input_shape):
        # Return a nn.Module
        ...

    def compute_metrics(self, y_true, y_pred):
        # Return {"accuracy": 0.95, ...}
        ...

    def postprocess(self, raw_output):
        # Format model output for the API
        ...
```

### Task Configuration (`config.yaml`)

Every field maps directly to `TaskConfig` in `registry.py`. Only the fields relevant to your task are required.

```yaml
# tasks/my_task/config.yaml
feature_type: spectrogram       # spectrogram | mfcc | raw
input_dir: "audio_input/my_task/data"
split_strategy: "group"         # group | random
num_classes: 4
class_labels:
  - class_a
  - class_b
  - class_c
  - class_d
model_architecture: resnet18
batch_size: 32
epochs: 20
learning_rate: 0.0005
optimizer: adam
criterion: cross_entropy
```

### Audio Processing Pipeline

`AudioProcessor` handles the raw-audio → feature transformation:

1. Loads `.wav` (and `.mp3`) files with `librosa` at 16 kHz
2. Slices each file into fixed-length segments (default 1 s, 25% overlap)
3. Extracts features per segment:
   - `spectrogram` → 128-bin mel-spectrogram, log-scaled, normalized to `[0, 1]`
   - `mfcc` → (extend `_extract` to add this)
   - `raw` → raw waveform array
4. Saves each segment as a `.npy` file under `storage/<task>/features/<label>/`

The **label** for each segment is inferred from its parent folder name — matching the standard audio classification layout:

```
audio_input/sentiment/
├── approval/
│   ├── file1.wav
│   └── file2.wav
├── disapproval/
│   └── file3.wav
└── neutral/
    └── file4.wav
```

### Dataset Splitting

`AudioDataset.create_splits_csv` produces train/val/test CSVs. Two strategies are supported:

- **`group`** (recommended for segmented audio): Splits by *source file* so segments from the same recording can't appear in both train and test. Uses stratified sampling to maintain class balance.
- **`random`**: Stratified random split by segment. Faster but risks data leakage from over-segmented recordings.

The group ID is parsed by splitting on `_seg` or `_chunk` in the filename, e.g. `speaker01_seg_003.npy` → group `speaker01`.

### Training & MLflow Logging

`ModelTrainer.train()` runs the standard PyTorch training loop and logs everything to MLflow:

- **Params**: all `TaskConfig` fields logged at run start
- **Metrics**: `train_loss` and `val_loss` per epoch; task-specific metrics (accuracy, F1, WER, etc.) per epoch
- **Artifacts**: best checkpoint `.pt` file, plus the full model via `mlflow.pytorch.log_model()`
- **Model Registry**: model is registered under `<TaskClass>_Model`

Run names are auto-generated: `SENTIMENT_resnet18_spectrogram_lr0.001_bs32_143022`

---

## Adding a New Task — Checklist

1. Create `audio_ops_dagster/tasks/<your_task>/`
2. Add `__init__.py` (empty is fine)
3. Add `config.yaml` with at least `feature_type`, `num_classes`, and `class_labels`
4. Add `task_<your_task>.py` with `@register_task("<your_task>")` on your `AudioTask` subclass
5. Put your audio files in `audio_input/<your_task>/`
6. Restart Dagster — your assets and jobs appear automatically

---

## Included Task: Sentiment

Classifies crowd/speaker audio into **approval / disapproval / neutral**.

- **Model**: ResNet18 (pretrained ImageNet), first conv adapted for single-channel mel-spectrogram input, custom classifier head with dropout
- **Input**: Mel-spectrogram segments (128 × ~87 time steps at 1 s / 16 kHz)
- **Metrics**: weighted accuracy and F1

Config: `tasks/sentiment/config.yaml`

---

## Project Structure

```
.
├── audio_ops_dagster/              Dagster orchestration service
│   ├── Dockerfile.dagster
│   ├── dagster_pyproject.toml      Full ML stack deps (torch, mlflow, etc.)
│   ├── definitions.py              Auto-discovers tasks, builds Definitions
│   ├── registry.py                 @register_task decorator + TaskConfig
│   ├── assets.py                   5-asset DAG factory (build_audio_assets)
│   ├── jobs.py                     Job factory (full_training, data_prep)
│   ├── core/
│   │   ├── base_task.py            AudioTask ABC
│   │   ├── audio_processor.py      Segmentation + feature extraction
│   │   ├── audio_dataset.py        PyTorch Dataset + CSV split logic
│   │   └── model_trainer.py        Train loop + MLflow integration
│   └── tasks/
│       ├── sentiment/
│       │   ├── config.yaml
│       │   └── task_sentiment.py
│       └── speech_to_text/         Stub — ready to implement
│           ├── config.yaml
│           └── model.py
│
├── audio-ops-inference/            FastAPI inference service
│   ├── Dockerfile.inference
│   └── inference_pyproject.toml   Minimal serving deps only
│
├── mlflow/
│   └── Dockerfile.mlflow
│
├── config/
│   ├── config.yaml                 Main typed config (all services)
│   ├── config_minio.yaml
│   ├── config_mlflow.yaml
│   ├── config_influxdb.yaml
│   └── config_inference.yaml
│
├── config_loader.py               Typed config dataclasses (AudioOpsConfig)
├── secrets_manager.py             .secrets.env loader with typed getters
├── docker-compose.yml
├── workspace.yaml                 Points Dagster at definitions.py
├── Makefile                       Dev shortcuts
├── setup.sh                       First-run setup script
├── pyproject.toml                 Root shared deps (pyyaml, pydantic, etc.)
│
├── .env                           Runtime config (ports, hosts, feature flags)
├── .secrets.env                   Credentials (never commit)
│
└── audio_input/                   Mount your audio here (created by setup)
    └── sentiment/
        ├── approval/
        ├── disapproval/
        └── neutral/
```

---

## Configuration System

Config loads in priority order: **environment variables → `config/config.yaml` → dataclass defaults**.

```python
from config_loader import get_config
config = get_config()              # Singleton; loads YAML then falls back to env

from secrets_manager import get_secrets_manager
secrets = get_secrets_manager()    # Loads .secrets.env
creds = secrets.get_minio_credentials()
```

The `AudioOpsConfig` dataclass covers all services (Dagster, MLflow, MinIO, InfluxDB, Inference, Label Studio, Grafana, Storage, Logging, Security) and can be exported/imported as YAML or JSON.

### Environment Files

| File | Purpose | Commit? |
|------|---------|---------|
| `.env` | Ports, hostnames, feature flags, non-secret settings | No |
| `.secrets.env` | Passwords, tokens, API keys | **Never** |
| `config/config.yaml` | Typed service config (mirrors .env but as YAML) | Yes (no secrets) |

---

## GPU Support

Both `dagster` and `dagster-daemon` include an NVIDIA GPU reservation in `docker-compose.yml`. The `AudioTask` base class and `ModelTrainer` both call `torch.cuda.is_available()` at runtime — no code changes needed to run on CPU instead.

To disable GPU reservation (CPU-only), remove or comment the `deploy.resources` block from the dagster services in `docker-compose.yml`.

---

## Development Workflow

```bash
make setup        # First time: create dirs + copy config templates
make validate     # Check .env and required vars are set
make up           # docker-compose up -d
make health       # Curl all service health endpoints
make logs         # Follow all service logs
make logs-dagster # Dagster only
make shell-dagster # bash into the dagster container

make backup       # tar.gz snapshot of all data dirs
make clean        # docker-compose down
make clean-all    # down + remove volumes (destructive)
```

### Running Locally Without Docker

```bash
# Install dagster service deps
cd audio_ops_dagster
poetry install

# Set env
export DAGSTER_HOME=$(pwd)/../dagster_home
export MLFLOW_TRACKING_URI=http://localhost:5000

# Launch
dagster-webserver -w ../workspace.yaml
```

---

## Makefile Reference

| Command | Description |
|---------|-------------|
| `make setup` | Create dirs, copy .env and .secrets.env from examples |
| `make validate` | Check config files and required vars |
| `make up` / `make down` | Start / stop all services |
| `make restart` | down + up |
| `make rebuild` | `--no-cache` Docker build |
| `make health` | Curl all endpoints |
| `make logs` | All logs |
| `make logs-<service>` | Single service logs |
| `make shell-<service>` | Interactive bash in container |
| `make backup` | Archive data directories |
| `make export-config` | Dump config to `config/config-export.yaml` |
| `make clean` | Remove containers |
| `make clean-volumes` | Remove containers + volumes |
| `make clean-all` | Remove everything (requires confirmation) |