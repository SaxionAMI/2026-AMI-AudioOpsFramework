# Audio Ops Framework — Quick Start

> From clone to a running pipeline in about 10 minutes.

---

## Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Docker | 20.10+ | `docker --version` |
| Docker Compose | v2 or `docker-compose` | `docker compose version` |
| Python | 3.10+ | `python3 --version` |
| bash | any | — |
| NVIDIA GPU | optional | `nvidia-smi` |

---

## Step 1 — Run Setup

```bash
chmod +x setup.sh
./setup.sh
```

This script does six things:

1. Checks Docker, Docker Compose, and Python 3 are installed
2. Creates all required data directories (`dagster_home/`, `mlflow/`, `minio_storage/`, `influxdb_data/`, `grafana_data/`, `label_studio_data/`, `logs/`)
3. Copies `.env.example` → `.env` and `.secrets.env.example` → `.secrets.env`
4. Ensures `.env` and `.secrets.env` are in `.gitignore`
5. Installs Poetry (if missing)
6. Installs root Python dependencies via `poetry install`

---

## Step 2 — Set Your Secrets

Open `.secrets.env` and change every `change_me_in_production` placeholder:

```bash
nano .secrets.env
```

Minimum required changes:

```env
MINIO_ROOT_PASSWORD=some_strong_password
INFLUXDB_ADMIN_PASSWORD=some_strong_password
INFLUXDB_ADMIN_TOKEN=some_random_token_string
GRAFANA_ADMIN_PASSWORD=some_strong_password
API_SECRET_KEY=some_long_random_string
```

Lock down the permissions:

```bash
chmod 600 .secrets.env
```

> **Never commit `.secrets.env` to version control.** The `.gitignore` set up by `setup.sh` already excludes it.

---

## Step 3 — (Optional) Adjust Ports or Settings

Edit `.env` if you need to change default ports or toggle features:

```bash
nano .env
```

Defaults:

```
DAGSTER_PORT=3000
MLFLOW_PORT=5000
INFERENCE_PORT=8000
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001
INFLUXDB_PORT=8086
GRAFANA_PORT=3001
LABEL_STUDIO_PORT=8080
```

Feature flags (set to `false` to skip starting a service):

```
ENABLE_MLFLOW=true
ENABLE_LABEL_STUDIO=true
ENABLE_INFERENCE=true
ENABLE_GRAFANA=true
ENABLE_MINIO=true
```

---

## Step 4 — Validate Configuration

```bash
make validate
```

Expected output:

```
✓ .env found
✓ .secrets.env found

Checking required variables...
✓ MINIO_ROOT_USER
✓ INFLUXDB_ADMIN_PASSWORD
✓ GRAFANA_ADMIN_PASSWORD

Configuration validation complete!
```

---

## Step 5 — Start Services

```bash
make up
# or
docker-compose --env-file .env up -d
```

Wait ~15 seconds for all containers to become healthy, then verify:

```bash
make health
```

You should see responses from all seven endpoints:

```
Dagster:    http://localhost:3000  ✓
MLflow:     http://localhost:5000  ✓
Inference:  http://localhost:8000  ✓
InfluxDB:   http://localhost:8086  ✓
MinIO:      http://localhost:9001  ✓
Grafana:    http://localhost:3001  ✓
Label Studio: http://localhost:8080  ✓
```

---

## Step 6 — Add Your Audio Files

The audio processor expects files organized by label in subdirectories:

```
audio_input/
└── sentiment/                ← task name (matches tasks/sentiment/config.yaml)
    ├── approval/
    │   ├── crowd_cheer_01.wav
    │   └── crowd_cheer_02.wav
    ├── disapproval/
    │   └── crowd_boo_01.wav
    └── neutral/
        └── ambient_01.wav
```

The folder name directly becomes the class label — no extra annotation needed.

For a custom task named `my_task`, put files in `audio_input/my_task/<label>/`.

---

## Step 7 — Run the Pipeline

Open Dagster at **http://localhost:3000**

1. Click **Assets** in the top nav
2. You should see the `sentiment` asset group with five assets:
   - `sentiment_raw_audio_files`
   - `sentiment_segmented_features`
   - `sentiment_dataset_splits`
   - `sentiment_trained_model`
   - `sentiment_model_evaluation`
3. Click **Materialize all** to run the full pipeline

Or run from the CLI inside the container:

```bash
docker exec audio-ops-dagster \
  dagster job execute -j sentiment_full_training_job -w /app/workspace.yaml
```

Or run data prep only (no training):

```bash
docker exec audio-ops-dagster \
  dagster job execute -j sentiment_data_prep_job -w /app/workspace.yaml
```

---

## Step 8 — Monitor the Run

**Dagster logs** — follow in real time:

```bash
make logs-dagster
```

**MLflow** — open http://localhost:5000

- Experiment: `Audio_Ops_Sentiment`
- Each run is named `SENTIMENT_resnet18_spectrogram_lr0.001_bs32_<timestamp>`
- Logged params: all `TaskConfig` fields
- Logged metrics: `train_loss`, `val_loss`, `val_accuracy`, `val_f1_weighted` per epoch
- Logged artifacts: `best_model.pt` and full PyTorch model

---

## Service Credentials (Defaults)

| Service | Username | Password source |
|---------|----------|-----------------|
| MinIO Console | `minioadmin` | `MINIO_ROOT_PASSWORD` in `.secrets.env` |
| InfluxDB | `admin` | `INFLUXDB_ADMIN_PASSWORD` in `.secrets.env` |
| Grafana | `admin` | `GRAFANA_ADMIN_PASSWORD` in `.secrets.env` |
| Label Studio | — | Create account on first login |
| MLflow | — | No auth by default |
| Dagster | — | No auth by default |

---

## Adding a New Task

1. Create `audio_ops_dagster/tasks/<your_task>/`
2. Add an empty `__init__.py`
3. Add `config.yaml`:

```yaml
feature_type: spectrogram
input_dir: "audio_input/your_task"
split_strategy: "group"
num_classes: 2
class_labels:
  - positive
  - negative
model_architecture: resnet18
batch_size: 32
epochs: 15
learning_rate: 0.001
```

4. Add `task_<your_task>.py`:

```python
from typing import Dict, Tuple, Any
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import accuracy_score

from audio_ops_dagster.core.base_task import AudioTask
from audio_ops_dagster.registry import register_task

@register_task("your_task")
class YourTask(AudioTask):
    def build_model(self, input_shape: Tuple[int, ...]) -> nn.Module:
        # Return any nn.Module
        ...

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        preds = y_pred.argmax(axis=1) if y_pred.ndim == 2 else y_pred
        return {"accuracy": float(accuracy_score(y_true, preds))}

    def postprocess(self, raw_output: torch.Tensor) -> Dict[str, Any]:
        probs = torch.softmax(raw_output, dim=-1)
        idx = probs.argmax(dim=-1).item()
        return {"label": self.config.class_labels[idx]}
```

5. Restart Dagster — your task's assets and jobs appear automatically:

```bash
docker-compose restart dagster dagster-daemon
```

---

## Python Environment (Local Development)

The Dagster service uses its own `dagster_pyproject.toml` (full ML stack). For local work:

```bash
# Activate the Poetry env for the root project
eval "$(poetry env activate)"
# or
source $(poetry env info --path)/bin/activate

# Install the Dagster service deps
cd audio_ops_dagster
poetry install

# Run tests
poetry run pytest
```

---

## Common Commands

```bash
make up             # Start all services
make down           # Stop all services
make health         # Check all endpoints
make logs           # All service logs
make logs-dagster   # Dagster only
make logs-mlflow    # MLflow only
make shell-dagster  # bash into dagster container
make backup         # Snapshot data dirs
make clean          # Remove containers
make clean-all      # Remove containers + volumes (destructive)
```

---

## Troubleshooting

**Services not starting**
```bash
docker-compose ps                      # Check container status
docker-compose logs <service-name>     # Read error output
make validate                          # Check config
netstat -tlnp | grep -E '3000|5000|8000|8086|9000|9001|3001|8080'  # Port conflicts
```

**Dagster shows no assets**

Check that `workspace.yaml` points to the right file, and that `discover_tasks()` ran without errors:
```bash
make logs-dagster | grep -E "Registered|Error"
```

**Audio files not found in container**

The `audio_input/` directory is mounted from the host. Verify the volume mapping:
```bash
docker inspect audio-ops-dagster | grep -A5 Mounts
```
The path inside the container is `/app/audio_input`.

**GPU not available**

The NVIDIA device reservation in `docker-compose.yml` requires the NVIDIA Container Toolkit. On CPU-only machines, the containers will still start — PyTorch falls back to CPU automatically. To suppress Docker warnings, remove the `deploy.resources` block from the dagster service definitions.

**MLflow run IDs not linking**

MLflow is set to `http://mlflow:5000` inside the Docker network. If you're running Dagster locally (outside Docker), set `MLFLOW_TRACKING_URI=http://localhost:5000` in your shell.

**`poetry` command not found after setup**

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Add this line to your `~/.bashrc` or `~/.zshrc` to make it permanent.