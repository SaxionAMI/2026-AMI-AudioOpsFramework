# Quickstart

Train, evaluate, batch-score and serve the bundled `example_classifier`, a
three-class tone classifier, without any dataset or external service.

## 1. Install

```bash
git clone https://github.com/SaxionAMI/2026-AMI-AudioOpsFramework
cd 2026-AMI-AudioOpsFramework
./scripts/setup.sh
source .venv/bin/activate
```

See [Installation](installation.md) for what the script does and its options.

## 2. Demo data

The setup script already wrote synthetic clips to
`audio_input/example_classifier/<class_a|class_b|class_c>/`. The folder name is
the label. To write them again, or in another folder: `audio-ops demo-data`.

## 3. Train

```bash
audio-ops run example_classifier
```

The pipeline:

1. scans `audio_input/example_classifier/`,
2. cuts each file into 1 s segments and writes one log-mel `.npy` per segment,
3. writes leak-free `train/val/test.csv` splits,
4. trains the CNN and writes `checkpoints/best_model.pt`,
5. evaluates on the test split → `outputs/evaluation_example_classifier.json`,
6. scores every segment → `outputs/inference_results_example_classifier.{json,csv}`.

Everything lands under `storage/example_classifier/`; MLflow logs to
`mlruns/mlflow.db` (`mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`).

!!! tip "Dagster UI"
    `audio-ops dev` starts Dagster on <http://localhost:3000> with the same
    pipeline as **Jobs → `example_classifier_full_training_job`**, plus run
    history, asset lineage and a live training-metric curve.

## 4. Predict

```bash
audio-ops predict example_classifier audio_input/example_classifier/class_b/class_b_00.wav --segments
```

## 5. Serve

```bash
audio-ops serve        # http://localhost:8000/docs
```

The API discovers every task that declares `serve` and has a checkpoint:

```bash
curl -s http://localhost:8000/models | jq
curl -s -F file=@audio_input/example_classifier/class_b/class_b_00.wav \
     http://localhost:8000/infer/file | jq '.tasks.example_classifier | {label, confidence}'
```

File uploads are segmented exactly like training; you get per-segment results
plus an aggregate. For live audio see [Streaming](../guide/streaming.md).

## 6. Or: everything in Docker

```bash
make setup            # if you haven't yet: .venv, .env, demo data, mediamtx/test_audio/demo.wav
make up-all           # Dagster :3000, inference :8000, MediaMTX :8554
```

This needs Docker with Compose 2.24 or newer (`docker compose version`).

Train from the Dagster UI, then `curl -X POST localhost:8000/models/reload` and
start a stream from the demo RTSP path:

```bash
curl -X POST localhost:8000/streams -H 'content-type: application/json' \
     -d '{"url": "rtsp://mediamtx:8554/demo", "stream_id": "demo"}'
curl -N localhost:8000/streams/demo/events
```

## Next

- [Write your first task](first-task.md)
- [How capabilities shape the pipeline](../concepts/capabilities.md)
