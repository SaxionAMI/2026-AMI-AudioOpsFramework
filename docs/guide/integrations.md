# Optional integrations

Everything here is **off by default**, switched on with environment variables
(see [`env.example`](https://github.com/SaxionAMI/2026-AMI-AudioOpsFramework/blob/main/env.example)),
and degrades to a no-op when unavailable. No code changes are needed.

| System | Enable with | Extra |
|---|---|---|
| Object storage (MinIO/S3/R2/B2) | `STORAGE_BACKEND=minio` + `MINIO_*`; per task `use_remote_storage: true` | `s3` |
| Metrics (InfluxDB v2) | `ENABLE_INFLUXDB=true` + `INFLUXDB_*` | `metrics` |
| Experiment tracking server | `MLFLOW_TRACKING_URI=http://mlflow:5000` (local SQLite otherwise; `ENABLE_MLFLOW=false` to disable) | `train` |
| LLM | `ENABLE_LLM=true` + `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL` | — |
| Fusion timeline | `ENABLE_FUSION=true` (+ `FUSION_LANGUAGE`) | — |
| Grafana / Label Studio | not bundled; add your own services pointing at the data sources above | — |

## Object storage

```bash
STORAGE_BACKEND=minio
MINIO_ENDPOINT=http://minio:9000     # omit for AWS S3
MINIO_ACCESS_KEY=...                 # omit to use boto3's credential chain
MINIO_SECRET_KEY=...
MINIO_BUCKET_NAME=audioops-bucket
```

Uploads happen per artifact (features, splits, checkpoint, results) for tasks
with `use_remote_storage: true`; failures are logged, never fatal. The bucket is
created on first use. The inference service downloads checkpoints from
`{prefix}/models/best_model.pt` when none exist locally.

A local MinIO for development:

```yaml
# docker-compose.override.yml
services:
  minio:
    image: minio/minio
    command: server /data --console-address :9001
    environment: { MINIO_ROOT_USER: dev, MINIO_ROOT_PASSWORD: devpassword }
    ports: ["9000:9000", "9001:9001"]
    networks: [audio-ops]
```

## InfluxDB metrics

```bash
ENABLE_INFLUXDB=true
INFLUXDB_URL=http://influxdb:8086
INFLUXDB_TOKEN=...
INFLUXDB_ORG=audioops
INFLUXDB_BUCKET=audio_metrics
```

Measurements written by the pipeline:

| Measurement | Content |
|---|---|
| `audio_pipeline` | lifecycle events (`feature_extraction_start`, `training_complete`, …) |
| `audio_training` | per-epoch / per-batch losses, accuracy, learning rate |
| `audio_inference` | per-segment batch-inference results (`format_influx_record`) |

and by the inference service: `inference_<task>` (classifier/event/STT results),
`stt_word`, `inference_fusion`, `stream_health`.

To send metrics elsewhere (Prometheus, a file, stdout), subclass
`audio_ops_dagster.core.metrics.NullMetricsResource`, override the writers you
need and pass it as the `influxdb_metrics` resource.

## MLflow

Training logs params, per-epoch metrics and the best checkpoint as an artifact.
Default backend: `sqlite:///mlruns/mlflow.db`.

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

A tracking outage never fails training. The tracker logs one warning and
disables itself for the run.

## LLM

Any Ollama or OpenAI-compatible endpoint (vLLM, LiteLLM, LocalAI, hosted APIs):

```bash
ENABLE_LLM=true
LLM_PROVIDER=openai_compatible     # or ollama (default)
LLM_BASE_URL=http://vllm:8000
LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
LLM_API_KEY=...                    # if required
```

Used by `POST /fusion/{stream}/llm` and available to tasks through
`postprocess_with_llm(..., llm_client=...)` or
`audio_ops_dagster.core.llm_client.LLMClient.from_env()`. Calls never raise;
a failure returns an empty string.
