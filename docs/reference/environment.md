# Environment variables

Every variable is optional. Booleans accept `true/false` (also `1/0`, `yes/no`
for most). `env.example` in the repo root mirrors this page.

## Core

| Variable | Default | Effect |
|---|---|---|
| `AUDIO_OPS_TASKS_DIR` | `./tasks` if it exists | extra task directories (os.pathsep- or comma-separated) |
| `AUDIO_OPS_DISABLE_BUILTIN_TASKS` | `false` | hide the bundled `example_classifier` |
| `AUDIO_OPS_LOG_LEVEL` | `INFO` | framework log level |
| `DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N`, `mps`; used for training, batch inference and serving |
| `INFERENCE_CHECKPOINT` | — | checkpoint for batch inference/serving of **every** task (prefer `checkpoint_path` per task) |
| `ALLOW_UNSAFE_CHECKPOINTS` | `false` | allow pickled (non-weights-only) checkpoints (trusted files only) |
| `DAGSTER_HOME` | Dagster's default | Dagster instance directory (`./dagster_home` with `make dev`) |

## Storage

| Variable | Default | Effect |
|---|---|---|
| `STORAGE_BACKEND` | `local` | `minio`/`s3` enables object storage |
| `ENABLE_MINIO` | — | legacy switch, same as `STORAGE_BACKEND=minio` for Dagster |
| `MINIO_ENDPOINT` | — | full URL (`http://minio:9000`); unset → AWS S3 |
| `MINIO_HOST`, `MINIO_PORT` | —, `9000` | alternative to `MINIO_ENDPOINT` |
| `MINIO_USE_SSL` | `false` | https for host/port endpoints |
| `MINIO_ACCESS_KEY` / `MINIO_ROOT_USER` | — | unset → boto3 default credential chain |
| `MINIO_SECRET_KEY` / `MINIO_ROOT_PASSWORD` | — | |
| `MINIO_BUCKET_NAME` | `audioops-bucket` | created on first use |
| `MINIO_REGION` | `us-east-1` | |

## Metrics (InfluxDB)

| Variable | Default | Effect |
|---|---|---|
| `ENABLE_INFLUXDB` | `false` | turn the sink on (pipeline and API) |
| `INFLUXDB_URL` | `http://influxdb:8086` | or `INFLUXDB_HOST`/`INFLUXDB_PORT`/`INFLUXDB_USE_SSL` (pipeline) |
| `INFLUXDB_TOKEN` / `INFLUXDB_ADMIN_TOKEN` | — | required when enabled |
| `INFLUXDB_ORG` | `audioops` | |
| `INFLUXDB_BUCKET` | `audio_metrics` | |

## Experiment tracking

| Variable | Default | Effect |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlruns/mlflow.db` | any MLflow backend or server |
| `ENABLE_MLFLOW` | `true` | `false` disables tracking |

## LLM and fusion

| Variable | Default | Effect |
|---|---|---|
| `ENABLE_LLM` | `false` | enable the LLM client |
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai_compatible` |
| `LLM_BASE_URL` | `http://ollama:11434` | |
| `LLM_MODEL` | `llama3.2` | |
| `LLM_API_KEY` | — | sent as a Bearer token |
| `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_TIMEOUT_SEC` | `0.3`, `512`, `30` | |
| `ENABLE_FUSION` | `false` | per-stream fusion timeline in the API |
| `FUSION_LANGUAGE` | `en` | `en` or `nl` tags and prompts |
| `FUSION_EMIT_EVERY` | `1` | emit a fusion event every N processed chunks |
| `FUSION_EMIT_WINDOW_S` | `120` | window of the live fusion event |
| `FUSION_LLM_WINDOW_S` | `60` | default window for `/fusion/{id}/llm` |
| `NONVERBAL_MIN_CONF` | `0.05` | floor for surfacing event labels into fusion |
| `NONVERBAL_BACKGROUND` | `speech,silence,unknown` | event labels never surfaced |

## Inference service

| Variable | Default | Effect |
|---|---|---|
| `INFERENCE_HOST`, `INFERENCE_PORT` | `0.0.0.0`, `8000` | bind address |
| `INFERENCE_WORKERS` | `1` | keep at 1 (stream state is in-process) |
| `INFERENCE_RELOAD` | — | `1` = auto-reload on code changes |
| `API_KEY` (alias `API_SECRET_KEY`) | — | require `X-API-Key` (or `?api_key=`) on all endpoints but `/health` |
| `CORS_ALLOW_ORIGINS` | `*` | comma-separated origins |
| `MAX_UPLOAD_MB` | `100` | `/infer/file` size limit |
| `MODELS_DIR` | `models` | extra checkpoint locations (`<dir>/<task>/best_model.pt`, `<dir>/<task>.pt`) and download target |
| `MODEL_CACHE_DIR` | `models` | HuggingFace / TF-Hub caches (`HF_HOME`, `TFHUB_CACHE_DIR`) |
| `CHUNK_MS` | `1000` | chunk size for streams |
| `PROC_QUEUE_MAX` | `5` | queued chunks before dropping the oldest (fast lane) |
| `STT_QUEUE_MAX` | `12` | same for the transcription lane |
| `AUTO_RTSP_STREAMS` | — | `url|stream_id,...` started at boot |
| `STREAM_AUTODISCOVER` | `false` | follow MediaMTX paths automatically |
| `STREAM_AUTODISCOVER_FILTER` | — | regex on path names |
| `STREAM_AUTODISCOVER_INTERVAL` | `5` | seconds between polls |
| `MEDIAMTX_API_URL` | `http://mediamtx:9997` | |
| `RTSP_BASE_URL` | `rtsp://mediamtx:8554` | |
| `DEMO_AUDIO_PATH` | `/test_audio/demo.wav` | file served by `GET /demo` |
| `STT_USE_PROMPT` | `0` | feed previous text back as Whisper `initial_prompt` |
| `STT_HALLUCINATION_PHRASES` | — | extra phrases to drop (comma-separated) |
| `YAMNET_BUFFER_S` | — | override the event-tagger buffer length |
