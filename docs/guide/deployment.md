# Deployment

Training and streaming inference have different operational profiles, so they
are separate **deployables** built from one repo.

```bash
docker compose up -d                                        # core (Dagster) only
docker compose -f docker-compose.inference.yml up -d        # inference only (own host)
docker compose -f docker-compose.yml \
               -f docker-compose.inference.yml up -d        # both
```

`make up`, `make up-inference`, `make up-all`, `make down`, `make logs` wrap these.

## Images

| Image | Dockerfile | Default command |
|---|---|---|
| core | `docker/Dockerfile.dagster` | `dagster-webserver -w workspace.yaml` (compose also runs `dagster-daemon`) |
| inference | `docker/Dockerfile.inference` | `audio-ops serve` on :8000, non-root user, healthcheck on `/health` |

Both install the framework package with build args:

| Build arg | Default | Purpose |
|---|---|---|
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cpu` | CPU wheels (small); use a `cu12x` index for CUDA |
| `EXTRAS` | `train,s3,metrics` / `serve,s3,metrics` | package extras |
| `EXTRA_PIP` | *(empty)* | your tasks' runtimes, e.g. `faster-whisper` |
| `PYTHON_VERSION` | `3.11` | base image |

```bash
docker build -f docker/Dockerfile.inference \
  --build-arg EXTRA_PIP="faster-whisper" -t my-inference .
```

## GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
docker compose -f docker-compose.inference.yml -f docker-compose.inference.gpu.yml up -d --build
```

The overlays rebuild with CUDA wheels (`TORCH_CUDA_INDEX_URL`, default `cu124`),
set `DEVICE=cuda` and reserve NVIDIA GPUs. They need the NVIDIA Container Toolkit.

## Volumes

The compose files are set up for development: the repo is bind-mounted so code
changes apply without rebuilding.

| Path | Core | Inference |
|---|---|---|
| `./storage` | read/write (artifacts) | read-only (checkpoints) |
| `./audio_input` | training data | — |
| `./mlruns`, `./dagster_home` | tracking DB, Dagster instance | — |
| model cache | — | named volume `model-cache` (HF/TF-Hub downloads) |
| `./mediamtx/test_audio` | — | demo audio for `GET /demo` |

For an immutable production deploy, remove the source bind mounts and bake
your tasks into the image (`COPY tasks /app/tasks` + `AUDIO_OPS_TASKS_DIR`).

## Scaling the inference service

The API keeps stream state in-process, so run **one worker per container** and
scale containers behind a load balancer with sticky routing per stream (the
WebSocket and SSE connections for a stream must hit the same instance). Models
are loaded once per container; after training, `POST /models/reload` picks up
new checkpoints without a restart.

## Security checklist

- Set `API_KEY` and terminate TLS in front of the service.
- Restrict `CORS_ALLOW_ORIGINS`.
- Don't expose `POST /streams` to untrusted callers (the server fetches the URL with ffmpeg).
- Keep `ALLOW_UNSAFE_CHECKPOINTS` off.
- Put secrets in `.env` / your secret store; nothing is baked into images.

See also [SECURITY.md](https://github.com/SaxionAMI/2026-AMI-AudioOpsFramework/blob/main/SECURITY.md).
