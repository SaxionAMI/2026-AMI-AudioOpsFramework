# Installation

## Requirements

- Linux or macOS. On Windows, use [WSL2](https://learn.microsoft.com/windows/wsl/install)
  with Ubuntu and run everything inside it.
- Python 3.10 to 3.14 (3.12 is a good choice) and `git`.
- ffmpeg, only for live RTSP streams (`sudo apt install ffmpeg` or `brew install ffmpeg`).
- Disk space: about 1.5 GB with the CPU build of PyTorch, 6 GB or more with the
  CUDA build.

## Quick setup (recommended)

```bash
git clone https://github.com/SaxionAMI/2026-AMI-AudioOpsFramework
cd 2026-AMI-AudioOpsFramework
./scripts/setup.sh          # or: make setup
```

The script:

1. finds a suitable Python and creates a virtual environment in `.venv`;
2. installs PyTorch: the CUDA build if `nvidia-smi` finds an NVIDIA GPU, the
   standard build on macOS (with Apple silicon support), otherwise the much
   smaller CPU-only build;
3. installs the framework from the checkout (editable, so your changes apply
   immediately) with the `train` and `serve` extras;
4. creates the runtime folders and a `.env` from `env.example`;
5. generates demo audio and runs `audio-ops doctor`.

It is safe to run again; it keeps what is already there. Options:

| Option | Effect |
|---|---|
| `--dev` | also install test, lint and docs tools (for contributors) |
| `--extras train,serve,s3` | choose the [extras](#extras) |
| `--torch cpu` / `cuda` / `skip` | force a PyTorch build, or keep the installed one |
| `--python /usr/bin/python3.12` | use a specific Python |
| `--no-demo` | skip the demo audio |

Afterwards the `make` targets (`make run`, `make dev`, `make serve`, `make test`)
use `.venv` automatically. To use `audio-ops` directly, activate it first:
`source .venv/bin/activate`.

For your own project, don't work inside the framework checkout. Copy
`project-template/` instead; it has its own `scripts/setup.sh`. See
[Project repos](../guide/project-repos.md).

## Extras

The framework is one Python package (`audio-ops-framework`) with extras, so
each deployment installs only what it needs.

| Extra | Installs | Use it for |
|---|---|---|
| *(none)* | numpy, PyYAML, pydantic, librosa, soundfile | config validation, feature registry, tooling |
| `train` | torch, Dagster (+ webserver), MLflow, scikit-learn, pandas | the offline pipeline |
| `serve` | torch, FastAPI, Uvicorn, python-multipart | the inference service, without Dagster/MLflow |
| `ast` | transformers, torchaudio | `feature_type: ast` (`ast_librosa` needs nothing extra) |
| `s3` | boto3 | MinIO / S3 / R2 / B2 artifact storage |
| `metrics` | influxdb-client | InfluxDB metrics sink |
| `all` | everything above | |
| `dev`, `docs` | test, lint and docs tools | contributors |

## Manual installation

### PyTorch first

PyTorch wheels differ per accelerator, so install the build you want **before**
the framework (the framework requires `torch>=2.4` with no upper bound and
won't replace a build you already have):

=== "CPU"

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```

=== "CUDA"

    ```bash
    # pick the index matching your driver, e.g. cu124
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    ```

=== "Apple silicon"

    ```bash
    pip install torch          # includes MPS; set DEVICE=mps (or leave auto)
    ```

### The framework

=== "From GitHub"

    ```bash
    pip install "audio-ops-framework[train,serve] @ git+https://github.com/SaxionAMI/2026-AMI-AudioOpsFramework.git@v1.2.0"
    ```

    While the repository is private, use
    `git+ssh://git@github.com/SaxionAMI/2026-AMI-AudioOpsFramework.git@v1.2.0`
    (this needs an [SSH key on your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)).

=== "From PyPI"

    Once a release is published on PyPI:

    ```bash
    pip install "audio-ops-framework[train,serve]"
    ```

=== "From a checkout"

    ```bash
    pip install -e ".[dev]"
    ```

### System packages

- **libsndfile**: audio IO (bundled with the `soundfile` wheel on most platforms).
- **ffmpeg**: needed by the inference service to pull RTSP/HTTP streams and by
  librosa for some formats (`apt install ffmpeg` / `brew install ffmpeg`).

### Check your environment

```bash
audio-ops doctor
```

prints which extras are importable, the torch accelerator and the relevant
environment switches.
