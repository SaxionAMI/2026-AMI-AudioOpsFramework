# Project repos

The framework is an installable package. A **project** is its own thin repo
that depends on it and brings only its tasks, data layout and project-specific
code. The framework holds the shared parts and the project holds its own tasks.

## The one mechanism

Discovery scans the framework's built-in tasks **and** your task folders:

- `./tasks` in the working directory, found automatically;
- or every directory in `AUDIO_OPS_TASKS_DIR` (os.pathsep- or comma-separated),
  for any other location.

```bash
export AUDIO_OPS_DISABLE_BUILTIN_TASKS=true    # optional: hide example_classifier
```

Dagster, the inference service and the CLI all use the same discovery.

## Layout

Copy `project-template/` from the framework repo:

```text
my-project/
├── pyproject.toml        # depends on audio-ops-framework[train,serve]
├── Makefile              # make setup, train, predict, serve, dev, test
├── scripts/setup.sh      # one-time setup (see below)
├── .env                  # local settings, created from .env.example
├── tasks/
│   └── sentiment/
│       ├── __init__.py
│       ├── config.yaml
│       └── task_sentiment.py
└── audio_input/sentiment/<label>/*.wav
```

## Setting up a project

```bash
make setup          # runs ./scripts/setup.sh
```

The script creates `.venv`, installs PyTorch (CUDA, Apple silicon or CPU-only,
depending on the machine) and every dependency in `pyproject.toml`, creates
`.env` and the runtime folders, validates the tasks and reports how many
training files it finds per label. It is safe to run again.

The framework entry in `pyproject.toml` decides where the framework comes from.
For a git URL, the script tries these in order and uses the first that works:

1. the URL over HTTPS (works once the repository is public);
2. the same repository over SSH (works while it is private, if your SSH key is
   on GitHub and you have access);
3. a framework checkout next to the project, `../2026-AMI-AudioOpsFramework`,
   installed editable.

Force one with `--framework https`, `--framework ssh` or
`--framework /path/to/checkout`. `./scripts/setup.sh --help` lists all options.

## Depending on the framework

=== "Git tag (default)"

    ```toml
    dependencies = ["audio-ops-framework[train,serve] @ git+https://github.com/SaxionAMI/2026-AMI-AudioOpsFramework.git@v1.2.0"]
    ```

=== "PyPI"

    Once releases are published on PyPI:

    ```toml
    dependencies = ["audio-ops-framework[train,serve]>=1.2,<2"]
    ```

=== "Working on both"

    Clone the framework next to the project, then:

    ```bash
    ./scripts/setup.sh --framework ../2026-AMI-AudioOpsFramework
    ```

    Changes in the framework checkout apply immediately.

The GRASS project
([SaxionAMI/2025-RAAKMKB-GRASS](https://github.com/SaxionAMI/2025-RAAKMKB-GRASS))
is a complete example of a project repo, including a Docker stack.

## Running each kind of project

```bash
# training / batch-inference project
audio-ops dev                       # = dagster dev -m audio_ops_dagster.definitions

# serving project
ENABLE_FUSION=true audio-ops serve  # e.g. a call-analysis project using fusion
```

| Project kind | Installs | Runs | Deploys |
|---|---|---|---|
| training | `[train]` | `audio-ops dev` / Dagster | core compose |
| serving | `[serve]` | `audio-ops serve` | inference compose |
| both | `[train,serve]` | both | both |

## Where project-specific code goes

- A model/task → `@register_task` in `tasks/<name>/task_<name>.py`.
- A feature → `@register_feature` in (or imported by) a task module.
- A serving runtime → `@register_wrapper` in (or imported by) a task module.
- Decision logic, custom fusion, dashboards → modules in the project repo.

Because discovery imports your task modules, every `@register_*` call in them
(or in modules they import) runs automatically.

## Docker for a project

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "audio-ops-framework[serve]" faster-whisper
WORKDIR /app
COPY tasks ./tasks
ENV AUDIO_OPS_DISABLE_BUILTIN_TASKS=true      # /app/tasks is found as ./tasks
# training: CMD ["dagster", "dev", "-m", "audio_ops_dagster.definitions", "-h", "0.0.0.0"]
CMD ["audio-ops", "serve"]
```
