# Using the framework as a package (thin project repos)

The framework is now an installable package (`audio_ops_dagster` +
`audio_ops_inference`). A **project** (e.g. stadium-sentiment, triage) is its own
thin repo that depends on the framework and brings only its own tasks + data +
project-specific code. The framework holds what's shared; the project holds
what's unique.

## The one mechanism that makes this work

Task auto-discovery scans the framework's built-in `tasks/` **and** any
directories listed in the `AUDIO_OPS_TASKS_DIR` env var. So a project keeps its
tasks in its own repo and points the env var at them:

```bash
export AUDIO_OPS_TASKS_DIR=./tasks
```

That's it. Both Dagster and the inference service honor it.

## A project repo looks like this

```
stadium-sentiment/
  pyproject.toml          # depends on audio-ops-framework
  .env                    # AUDIO_OPS_TASKS_DIR=./tasks, etc.
  tasks/
    sentiment/
      __init__.py
      config.yaml
      task_sentiment.py
  audio_input/sentiment/<label>/*.wav
```

Copy `project-template/` to start one.

## Depending on the framework

While you're co-developing the framework and projects (recommended for now), use
a **path / editable** dependency so framework edits are picked up immediately:

```toml
# project pyproject.toml
[tool.poetry.dependencies]
python = ">=3.10,<3.15"
audio-ops-framework = { path = "../audio-ops-framework", develop = true }
```

or with pip:

```bash
pip install -e ../audio-ops-framework
```

Once the framework stabilizes, pin to a tag instead:

```toml
audio-ops-framework = { git = "https://github.com/your-org/audio-ops-framework", tag = "v0.3.0" }
```

## Running each project type

**Stadium (Dagster consumer — training + batch inference):**

```bash
cd stadium-sentiment
export AUDIO_OPS_TASKS_DIR=$PWD/tasks
dagster dev -m audio_ops_dagster.definitions      # http://localhost:3000
```

The framework's `definitions.py` discovers your `tasks/sentiment`, builds its
assets, and you train/evaluate/batch-infer it. No framework source needed beyond
the installed package.

**Triage (inference consumer — serving + streaming):**

```bash
cd triage
export AUDIO_OPS_TASKS_DIR=$PWD/tasks
export ENABLE_FUSION=true            # triage uses the fusion timeline
audio-ops-serve                       # the inference API on :8000
```

`audio-ops-serve` is the console entry point installed with the package (runs
the FastAPI app via uvicorn; honors `INFERENCE_PORT`/`INFERENCE_WORKERS`).

## Where project-specific code goes (important)

The framework stays generic; project specifics live in the **project repo** and
plug in through the registries, imported from your task modules:

- A custom model/task → `@register_task` in `tasks/<name>/task_<name>.py`.
- A custom feature → `@register_feature` (import it from a task module so it
  loads during discovery).
- A custom serving wrapper → `@register_wrapper`.
- Triage's specific fusion / decision logic → a module in the **triage repo**,
  not in the framework's `extras/`. The framework's `extras/fusion.py` is a
  generic example; override or replace it in your project.

Because discovery imports your task modules, any `@register_*` calls in them (or
in modules they import) run automatically — so a project can extend the
framework without touching it.

## Docker (optional)

A thin repo can build a tiny image that installs the framework and runs its
tasks. Minimal `Dockerfile`:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir audio-ops-framework   # or: COPY+install your path
COPY tasks ./tasks
ENV AUDIO_OPS_TASKS_DIR=/app/tasks
# training:   CMD ["dagster", "dev", "-m", "audio_ops_dagster.definitions", "-h", "0.0.0.0"]
# serving:    CMD ["audio-ops-serve"]
```

For full GPU/streaming deployments, reuse the framework's
`docker-compose.inference.yml` pattern with `AUDIO_OPS_TASKS_DIR` pointed at your
mounted `tasks/`.

## Summary

| Project | Depends on | Discovers via | Runs | Deploys |
|---------|-----------|---------------|------|---------|
| stadium | framework | `AUDIO_OPS_TASKS_DIR` | `dagster dev -m audio_ops_dagster.definitions` | core compose |
| triage  | framework | `AUDIO_OPS_TASKS_DIR` | `audio-ops-serve` | inference compose |

One framework package, two thin repos, each carrying only its own tasks and
specifics — no copy-paste, no drift.
