# My Audio Project (template)

A thin repo built on the **Audio Ops Framework**. It contains only this
project's tasks, data, and any project-specific code — the framework is a
dependency.

## Setup

```bash
cp .env.example .env
poetry install          # installs the framework (path/git dep in pyproject)
mkdir -p audio_input
```

Add a task:

```bash
# from the framework repo's scaffolder, or copy an example into tasks/
# tasks/<name>/{__init__.py, config.yaml, task_<name>.py}
```

Put labelled audio under `audio_input/<task>/<label>/*.wav`.

## Run

Training / batch inference (Dagster):

```bash
export AUDIO_OPS_TASKS_DIR=$PWD/tasks
dagster dev -m audio_ops_dagster.definitions      # http://localhost:3000
```

Serving (inference API), if this project serves:

```bash
export AUDIO_OPS_TASKS_DIR=$PWD/tasks
audio-ops-serve                                    # http://localhost:8000
```

## Project-specific code

Custom tasks, features, or serve wrappers live here and register via the
framework's decorators (`@register_task`, `@register_feature`,
`@register_wrapper`) — imported from your task modules, they load automatically.
See the framework's `PACKAGING.md`.
