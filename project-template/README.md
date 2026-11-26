# My Audio Project

A project built on the
[Audio Ops Framework](https://github.com/SaxionAMI/2026-AMI-AudioOpsFramework).
This repo contains only the project's own tasks and data; the framework is
installed as a package.

## Requirements

- Linux or macOS. On Windows, use WSL2 with Ubuntu.
- Python 3.10 to 3.14 and git.
- ffmpeg, only for live RTSP streams.

## Setup

```bash
make setup                  # or: ./scripts/setup.sh
```

This creates `.venv`, installs PyTorch for your hardware, the framework and the
other dependencies from `pyproject.toml`, creates `.env` and checks the tasks.
It is safe to run again. If the framework repository can't be reached, the
script tells you what to do; `./scripts/setup.sh --help` lists the options.

## Add a task

```bash
source .venv/bin/activate
audio-ops new-task my_task --config-only --classes yes_class,no_class
audio-ops new-task loudness --type regression
```

Tasks are written to `./tasks` and found automatically. Put labelled audio in
`audio_input/<task>/<label>/*.wav`, then run `make setup` again to see how many
files it finds per label.

## Run

```bash
make train                  # train, evaluate and score (TASK=name to pick a task)
make predict FILE=a.wav     # score files
make serve                  # inference API on http://localhost:8000/docs
make dev                    # the Dagster UI on http://localhost:3000
make test                   # validate the tasks (and run tests/ if present)
```

## Project-specific code

Custom tasks, features and serve wrappers live here and register through the
framework's decorators (`@register_task`, `@register_feature`,
`@register_wrapper`). Task folders are imported as packages, so a task can
split its code into modules and use relative imports.
