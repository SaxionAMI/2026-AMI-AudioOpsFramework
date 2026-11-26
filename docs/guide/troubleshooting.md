# Troubleshooting

Start with `audio-ops doctor` (environment) and `audio-ops validate` (configs).

## Setup

??? question "`No Python 3.10 to 3.14 found`"
    Install one, for example `sudo apt install python3.12 python3.12-venv`
    (Ubuntu) or `brew install python@3.12` (macOS), then run the setup script
    again. Or point to it with `--python /path/to/python3.12`.

??? question "`Could not create the virtual environment`"
    On Debian and Ubuntu the venv module is a separate package:
    `sudo apt install python3-venv` (or `python3.12-venv` for that version).

??? question "`Could not get the framework` in a project repo"
    The framework repository is private, or you are offline. Either add an SSH
    key to your GitHub account and ask for access, then run
    `./scripts/setup.sh --framework ssh`, or clone the framework next to the
    project (`git clone ... ../2026-AMI-AudioOpsFramework`) and run the script
    again. If your SSH key has a passphrase, load it first:
    `eval "$(ssh-agent -s)" && ssh-add`.

??? question "The download is very large"
    On Linux, the standard PyTorch build includes CUDA (several GB). The setup
    script uses the CPU-only build when it finds no NVIDIA GPU. To force it:
    `./scripts/setup.sh --torch cpu`.

??? question "`audio-ops: command not found`"
    The virtual environment is not active. Use the `make` targets, which find
    `.venv` themselves, or run `source .venv/bin/activate` first.

## Running

??? question "My task doesn't show up in Dagster / the API"
    - `audio-ops list`: is it discovered? Task folders are found in `./tasks`
      (relative to where you run the command) or in `AUDIO_OPS_TASKS_DIR`, and
      must contain a `config.yaml`.
    - `audio-ops validate <task>`: config errors skip the task with a message
      in the Dagster code-location logs.
    - "no class registered": the module must be `task_<folder>.py` (or
      `task.py`) and decorate the class with `@register_task("<folder>")`.
      Config-only folders work for `classification`, `regression` and
      `detection`; other types need Python.
    - Import errors in the task module are logged as `Error loading task module …`.

??? question "The API says the task is `not_loaded`"
    `GET /models` lists the reason. Usually there's no checkpoint yet: train the
    task, then `POST /models/reload`. A *does not match the model* error means
    the checkpoint was trained with a different architecture or `num_classes`;
    retrain or point `checkpoint_path` at the right file.

??? question "Accuracy in serving is much worse than in evaluation"
    Serving uses the task's `feature_type`, `sample_rate` and
    `segment_duration`. If you changed any of these after training, retrain
    (the API warns when the checkpoint's recorded feature type differs). Make
    sure the stream really is the audio you think (check `rms` in events).

??? question "Features are recomputed every run"
    The feature manifest changed (a feature setting differs) or
    `force_recompute: true` is set. A directory from an older version without a
    manifest is recomputed once.

??? question "`MlflowException: The filesystem tracking backend ... maintenance mode`"
    MLflow 3 refuses `./mlruns` file stores. Use the default SQLite URI
    (`sqlite:///mlruns/mlflow.db`), a server, or set `MLFLOW_ALLOW_FILE_STORE=true`.

??? question "`Need at least 3 recordings to make train/val/test splits`"
    Group splits work on recordings, not segments. Add recordings, or use
    `split_strategy: random` for a quick experiment.

??? question "`labels [...] are not in class_labels`"
    A label folder exists in `input_dir` that isn't listed in `class_labels`.
    Add it, or remove/move the folder.

??? question "`class_labels contains booleans`"
    YAML reads `yes`, `no`, `on`, `off` as booleans. Quote them: `- "yes"`.

??? question "Streams connect but no events arrive"
    - `GET /streams`: are `frames` increasing? If not, ffmpeg can't read the URL
      (is `ffmpeg` installed in the container? is the RTSP path publishing?).
    - Are any models loaded (`GET /health`)?
    - With `API_KEY` set, SSE/WebSocket clients must pass `?api_key=`.

??? question "CUDA is installed but training runs on CPU"
    `audio-ops doctor` shows the accelerator torch sees. Install a CUDA build of
    torch (see [Installation](../getting-started/installation.md)) and set
    `DEVICE=cuda` (or leave `auto`).
