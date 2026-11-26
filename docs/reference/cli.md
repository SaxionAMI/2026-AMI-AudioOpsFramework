# CLI

`audio-ops` is installed with the package. `audio-ops <command> --help` shows all options.

| Command | What it does |
|---|---|
| `audio-ops new-task NAME` | scaffold a task (`--type classification\|regression\|detection\|transcription`, `--config-only`, `--archetype`, `--feature`, `--classes a,b,c`, `--dir`) |
| `audio-ops list` | discovered tasks with type, capabilities, checkpoint status and config status |
| `audio-ops validate [NAME ...]` | validate configs; exit code 1 on errors (use it in CI) |
| `audio-ops run TASK` | run the pipeline headlessly: `--job train\|evaluate\|infer\|prep` (default: whatever the task has), `--input-dir`, `--output-dir`, `--log-level` |
| `audio-ops predict TASK FILE...` | score audio files without a server (`--segments`, `--json`, `--checkpoint`) |
| `audio-ops demo-data` | synthetic labelled audio (`--out`, `--files`, `--stream-demo WAV`) |
| `audio-ops dev [ARGS]` | `dagster dev -m audio_ops_dagster.definitions` with `DAGSTER_HOME=./dagster_home` unless set (extra args pass through) |
| `audio-ops serve` | run the inference API (`--host`, `--port`, `--reload`) |
| `audio-ops doctor` | installed extras, torch accelerator, ffmpeg, relevant env vars |
| `audio-ops version` | print the version |

Compatibility entry points: `audio-ops-serve` (= `audio-ops serve`) and
`python scripts/new_task.py` (= `audio-ops new-task`).

## Examples

```bash
audio-ops new-task cough --type detection --classes cough,speech,noise
audio-ops validate || exit 1
audio-ops run cough
audio-ops run cough --job infer --input-dir /data/new --output-dir /data/new_out
audio-ops predict cough clip.wav --json
audio-ops dev --port 3001
audio-ops serve --reload
```

## Make targets (repo checkout)

| Target | |
|---|---|
| `make install` | editable install with dev + docs extras, pre-commit hooks |
| `make setup` / `make demo` | runtime dirs + `.env` / synthetic data + streaming demo file |
| `make dev` / `make serve` | Dagster / API locally |
| `make test` / `make test-fast` | all tests / unit tests only |
| `make lint` / `make format` | ruff |
| `make docs` / `make docs-serve` | build / preview this site |
| `make build` | sdist + wheel, `twine check` |
| `make up` / `up-inference` / `up-all` / `down` / `logs` / `ps` / `images` | Docker |
| `make new-task NAME=x [TYPE=y]` | scaffold |
| `make clean` | remove local runtime artifacts |
