# Quickstart

```bash
make setup            # runtime dirs + .env from env.example
```

Put a little labelled audio under:

```
audio_input/example_classifier/class_a/*.wav
audio_input/example_classifier/class_b/*.wav
audio_input/example_classifier/class_c/*.wav
```

## Run locally (no Docker)

```bash
poetry install        # installs deps
dagster dev           # open http://localhost:3000
```

Materialize the `example_classifier` assets in the UI, or run the
`example_classifier_full_training_job`. Output lands in
`storage/example_classifier/`.

## Run with Docker

```bash
docker compose up -d                                   # core: Dagster :3000
# add streaming inference (:8000) + media server:
docker compose -f docker-compose.yml -f docker-compose.inference.yml up -d
docker compose logs -f
```

## Make your own task

```bash
python scripts/new_task.py my_task --archetype trainable_model
```

Implement `build_model / compute_loss / compute_metrics / postprocess` in
`audio_ops_dagster/tasks/my_task/task_my_task.py`, add audio under
`audio_input/my_task/<label>/`, and re-run. See README.md and MIGRATION.md
for capabilities and optional integrations.
