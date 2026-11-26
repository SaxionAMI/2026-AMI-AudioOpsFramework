# Roadmap

Known limitations and the next steps worth taking. None of them block normal
use. Contributions are welcome; open an issue to discuss before starting.

## Planned

- **Portable model artifacts (ONNX / TorchScript).** Serving still imports the
  task module to call `build_model`. Exporting a self-contained artifact plus a
  sidecar config as the last training step would let a serving image run
  without the task code at all.
- **Labels from a manifest.** Targets currently come from label folders. A
  `labels.csv` (path → target(s)) would enable true multi-label detection,
  regression on arbitrary values and per-file metadata.
- **Evaluate on all data.** `eval_only` scores the test split; an
  `eval_split: all` option would benchmark an external checkpoint on the whole set.
- **Per-task inference checkpoint override from the Launchpad** (today:
  `checkpoint_path` in the config or the process-wide `INFERENCE_CHECKPOINT`).
- **Dagster partitions** for large or continuously growing audio archives.

## Ideas

- Data augmentation hooks on the training dataset (SpecAugment, mixing).
- A Prometheus metrics sink next to the InfluxDB one.
- A minimal browser demo page for WebSocket ingest + SSE visualisation.
- Multi-worker serving with an external stream registry.

## Done

- **1.2**: simplification: one `TaskConfig`, ready-made task bases and
  config-only tasks, `from audio_ops import …`, `./tasks` auto-discovery,
  `audio-ops run` / `predict`, ~2,000 lines of duplicate code removed.
- **1.1**: task-driven training, train/serve feature parity, per-stream
  isolation, real `eval_only`, API auth, the CLI, demo data, a real test suite.

See the [changelog](changelog.md).
