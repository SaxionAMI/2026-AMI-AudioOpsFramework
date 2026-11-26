# The training pipeline

For a task with `train`/`evaluate`, the assets run in this order (`assets.py`).
All of them live in a Dagster asset group named after the task.

## 1. `raw_audio_files`

Scans `input_dir` recursively for `.wav .mp3 .flac .ogg .m4a`. Convention:
audio lives under `input_dir/<label>/…`, so the folder name is the label
(files directly in `input_dir` get the label `unknown`). Accepts a per-run
`input_dir`/`output_dir` override (see [Batch inference](../guide/batch-inference.md)).

## 2. `segmented_features`

`AudioProcessor` loads each file at `sample_rate`, slices it into
`segment_duration` windows with `segment_overlap`, and writes one feature array
per window:

```text
<output_dir>/features/<label>/<recording>_seg_NNN.npy
```

- The extractor is whatever `feature_type` names in the
  [feature registry](../guide/features.md).
- The last window is zero-padded; a trailing window with less than half a hop
  of real audio is dropped.
- **Caching:** files whose segments exist are skipped. A
  `_features_manifest.json` records the feature settings; change any of them
  and the next run recomputes everything (stale segments are deleted). Set
  `force_recompute: true` to always recompute.
- The step logs the actual feature shape and warns when `input_shape` differs.

## 3. `dataset_splits`

`AudioDataset.create_splits_csv` writes `train/val/test.csv` (`split_ratios`,
default 70/15/15, `seed` 42).

- `split_strategy: group` keeps every segment of a recording in the same split
  (no leakage) and stratifies over recordings so class balance is preserved.
  By default each source file is one recording, scoped by its label folder.
- If you cut one long recording into many files, set `group_pattern` to a
  regular expression that extracts the recording id from the file name. The
  first capture group is used, and matching files share a group across label
  folders. With `group_pattern: '^(.+)_\d+$'`, the files `match3_001.wav` and
  `match3_002.wav` both belong to recording `match3`, so a match never ends up
  in both train and test.
- `split_strategy: random` splits segments (stratified when possible).
- Use `group` whenever segments are correlated, which is almost always.

## 4. `trained_model`

Builds the task's model and trains it with `ModelTrainer`:

- loss from `task.compute_loss`, optimizer from `task.get_optimizer`;
- warm-up + cosine LR schedule, gradient clipping, AMP on CUDA;
- early stopping on `val_loss` or `val_acc` (`early_stop_metric`, `early_stop_patience`);
- resumable: `last_checkpoint.pt` holds optimizer/scheduler state; an
  interrupted run continues where it stopped if the setup fingerprint matches;
- writes `checkpoints/best_model.pt` (weights only), with class names,
  input shape, task type and feature config;
- logs to MLflow (optional) and InfluxDB (optional), and emits per-epoch asset
  observations so the Dagster UI shows a metric curve.

Label indices come from `class_labels` (order = index), so train, val, test and
serving share one mapping.

## 5. `model_evaluation`

Loads the checkpoint, scores `test.csv` with the task's `compute_metrics(y_true, y_pred)`
(`y_pred` are raw model outputs), writes `outputs/evaluation_<task>.json` and
attaches the metrics as asset metadata.

## 6. `inference_results` (`batch_infer`)

Runs the model over every segment and writes `outputs/inference_results_<task>.json`
and `.csv`. Each row has the task's `postprocess` fields plus `source`,
`segment_index`, `start_sec`, `end_sec`. Transcription tasks instead transcribe
each recording in `transcription_chunk_seconds` windows.

When the full training job runs, inference waits for the fresh checkpoint;
`<task>_inference_job` skips training and uses the checkpoint on disk.

## Remote storage

Every step optionally syncs its artifact to object storage **only if** the task
sets `use_remote_storage: true` *and* a storage backend is enabled. Otherwise
everything stays on local disk. Upload failures are logged, never fatal.
