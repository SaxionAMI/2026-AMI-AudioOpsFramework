# Batch inference

"Score this folder with a trained task" is the task's `<task>_inference_job`:
`raw_audio_files → segmented_features → inference_results`. No training, no
daemon, no scripts.

## From the command line

```bash
audio-ops run example_classifier --job infer \
    --input-dir /data/new_batch --output-dir /data/new_batch_out
```

## From the Dagster UI

Open the job's **Launchpad** and set the run config once on the root asset:

```yaml
ops:
  example_classifier_raw_audio_files:
    config:
      input_dir: /data/new_batch          # audio to score
      output_dir: /data/new_batch_out      # features + results land here
```

The override threads through the whole path. Leave it unset to use the task's
configured dirs.

## From Python

```python
from audio_ops_dagster.definitions import build_definitions

defs = build_definitions()
defs.resolve_job_def("example_classifier_inference_job").execute_in_process(run_config={
    "ops": {"example_classifier_raw_audio_files": {"config": {
        "input_dir": "/data/new_batch", "output_dir": "/data/new_batch_out"}}}
})
```

## Which checkpoint?

1. `checkpoint_path` in the task's `config.yaml`;
2. the `INFERENCE_CHECKPOINT` env var (applies to every task in the process);
3. `<task output_dir>/checkpoints/best_model.pt`.

Without any checkpoint the job warns and scores with an untrained model.

## Output

`<output_dir>/outputs/inference_results_<task>.json`, keyed by segment path:

```json
{
  "class_b/meeting_03_seg_004.npy": {
    "label": "class_b",
    "confidence": 0.97,
    "scores": {"class_a": 0.01, "class_b": 0.97, "class_c": 0.02},
    "source": "class_b/meeting_03",
    "segment_index": 4,
    "start_sec": 3.0,
    "end_sec": 4.0
  }
}
```

and a flat CSV with one row per segment (scalar fields only). With InfluxDB
enabled, every segment is also written as a point using the task's
`format_influx_record`.

## Transcription tasks

For `TASK_TYPE = "transcription"` the job transcribes each source recording in
`transcription_chunk_seconds` windows (default 30 s) through the task's
`predict`, keyed `<label>/<file>_chunk_NNN`, with `start_sec`/`end_sec`.

## `pretrained_batch`

A task with `archetype: pretrained_batch` gets only this job. Point
`checkpoint_path` at weights trained elsewhere (or make `build_model` load a
pretrained network) and score datasets without any training stage.
