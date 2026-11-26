# Task configuration (`config.yaml`)

Every key the framework reads, with its default, all defined once in
`audio_ops.TaskConfig`. Unknown keys are allowed (and available to your task as
`self.config.<key>`); `audio-ops validate` suggests the intended key for likely
typos. A minimal config can be just `class_labels: [a, b]`.

## What the task does

| Key | Default | Description |
|---|---|---|
| `archetype` | — | `trainable_model`, `pretrained_servable`, `pretrained_batch`, `eval_only` |
| `capabilities` | — | explicit list of `train`, `evaluate`, `batch_infer`, `serve` (wins over `archetype`) |
| `task_name` | folder name | optional; must equal the folder name |
| `task_type` | `classification` | `classification`, `regression`, `detection`, `transcription`, `nonverbal_pretrained`; the class's `TASK_TYPE` wins |
| `serve_wrapper` | inferred | force a serve wrapper type (`classifier`, `transcription`, `pretrained_event`, or your own) |

With neither `capabilities` nor `archetype`, the legacy
`dagster: {training_enabled: false}` means serve-only; otherwise `trainable_model`.

## Data and features

| Key | Default | Description |
|---|---|---|
| `input_dir` | `audio_input/<task>` | audio under `<input_dir>/<label>/*` |
| `output_dir` | `storage/<task>` | features, splits, checkpoints, outputs |
| `feature_type` | `spectrogram` | a registered [feature extractor](../guide/features.md) |
| `sample_rate` | `16000` | resample rate for training and serving |
| `segment_duration` | `1.0` | seconds per segment, also the serving window |
| `segment_overlap` | `0.25` | fraction in `[0, 1)` |
| `force_recompute` | `false` | ignore the feature cache |
| `input_shape` | — | model input `(C, F, T)`; passed to `build_model` |
| `class_labels` | sorted folder names | label folder names; order defines class indices (stored in the checkpoint) |
| `num_classes` | `len(class_labels)` | only needed without `class_labels`; must match if both are set |
| `detection_threshold` | `0.5` | `DetectionTask` decision threshold |

## Splits

| Key | Default | Description |
|---|---|---|
| `split_strategy` | `random` | `group` (no recording leaks across splits) or `random` |
| `group_pattern` | none | regex on the file name; its first group is the recording id (see below) |
| `split_ratios` | `[0.7, 0.15, 0.15]` | train/val/test, must sum to 1 |
| `seed` | `42` | splits and training |

## Training

| Key | Default | Description |
|---|---|---|
| `batch_size` | `32` | |
| `epochs` | `10` | |
| `learning_rate` | `0.001` | |
| `optimizer` | `adamw` | `adamw`, `adam`, `sgd` (momentum 0.9), via `get_optimizer` |
| `weight_decay` | `0.0` | |
| `criterion` | by `task_type` | `cross_entropy` (classification), `bce` (detection), `mse` (regression); also `l1`/`mae`, `bce_with_logits`, `ctc` |
| `class_weights` | — | per-class weights (`cross_entropy`) / `pos_weight` (`bce`) |
| `label_smoothing` | `0.0` | `cross_entropy` only |
| `warmup_epochs` | `0` | linear warm-up before cosine decay |
| `grad_clip` | `1.0` | max grad norm; `null` disables |
| `early_stop_metric` | `val_loss` | `val_loss` or `val_acc` (classification) |
| `early_stop_patience` | `10` | epochs without improvement |
| `mixed_precision` | `true` | AMP on CUDA |
| `num_workers` | `0` | DataLoader workers |
| `lr_min` | `1e-6` | floor of the cosine schedule |
| `grad_accumulation_steps` | `1` | accumulate gradients over N batches |
| `resume` | `true` | continue an interrupted run from `last_checkpoint.pt` |
| `log_every_n_batches` | `50` | progress line interval |
| `model_architecture` | `cnn` | a label for logs/MLflow/checkpoints |

## Inference

| Key | Default | Description |
|---|---|---|
| `checkpoint_path` | — | checkpoint for evaluation (eval-only), batch inference and serving |
| `transcription_chunk_seconds` | `30.0` | window for batch transcription |
| `stream_buffer_seconds` | 5 (STT) / 2 (events) | buffer per call in streaming |
| `silence_rms` | `0.006` | STT silence gate |
| `model_name` | — | free-form backend model id for your task |

## Storage

| Key | Default | Description |
|---|---|---|
| `use_remote_storage` | `false` | upload artifacts when a storage backend is enabled |
| `storage_bucket_prefix` | task name | key prefix in the bucket |

## Example

```yaml
archetype: trainable_model
task_type: classification

input_dir: audio_input/crowd_mood
feature_type: logmel
sample_rate: 16000
segment_duration: 2.0
segment_overlap: 0.5
input_shape: [1, 128, 63]

class_labels: [calm, cheering, angry]
class_weights: [1.0, 1.0, 2.5]

split_strategy: group
batch_size: 32
epochs: 40
optimizer: adamw
learning_rate: 0.0005
weight_decay: 0.01
warmup_epochs: 2
early_stop_metric: val_acc
early_stop_patience: 6
```
