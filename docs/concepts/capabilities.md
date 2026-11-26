# Capabilities

Every task resolves to a set of capabilities (`capabilities.py`):

| Capability | Meaning | Dagster assets built |
|---|---|---|
| `train` | produces a checkpoint | raw_audio → features → splits → **trained_model** |
| `evaluate` | scores the test split | raw_audio → features → splits → **model_evaluation** |
| `batch_infer` | offline scoring of a dataset | raw_audio → features → **inference_results** |
| `serve` | online serving via the API | *(none, handled by the inference service)* |

With `train` + `evaluate`, evaluation scores the freshly trained checkpoint.
With `evaluate` **without** `train`, it scores an existing checkpoint
(`checkpoint_path` in the config, else `<output_dir>/checkpoints/best_model.pt`).

## Archetypes

Named bundles so configs read as intent:

```yaml
archetype: trainable_model        # train + evaluate + batch_infer + serve
archetype: pretrained_servable    # serve only (e.g. Whisper, YAMNet)
archetype: pretrained_batch       # batch_infer only
archetype: eval_only              # evaluate only (benchmark a checkpoint)
```

Or list capabilities explicitly. Archetype names may be mixed in:

```yaml
capabilities: [train, evaluate]   # train + eval, never served
capabilities: [pretrained_batch, serve]
```

## Resolution order

`resolve_capabilities(config)`:

1. an explicit `capabilities:` list wins;
2. otherwise `archetype:`;
3. otherwise the legacy `dagster: {training_enabled: false}` flag (→ serve only);
4. otherwise `trainable_model`.

Unknown names raise a `CapabilityError` with the valid choices, shown by
`audio-ops validate` and when Dagster loads.

## How assembly works

`definitions.py` loops over every discovered task and:

- validates it (`check_task`: schema, folder/`task_name` match, a registered class);
- registers its config as a Dagster resource (`<task>_config`), always;
- calls `build_audio_assets(task, resource_key, caps)`, which returns only the
  assets the capabilities need, resolving prerequisites;
- calls `build_audio_jobs(task, caps)`, which only emits jobs whose assets exist.

| Capabilities include | Jobs |
|---|---|
| `train` | `<task>_full_training_job` (whole group), `<task>_data_prep_job` |
| `evaluate` without `train` | `<task>_evaluation_job`, `<task>_data_prep_job` |
| `batch_infer` | `<task>_inference_job` |

A serve-only task passes `produces_dagster_assets(caps) == False` and adds
nothing to Dagster. It exists only for the API. This is what lets a
transcription project deploy only the API, while a classifier project deploys
the full training stack.
