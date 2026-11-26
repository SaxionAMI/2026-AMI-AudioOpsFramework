# Concepts

Three ideas explain most of how the framework works.

## Tasks

You don't edit the pipeline. You write a task: a folder with a `config.yaml`
and, if the ready-made bases don't fit, a class that implements four methods
(`build_model`, `compute_loss`, `compute_metrics` and `postprocess`). The
framework finds the folder and builds the pipeline for it.

## Capabilities

A task lists what it can do: `train`, `evaluate`, `batch_infer` and/or `serve`.
The framework adds only those stages. A model that only serves adds no Dagster
assets, and a project that only trains never starts the API.

## Batch inference and serving

"Inference" means two different things here. Scoring a dataset offline is a
Dagster asset (`batch_infer`). The RTSP/WebSocket/SSE service is a separate
long-running process (`serve`). The two are configured and deployed
separately.

## More detail

- [Architecture](architecture.md): the two runtimes and what they share
- [Capabilities](capabilities.md): how a config becomes a pipeline
- [The training pipeline](pipeline.md): each asset in turn
- [Serving](serving.md): how the inference service loads and runs tasks
- [Storage](storage.md): artifact layout and object storage
- [Comparison with other tools](why-audio-ops.md)
