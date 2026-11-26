---
title: Comparison with other tools
---

# Comparison with other tools

Audio Ops was written for applied research projects where a small team trains
audio models on its own recordings and then runs them on live streams. In our
projects this meant crowd audio in football stadiums and speech in care
settings. We kept running into the same problems: feature code that differed
between training and serving, separate repositories for the training pipeline
and the live service, and recordings leaking between train and test splits.

The framework itself uses standard parts: Dagster, PyTorch, FastAPI, MLflow,
ffmpeg and Whisper. What it adds is the way they are connected:

- A task is a folder with a `config.yaml` and, if needed, a Python file. The
  config lists what the task can do (`train`, `evaluate`, `batch_infer`,
  `serve`), and the framework only builds those stages. A serve-only model adds
  no Dagster assets. A model that is only trained never starts the API.
- The Dagster pipeline and the streaming API use the same task class and the
  same code for segmentation, feature extraction and input shaping. A model
  sees the same features in training, in batch scoring and on a live stream.
- All results on a stream are stamped with one media clock. The optional
  fusion timeline puts the transcript, non-verbal events and sentiment side by
  side, so you can check what was said when the crowd got louder.
- Standard classification, regression and detection tasks need only a config
  file. You write Python when you want a different model, loss or metric.
- Nothing has to be installed besides the Python packages. Storage, tracking
  and metrics default to local files and SQLite. S3/MinIO, InfluxDB, an MLflow
  server and an LLM are enabled with environment variables.

## Other tools and how they relate

### Speech and audio toolkits

SpeechBrain, ESPnet, NVIDIA NeMo, Hugging Face Transformers, torchaudio,
pyannote, openSMILE and Essentia provide models, recipes and feature
extractors. SpeechBrain alone has
[more than 200 recipes and over 100 pretrained models](https://arxiv.org/abs/2407.00463).
They are built for training and research. They do not include a pipeline
orchestrator or a service that ingests several live streams.

You can use them inside Audio Ops. A pretrained model becomes a
`pretrained_servable` or `pretrained_batch` task, or you return one from
`build_model`. The AST feature extractor and the Whisper speech-to-text lane
already work this way.

### Voice agent frameworks

Pipecat, LiveKit Agents, NVIDIA Riva and WhisperLive are made for
conversational agents that listen, call an LLM and speak back with low
latency. [Pipecat](https://github.com/pipecat-ai/pipecat) lists voice
assistants and customer-support bots as typical uses. They do not train models.

Audio Ops does not talk back. It runs several analysers on incoming streams,
stores the time-aligned results and metrics, and retrains the analysers on new
recordings. If you need a voice bot, use one of the frameworks above.

### Edge platforms

[Edge Impulse](https://docs.edgeimpulse.com/docs/audio-classification) has a
similar pipeline idea: raw audio, MFCC/MFE/spectrogram processing and a
classifier, deployed to microcontrollers from a hosted studio. Audio Ops runs
on servers and in containers, takes RTSP and WebSocket input and can use larger
models such as AST and Whisper. It does not target microcontrollers.

### Orchestration and experiment tracking

Audio Ops runs on Dagster and logs to MLflow. ZenML, Kedro, Metaflow, Flyte and
Kubeflow are in the same category. None of them handles audio specifically, so
you would write segmentation, feature caching, splits per recording, label
folders and checkpoint handling yourself. Audio Ops has those built in.

### Model serving

BentoML, Ray Serve, NVIDIA Triton, KServe and TorchServe serve models over
request/response APIs. They do not pull RTSP streams, keep state per stream or
combine the output of several models on one timeline. If one model needs a lot
of throughput, you can host it on Triton and call it from an Audio Ops serve
wrapper.

## When to use it

Audio Ops fits when:

- you work with a partner organisation and need to go from recordings to a
  running demo and then a trial on site;
- several analyses have to run on the same streams and line up in time;
- some team members don't write PyTorch, and config-only tasks and the CLI
  are enough for them;
- audio has to stay on your own hardware.

Use something else when:

- you do ASR or TTS research or large-scale pretraining (SpeechBrain, ESPnet,
  NeMo);
- you build a conversational voice bot (Pipecat, LiveKit Agents);
- the model has to run on a microcontroller (Edge Impulse, TFLite Micro);
- you only need one model behind one REST endpoint (BentoML or a small FastAPI
  app is simpler).

## Background

Audio Ops started at Saxion University of Applied Sciences and has been used in
the projects listed under [Used in](../project/used-in.md).
