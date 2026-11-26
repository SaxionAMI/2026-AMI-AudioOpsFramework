# Third-party software

The framework does not vendor third-party code. It depends on the packages
below, installed from PyPI (or your package index) under their own licenses.
All are permissive and compatible with this project's Apache-2.0 license.
Versions and licenses can change — check the packages you actually install
(`pip-licenses` is a convenient tool for that).

| Package | Used for | Extra | License (as published) |
|---|---|---|---|
| NumPy | arrays | core | BSD-3-Clause |
| PyYAML | config files | core | MIT |
| pydantic | config validation | core | MIT |
| librosa | audio IO + features | core | ISC |
| soundfile (libsndfile) | audio IO | core | BSD-3-Clause (libsndfile: LGPL-2.1, dynamically linked) |
| PyTorch | models, training, serving | train, serve | BSD-3-Clause |
| Dagster, dagster-webserver | orchestration | train | Apache-2.0 |
| MLflow | experiment tracking | train | Apache-2.0 |
| scikit-learn | splits, metrics | train | BSD-3-Clause |
| pandas | split CSVs | train | BSD-3-Clause |
| FastAPI, Starlette | inference API | serve | MIT / BSD-3-Clause |
| Uvicorn | ASGI server | serve | BSD-3-Clause |
| python-multipart | file uploads | serve | Apache-2.0 |
| transformers, torchaudio | `feature_type: ast` | ast | Apache-2.0 / BSD-2-Clause |
| boto3 | S3/MinIO storage | s3 | Apache-2.0 |
| influxdb-client | metrics sink | metrics | MIT |

## Container images and services

| Component | Where | License (as published) |
|---|---|---|
| `python:3.11-slim` base image | Dockerfiles | various (Debian + PSF) |
| FFmpeg (Debian/Alpine package) | Dockerfiles | LGPL-2.1+ / GPL-2.0+ depending on build |
| MediaMTX (`bluenviron/mediamtx`) | `mediamtx/` | MIT |

FFmpeg is invoked as a separate executable, never linked. If you redistribute
images, ship the corresponding license texts (the distro packages include them).

## Models and data

No pretrained weights or audio datasets are included. `audio-ops demo-data`
*generates* synthetic tones, so the demo audio carries no third-party rights.
Weights downloaded at runtime by tasks (AST, Whisper, YAMNet, …) are governed by
their publishers' licenses.
