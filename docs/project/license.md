# License & notices

The Audio Ops Framework is licensed under the **Apache License, Version 2.0**.

## Copyright

The copyright is held by Saxion University of Applied Sciences (research group
Ambient Intelligence). The framework was developed in applied research projects
co-funded by Regieorgaan SIA and is published as open source so that others can
use and build on the results.

## Why Apache-2.0

- **Permissive**: you may use it commercially, modify it and ship it in closed
  products. You have to keep the license and notices and state your changes.
- **Explicit patent grant** from contributors, which matters for ML/audio
  tooling adopted by companies.
- **Same license as the ecosystem it builds on** (Dagster, MLflow, transformers,
  boto3) and compatible with every dependency (BSD, MIT, ISC).

Contributions are accepted under the same license with a
[DCO sign-off](contributing.md#developer-certificate-of-origin). There is no CLA.

## What's included

The repository contains no third-party source code, model weights or audio.
Dependencies are installed from their own distributions under their own
licenses. See `THIRD_PARTY.md` in the repository. The demo audio is generated
synthetically at runtime.

!!! warning "Models carry their own licenses"
    Weights your tasks download (AST, Whisper, YAMNet, …) and datasets you train
    on have their own terms. Check them before redistributing a trained
    checkpoint or a container image that bundles them.

## Full text

```text
--8<-- "LICENSE"
```

## NOTICE

```text
--8<-- "NOTICE"
```
