# Contributing

Thanks for helping improve the Audio Ops Framework! Bug reports, docs fixes,
new feature extractors, serve wrappers and example tasks are all welcome.

## Ground rules

- Be kind. This project follows the [Code of Conduct](https://github.com/SaxionAMI/2026-AMI-AudioOpsFramework/blob/main/CODE_OF_CONDUCT.md).
- Security issues go through a **private** report, never a public issue. See
  [SECURITY.md](https://github.com/SaxionAMI/2026-AMI-AudioOpsFramework/blob/main/SECURITY.md).
- For anything bigger than a bug fix, open an issue or discussion first so we can
  agree on the approach before you invest time.

## Development setup

```bash
git clone https://github.com/SaxionAMI/2026-AMI-AudioOpsFramework
cd 2026-AMI-AudioOpsFramework
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: CPU-only torch
make install            # pip install -e ".[dev,docs]" + pre-commit hooks
make test               # unit + end-to-end tests (CPU, ~30 s)
```

`ffmpeg` and `libsndfile` must be on your system for the streaming paths
(`apt install ffmpeg libsndfile1` / `brew install ffmpeg libsndfile`).

Handy targets: `make test-fast` (unit tests only), `make lint`, `make format`,
`make docs-serve` (live docs preview), `make demo && make dev` (try the pipeline).

## Design principles (please keep them)

1. **Tasks, not pipelines.** New behaviour should plug in through a registry
   (`@register_task`, `@register_feature`, `@register_wrapper`) or a task hook,
   not through a task-name `if` in core code.
2. **Capabilities are additive.** A task declares what it does; the builders add
   exactly those stages. Serve-only tasks contribute nothing to Dagster.
3. **Zero required services.** Every integration (object storage, InfluxDB,
   MLflow server, LLM, fusion) is optional, env-driven and degrades to a no-op.
4. **Train/serve parity.** Features, segmentation and input shaping are shared
   code (`core/runtime.py`, `core/audio_processor.py`), never re-implemented.
5. **Extras stay separate.** `serve` must not import Dagster/MLflow; the core
   must not import torch at module import time.

## Pull requests

- Add or update tests (`tests/`). End-to-end tests use synthetic audio from
  `audio_ops_dagster.demo`. Never commit real recordings.
- Update the docs under `docs/` when behaviour or configuration changes, and add
  a line to `CHANGELOG.md` under **Unreleased**.
- Keep PRs focused; one logical change per PR.
- CI must be green (lint, tests on Python 3.10–3.13, serve-only install, package
  build, docs build).

### Developer Certificate of Origin

We use the [DCO](https://developercertificate.org/) instead of a CLA. Sign off
every commit to certify you wrote it or have the right to submit it under the
project's Apache-2.0 license:

```bash
git commit -s -m "Fix segment offsets in batch inference"
```

This adds a `Signed-off-by: Your Name <you@example.com>` trailer.

## Releasing (maintainers)

1. Update `version` in `pyproject.toml` and the fallback in
   `audio_ops_dagster/__init__.py`; move *Unreleased* notes in `CHANGELOG.md`
   under the new version.
2. Merge to `main`, then tag: `git tag vX.Y.Z && git push --tags`.
3. The `Release` workflow builds, checks and publishes to PyPI (Trusted
   Publishing) and creates the GitHub release. Docs deploy on every push to `main`.
