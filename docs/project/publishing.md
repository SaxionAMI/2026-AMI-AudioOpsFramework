# Publishing checklist (maintainers)

Steps to take before the repository goes public, and for each release.

## One-time, before going public

- [x] Links point to `SaxionAMI/2026-AMI-AudioOpsFramework` and the docs to
      <https://saxionami.github.io/2026-AMI-AudioOpsFramework/>.
- [x] Contact address in `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`
      and `pyproject.toml`: e.m.vanwolfswinkel@saxion.nl.
- [x] Copyright holder: Saxion University of Applied Sciences (`NOTICE`,
      `pyproject.toml`, `CITATION.cff`, docs footer).
- [ ] Check the git history for secrets, internal hostnames, customer audio or
      data before making the repo public. A clean way is to publish a fresh
      history (`git checkout --orphan public && git commit`) from this tree.
- [ ] Enable on GitHub: *Private vulnerability reporting*, *Discussions*,
      *Pages* (source: GitHub Actions), branch protection on `main` requiring CI.
- [ ] Add a `pypi` environment and configure **PyPI Trusted Publishing** for
      the `release.yml` workflow (project name `audio-ops-framework`).
- [ ] **Confirm the [Used in](used-in.md) page** with the AI-ZI project lead
      (the framework's role, permission to name partners) and remove the HTML
      maintainer comment at the top of the page afterwards.
- [ ] Make the GRASS repository public at the same time, so the link on the
      Used in page works.
- [ ] Optional: add the DCO GitHub App to enforce sign-offs.

## Every release

- [ ] Bump `version` in `pyproject.toml` and the fallback in
      `audio_ops_dagster/__init__.py`, update `CITATION.cff`.
- [ ] Move *Unreleased* entries in `CHANGELOG.md` under the new version.
- [ ] `make lint test build docs` locally.
- [ ] Tag `vX.Y.Z` on `main` and push. The release workflow publishes to PyPI
      and creates the GitHub release; docs deploy from `main`.
