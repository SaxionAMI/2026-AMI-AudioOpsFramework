# Security policy

## Supported versions

Security fixes are made on the latest minor release. Upgrade to it before
reporting.

| Version | Supported |
|---------|-----------|
| 1.2.x   | ✅        |
| < 1.2   | ❌        |

## Reporting a vulnerability

Please **do not** open a public issue. Use GitHub's
[private vulnerability reporting](https://github.com/SaxionAMI/2026-AMI-AudioOpsFramework/security/advisories/new)
or email **e.m.vanwolfswinkel@saxion.nl**. Include
the version, a description of the impact and steps to reproduce. You'll get an
acknowledgement within a few working days; we'll agree on a disclosure date
together once a fix is ready.

## Hardening notes for deployments

- **Set `API_KEY`** whenever the inference service is reachable by anyone else.
  Without it every endpoint (including stream registration, which makes the
  server run `ffmpeg` on a URL you supply) is open. Put the service behind TLS.
- **Restrict `CORS_ALLOW_ORIGINS`** to the dashboards that need it.
- **Checkpoints are code if pickled.** The framework writes weights-only
  checkpoints and loads them with `torch.load(weights_only=True)`. Only set
  `ALLOW_UNSAFE_CHECKPOINTS=true` for files you trust completely.
- `POST /streams` accepts arbitrary URLs and `ffmpeg` will fetch them. Don't
  expose it to untrusted callers, or front it with an allow-list proxy.
- Uploads are capped by `MAX_UPLOAD_MB` (default 100).
- Never commit `.env`; credentials for S3/InfluxDB/LLM are read from the
  environment and no defaults are baked in.
