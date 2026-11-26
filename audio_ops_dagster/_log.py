"""Tiny logging helper shared by the framework.

Framework modules log through ``logging.getLogger("audio_ops.<module>")``. When
the host application (Dagster, uvicorn, a notebook) has not configured logging,
``get_logger`` attaches one plain stream handler to the ``audio_ops`` root so
discovery/validation messages are still visible instead of silently dropped.
Set ``AUDIO_OPS_LOG_LEVEL`` (default ``INFO``) to change verbosity.
"""

from __future__ import annotations

import logging
import os

_ROOT = "audio_ops"
_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    root = logging.getLogger(_ROOT)
    if not _configured:
        _configured = True
        level = os.environ.get("AUDIO_OPS_LOG_LEVEL", "INFO").upper()
        root.setLevel(getattr(logging, level, logging.INFO))
        if not root.handlers and not logging.getLogger().handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
            root.addHandler(handler)
    short = name.split(".")[-1] if name.startswith("audio_ops_") else name
    return logging.getLogger(f"{_ROOT}.{short}")
