#!/usr/bin/env python3
"""Backwards-compatible wrapper: prefer `audio-ops new-task`.

    python scripts/new_task.py wake_word
    python scripts/new_task.py loudness --type regression
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio_ops_dagster.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["new-task", *sys.argv[1:]]))
