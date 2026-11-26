"""
cli.py — ``audio-ops-serve``, kept for backwards compatibility.

Equivalent to ``audio-ops serve``. Honors INFERENCE_HOST, INFERENCE_PORT,
INFERENCE_WORKERS; pass --reload (or INFERENCE_RELOAD=1) to auto-restart on
code changes (forces a single worker).
"""

import sys


def main() -> None:
    from audio_ops_dagster.cli import main as cli_main

    raise SystemExit(cli_main(["serve", *sys.argv[1:]]))


if __name__ == "__main__":
    main()
