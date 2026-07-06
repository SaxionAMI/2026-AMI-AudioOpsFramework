"""
cli.py — console entry point for the inference service.

Installed as `audio-ops-serve` (see pyproject [tool.poetry.scripts]). Runs the
FastAPI app with uvicorn. A thin project repo can simply:

    pip install audio-ops-framework
    AUDIO_OPS_TASKS_DIR=./tasks audio-ops-serve

Honors INFERENCE_PORT, INFERENCE_HOST, INFERENCE_WORKERS.

Dev: pass --reload (or set INFERENCE_RELOAD=1) to auto-restart on code changes.
This is used by the dev compose override, which bind-mounts the framework source
so edits take effect without rebuilding the image. --reload forces a single
worker (uvicorn requirement).
"""

import os
import sys


def main() -> None:
    import uvicorn

    reload = "--reload" in sys.argv[1:] or os.environ.get("INFERENCE_RELOAD") == "1"
    uvicorn.run(
        "audio_ops_inference.api:app",
        host=os.environ.get("INFERENCE_HOST", "0.0.0.0"),
        port=int(os.environ.get("INFERENCE_PORT", "8000")),
        workers=1 if reload else int(os.environ.get("INFERENCE_WORKERS", "1")),
        reload=reload,
        # watch the installed package dir so bind-mounted edits trigger reloads
        reload_dirs=(["/usr/local/lib/python3.10/site-packages/audio_ops_inference"]
                     if reload else None),
    )


if __name__ == "__main__":
    main()