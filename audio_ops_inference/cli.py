"""
cli.py — console entry point for the inference service.

Installed as `audio-ops-serve` (see pyproject [tool.poetry.scripts]). Runs the
FastAPI app with uvicorn. A thin project repo can simply:

    pip install audio-ops-framework
    AUDIO_OPS_TASKS_DIR=./tasks audio-ops-serve

Honors INFERENCE_PORT, INFERENCE_HOST, INFERENCE_WORKERS.
"""

import os


def main() -> None:
    import uvicorn
    uvicorn.run(
        "audio_ops_inference.api:app",
        host=os.environ.get("INFERENCE_HOST", "0.0.0.0"),
        port=int(os.environ.get("INFERENCE_PORT", "8000")),
        workers=int(os.environ.get("INFERENCE_WORKERS", "1")),
    )


if __name__ == "__main__":
    main()
