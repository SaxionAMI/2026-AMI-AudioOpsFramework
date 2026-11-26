"""audio_ops_inference — the streaming inference service.

``audio_ops_inference.api:app`` is the FastAPI application; register custom
serve wrappers from ``audio_ops_inference.wrappers`` (importing it does not
start the app).
"""
