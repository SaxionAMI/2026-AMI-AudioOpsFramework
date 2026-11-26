"""
audio_ops — the public API of the Audio Ops Framework, in one import.

    from audio_ops import ClassificationTask, register_task

    @register_task("wake_word")
    class WakeWord(ClassificationTask):
        pass

Names are loaded lazily, so ``import audio_ops`` stays cheap and only pulls in
torch when you touch a task class. The implementation lives in
``audio_ops_dagster`` (shared core + Dagster) and ``audio_ops_inference`` (the
streaming service); those import paths keep working.
"""

from __future__ import annotations

import importlib
from typing import Any

from audio_ops_dagster import __version__

_EXPORTS = {
    # task contract
    "AudioTask": "audio_ops_dagster.core.base_task",
    "ClassificationTask": "audio_ops_dagster.core.task_types",
    "RegressionTask": "audio_ops_dagster.core.task_types",
    "DetectionTask": "audio_ops_dagster.core.task_types",
    "SmallAudioCNN": "audio_ops_dagster.core.models",
    # registries
    "register_task": "audio_ops_dagster.registry",
    "get_task": "audio_ops_dagster.registry",
    "list_tasks": "audio_ops_dagster.registry",
    "register_feature": "audio_ops_dagster.core.features",
    "list_features": "audio_ops_dagster.core.features",
    "register_wrapper": "audio_ops_inference.wrappers",
    "ServeWrapper": "audio_ops_inference.wrappers",
    # config & helpers
    "TaskConfig": "audio_ops_dagster.core.config",
    "validate_task_config": "audio_ops_dagster.core.task_config_schema",
    "load_task_model": "audio_ops_dagster.core.runtime",
    "LLMClient": "audio_ops_dagster.core.llm_client",
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'audio_ops' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
