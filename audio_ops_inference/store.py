"""
store.py — discovers servable tasks and builds their serve wrappers.

A task is loaded when its config declares the ``serve`` capability. Its
wrapper type comes from ``serve_wrapper`` or is inferred from the task type
(see :func:`audio_ops_inference.wrappers.resolve_wrapper_type`). Checkpoints
are found locally first and, with ``STORAGE_BACKEND=minio``, pulled from
object storage. Only weights-only checkpoints are loaded unless
``ALLOW_UNSAFE_CHECKPOINTS=true``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from audio_ops_dagster._log import get_logger
from audio_ops_inference.wrappers import ServeWrapper, get_wrapper_factory, resolve_wrapper_type

logger = get_logger("audio_ops_inference.store")


class ModelStore:
    def __init__(self):
        self._models: Dict[str, ServeWrapper] = {}
        self._skipped: Dict[str, str] = {}

    def load_all(self) -> None:
        """(Re)load every servable task. Swaps the model table atomically."""
        models: Dict[str, ServeWrapper] = {}
        skipped: Dict[str, str] = {}
        self._load_task_models(models, skipped)
        self._models, self._skipped = models, skipped
        if not self._models:
            logger.warning("No models loaded — inference endpoints will return empty results. "
                           "Train a task (or add a serve-only task) and restart.")
        else:
            logger.info("Models ready: %s", sorted(self._models))

    def _candidates(self):
        """(task_name, task_dir, raw_cfg) for every discovered task directory."""
        from audio_ops_dagster.registry import discover_tasks, iter_task_dirs

        discover_tasks()
        for t in iter_task_dirs():
            yield t.name, str(t.directory), t.raw_config

    def _load_task_models(self, models: Dict[str, ServeWrapper],
                          skipped: Dict[str, str]) -> None:
        from audio_ops_dagster.capabilities import SERVE, CapabilityError, resolve_capabilities
        from audio_ops_dagster.registry import TaskConfig, get_task

        for task_name, task_dir, raw in self._candidates():
            try:
                if SERVE not in resolve_capabilities(raw):
                    continue
            except CapabilityError as exc:
                skipped[task_name] = str(exc)
                logger.warning("Skipping '%s': %s", task_name, exc)
                continue
            try:
                cfg = TaskConfig.from_dict(raw, task_name=task_name)
                task = get_task(task_name, cfg)
                wtype = resolve_wrapper_type(task, cfg)
                factory = get_wrapper_factory(wtype)
                if factory is None:
                    skipped[task_name] = f"no serve wrapper '{wtype}'"
                    logger.warning("No serve wrapper '%s' for task '%s' — register one with "
                                   "@register_wrapper.", wtype, task_name)
                    continue
                wrapper = factory(task, task_name, cfg)
                if wrapper is None:
                    skipped[task_name] = "not ready (no checkpoint?)"
                    continue
                models[task_name] = wrapper
                logger.info("Serving '%s' via '%s' wrapper (%s).", task_name, wtype, task_dir)
            except Exception as exc:  # noqa: BLE001 — one bad task must not kill the API
                skipped[task_name] = str(exc)
                logger.warning("Could not load task %s: %s", task_name, exc)

    def get(self, task: str) -> Optional[ServeWrapper]:
        return self._models.get(task)

    def all(self) -> Dict[str, ServeWrapper]:
        return dict(self._models)

    def names(self) -> List[str]:
        return sorted(self._models)

    def skipped(self) -> Dict[str, str]:
        return dict(self._skipped)

    def for_stream(self) -> Dict[str, ServeWrapper]:
        """Fresh per-stream wrapper instances (shared models, private buffers)."""
        return {name: w.for_stream() for name, w in self._models.items()}
