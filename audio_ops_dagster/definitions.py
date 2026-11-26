"""
definitions.py — Dagster ``Definitions`` entry point.

Discovers every task (built-in + ``AUDIO_OPS_TASKS_DIR``), validates its
``config.yaml``, resolves its capabilities and assembles only the assets and
jobs those capabilities need. A task with config errors is skipped with a
clear message instead of breaking the whole code location.

Serve-only tasks contribute nothing here — they run in the inference service.

Optional extras (env-driven, all off by default):
  ``STORAGE_BACKEND=minio``    object-storage resource
  ``ENABLE_INFLUXDB=true``     InfluxDB metrics sink (else a no-op sink)
"""

from __future__ import annotations

import json
import os
from typing import Optional

from dagster import ConfigurableResource, Definitions

from audio_ops_dagster._log import get_logger
from audio_ops_dagster.assets import build_audio_assets
from audio_ops_dagster.capabilities import describe, produces_dagster_assets, resolve_capabilities
from audio_ops_dagster.core.influxdb_resource import InfluxDBMetricsResource
from audio_ops_dagster.core.metrics import NullMetricsResource
from audio_ops_dagster.core.storage_resource import MinIOStorageResource
from audio_ops_dagster.jobs import build_audio_jobs
from audio_ops_dagster.registry import TaskConfig, check_task, discover_tasks, iter_task_dirs

logger = get_logger(__name__)

# Auto-discovery: imports all task modules, triggers @register_task decorators.
discover_tasks()


class AudioTaskResource(ConfigurableResource):
    """A task's config.yaml as a Dagster resource.

    The full config travels as JSON (validated into a :class:`TaskConfig` on
    use), so Dagster and the inference service always see identical settings.
    ``input_dir``/``output_dir`` are surfaced for the UI.
    """

    task_name: str
    input_dir: str
    output_dir: str
    config_json: str = "{}"

    def to_task_config(self) -> TaskConfig:
        return TaskConfig.from_dict(json.loads(self.config_json or "{}"),
                                    task_name=self.task_name)

    @classmethod
    def from_config(cls, task_name: str, raw: dict) -> AudioTaskResource:
        cfg = TaskConfig.from_dict(raw, task_name=task_name)
        return cls(task_name=task_name, input_dir=cfg.input_dir, output_dir=cfg.output_dir,
                   config_json=json.dumps(cfg.to_dict(), default=str))


def _metrics_resource():
    """InfluxDB if explicitly enabled, otherwise a no-op sink."""
    if os.getenv("ENABLE_INFLUXDB", "false").lower() == "true":
        return InfluxDBMetricsResource.from_env()
    return NullMetricsResource()


def build_definitions(loggers: Optional[dict] = None) -> Definitions:
    """Build the code location from the current environment (re-runs discovery).

    ``loggers`` replaces Dagster's default console logger (the CLI's
    ``audio-ops run`` passes a compact one).
    """
    discover_tasks()
    all_assets: list = []
    all_jobs: list = []
    resources: dict = {
        "minio_storage": MinIOStorageResource.from_env(),
        "influxdb_metrics": _metrics_resource(),
    }

    for task in iter_task_dirs():
        errors, warnings = check_task(task)
        for w in warnings:
            logger.warning("%s: %s", task.name, w)
        if errors:
            for e in errors:
                logger.error("%s: %s", task.name, e)
            logger.error("→ skipping '%s' until its config is fixed (run `audio-ops validate`).",
                         task.name)
            continue

        caps = resolve_capabilities(task.raw_config)
        logger.info("• %s: capabilities = %s", task.name, describe(caps))

        resource_key = f"{task.name}_config"
        resources[resource_key] = AudioTaskResource.from_config(task.name, task.raw_config)

        if not produces_dagster_assets(caps):
            continue  # serve-only: lives in the inference service
        all_assets.extend(build_audio_assets(task.name, resource_key, caps))
        all_jobs.extend(build_audio_jobs(task.name, caps))

    return Definitions(assets=all_assets, jobs=all_jobs, resources=resources,
                       loggers=loggers)


defs = build_definitions()
