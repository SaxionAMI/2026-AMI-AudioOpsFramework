"""
definitions.py — Dagster Definitions entry point.

v4: Replaced the training_enabled guard with the capability model. Each task
declares what it can do (capabilities / archetype in config.yaml); the builder
assembles only the matching assets and jobs. serve-only tasks (Whisper, YAMNet)
contribute nothing to the Dagster graph and run via the inference service.
Legacy configs using `dagster.training_enabled` still resolve correctly.
"""

import os
import yaml
import sys
from pathlib import Path
from typing import Optional, List
from dagster import Definitions, ConfigurableResource
import inspect

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio_ops_dagster.registry import TaskConfig, discover_tasks, task_search_dirs
from audio_ops_dagster.capabilities import resolve_capabilities, produces_dagster_assets, describe
from audio_ops_dagster.assets import build_audio_assets
from audio_ops_dagster.jobs import build_audio_jobs
from audio_ops_dagster.core.storage_resource import MinIOStorageResource
from audio_ops_dagster.core.influxdb_resource import InfluxDBMetricsResource
from audio_ops_dagster.core.metrics import NullMetricsResource
from audio_ops_dagster.core.task_config_schema import validate_task_config
from audio_ops_dagster.streaming.dagster_stream_sensor import (
    stream_discovery_sensor,
    stream_inference_job,
    stream_config_asset,
    stream_status_asset,
)

# Auto-discovery: imports all task modules, triggers @register_task decorators
discover_tasks()


class AudioTaskResource(ConfigurableResource):
    task_name: str
    input_dir: str
    output_dir: str
    feature_type: str = "mel"
    num_classes: Optional[int] = None
    class_labels: Optional[List[str]] = None
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"
    criterion: str = "cross_entropy"
    model_architecture: str = "cnn"
    split_strategy: str = "random"
    input_shape: Optional[List[int]] = None

    # ── Feature / segmentation knobs ─────────────────────────────────────────
    # Declared here too, otherwise the `valid_fields` filter below strips them
    # from config.yaml before the resource is built, and segmented_features'
    # getattr(config, "segment_duration", 1.0) silently falls back to 1.0.
    sample_rate: int = 16000
    segment_duration: float = 1.0
    segment_overlap: float = 0.25
    force_recompute: bool = False

    # ── Training knobs (flow through to_task_config() into the trainer) ──────
    weight_decay: float = 0.0
    warmup_epochs: int = 0
    grad_clip: Optional[float] = None
    early_stop_patience: int = 10
    early_stop_metric: str = "val_loss"
    class_weights: Optional[List[float]] = None

    use_remote_storage: bool = False
    storage_bucket_prefix: str = ""
    model_name: Optional[str] = None

    def to_task_config(self) -> TaskConfig:
        data = self.dict()
        expected_keys = set(inspect.signature(TaskConfig).parameters.keys())
        filtered_data = {k: v for k, v in data.items() if k in expected_keys}
        return TaskConfig(**filtered_data)


minio_storage_resource = MinIOStorageResource.from_env()

# Pick the metrics sink centrally: InfluxDB if explicitly enabled, otherwise a
# no-op. Assets only ever talk to "influxdb_metrics" — they don't know or care
# which sink it is.
if os.getenv("ENABLE_INFLUXDB", "false").lower() == "true":
    influxdb_metrics_resource = InfluxDBMetricsResource.from_env()
else:
    influxdb_metrics_resource = NullMetricsResource()

all_assets = []
all_jobs = []
all_resources = {
    "minio_storage": minio_storage_resource,
    "influxdb_metrics": influxdb_metrics_resource,
}

seen_tasks = set()
for tasks_root in task_search_dirs():
    if not tasks_root.exists():
        continue
    for task_dir in sorted(tasks_root.iterdir()):
        if not task_dir.is_dir():
            continue
        config_file = task_dir / "config.yaml"
        if not config_file.exists():
            continue

        task_name = task_dir.name
        if task_name in seen_tasks:
            print(f"  ⚠ duplicate task '{task_name}' in {tasks_root} ignored (already loaded)")
            continue
        seen_tasks.add(task_name)
        resource_key = f"{task_name}_config"

        with open(config_file) as f:
            data = yaml.safe_load(f)

        # Validate early so mistakes surface here with a clear message rather
        # than deep inside a run. Blocking errors skip the task; warnings inform.
        errors, warnings = validate_task_config(data)
        for w in warnings:
            print(f"  ⚠ {task_name}: {w}")
        if errors:
            for e in errors:
                print(f"  ✗ {task_name}: {e}")
            print(f"  → skipping '{task_name}' until its config is fixed.")
            continue

        # Resolve what this task can DO (explicit capabilities → archetype →
        # legacy training_enabled fallback). The guard is gone: assembly is
        # additive and each task only contributes the stages it declares.
        caps = resolve_capabilities(data)
        print(f"• {task_name}: capabilities = {describe(caps)}")

        # Every task registers its resource (used by Dagster batch assets and
        # available to tooling), regardless of capabilities.
        data["task_name"] = task_name
        data.setdefault("input_dir", f"audio_input/{task_name}/")
        data.setdefault("output_dir", f"storage/{task_name}/")
        data.setdefault("feature_type", "raw_waveform")
        valid_fields = set(AudioTaskResource.__fields__.keys())
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        all_resources[resource_key] = AudioTaskResource(**filtered)

        # serve-only tasks (Whisper, YAMNet) add nothing here — they live in the
        # inference service. Skip asset/job assembly entirely for them.
        if not produces_dagster_assets(caps):
            continue

        all_assets.extend(build_audio_assets(task_name, resource_key, caps))
        all_jobs.extend(build_audio_jobs(task_name, caps))


all_assets += [stream_config_asset, stream_status_asset]
all_jobs += [stream_inference_job]

defs = Definitions(
    assets=all_assets,
    jobs=all_jobs,
    resources=all_resources,
    sensors=[stream_discovery_sensor],
)