"""
definitions.py — Dagster Definitions entry point.

v2 additions:
  • InfluxDBMetricsResource registered as "influxdb_metrics"
  • All assets now have access to time-series logging
  • Grafana dashboard auto-provisioning on startup
  • Stream inference sensor + jobs wired in
"""

import os
import yaml
import sys
from pathlib import Path
from typing import Optional, List
from dagster import Definitions, ConfigurableResource
import inspect

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio_ops_dagster.registry import TaskConfig, discover_tasks
from audio_ops_dagster.assets import build_audio_assets
from audio_ops_dagster.jobs import build_audio_jobs
from audio_ops_dagster.core.storage_resource import MinIOStorageResource
from audio_ops_dagster.core.influxdb_resource import InfluxDBMetricsResource
from audio_ops_dagster.streaming.dagster_stream_sensor import (
    stream_discovery_sensor,
    stream_inference_job,
    stream_config_asset,
    stream_status_asset,
)

# ── 1. Automatic task code discovery ─────────────────────────────────────────
discover_tasks()


# ── 2. AudioTaskResource — mirrors TaskConfig + orchestration-only fields ────
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
    use_remote_storage: bool = False
    storage_bucket_prefix: str = ""
    model_name: Optional[str] = None

    def to_task_config(self) -> TaskConfig:
        data = self.dict()
        expected_keys = set(inspect.signature(TaskConfig).parameters.keys())
        filtered_data = {k: v for k, v in data.items() if k in expected_keys}
        return TaskConfig(**filtered_data)


# ── 3. Build shared resources from environment variables ─────────────────────
minio_storage_resource = MinIOStorageResource.from_env()
influxdb_metrics_resource = InfluxDBMetricsResource.from_env()


# ── 4. Iterate task folders and build assets/jobs ────────────────────────────
all_assets = []
all_jobs = []
all_resources = {
    "minio_storage": minio_storage_resource,
    "influxdb_metrics": influxdb_metrics_resource,
}

tasks_root = Path(__file__).parent / "tasks"

if tasks_root.exists():
    for task_dir in tasks_root.iterdir():
        if task_dir.is_dir() and (task_dir / "config.yaml").exists():
            task_name = task_dir.name
            resource_key = f"{task_name}_config"

            with open(task_dir / "config.yaml") as f:
                data = yaml.safe_load(f)

                data["task_name"] = task_name
                data.setdefault("input_dir", "audio_input/")
                data.setdefault("output_dir", f"storage/{task_name}/")

                valid_fields = set(AudioTaskResource.__fields__.keys())
                filtered = {k: v for k, v in data.items() if k in valid_fields}

                res_instance = AudioTaskResource(**filtered)
                all_resources[resource_key] = res_instance

            all_assets.extend(build_audio_assets(task_name, resource_key))
            all_jobs.extend(build_audio_jobs(task_name))


# ── 5. Streaming assets and jobs ─────────────────────────────────────────────
all_assets += [stream_config_asset, stream_status_asset]
all_jobs += [stream_inference_job]


# ── 6. Final aggregated Definitions ──────────────────────────────────────────
defs = Definitions(
    assets=all_assets,
    jobs=all_jobs,
    resources=all_resources,
    sensors=[stream_discovery_sensor],
)