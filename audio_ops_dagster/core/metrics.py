"""
metrics.py — a small metrics façade so the pipeline depends on a *sink*, not on
InfluxDB specifically.

Assets emit domain events (training metrics, pipeline events, inference points).
Where those go is a deployment choice:

  • InfluxDBMetricsResource  — writes to InfluxDB     (core/influxdb_resource.py)
  • NullMetricsResource      — drops everything       (default; no service needed)

Both expose the same method surface, so swapping one for the other changes
nothing in asset code. `definitions.py` picks based on ENABLE_INFLUXDB.

To add a sink (Prometheus, stdout, a file), subclass NullMetricsResource and
override the writers you care about.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from dagster import ConfigurableResource


class NullMetricsResource(ConfigurableResource):
    """No-op metrics sink. Every write is silently dropped."""

    def write_point(
        self,
        measurement: str,
        tags: Dict[str, str],
        fields: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> None:
        return None

    def write_points(
        self,
        measurement: str,
        records: List[Dict[str, Any]],
        common_tags: Optional[Dict[str, str]] = None,
    ) -> None:
        return None

    def write_inference_result(self, *args, **kwargs) -> None:
        return None

    def write_training_metric(self, *args, **kwargs) -> None:
        return None

    def write_pipeline_event(self, *args, **kwargs) -> None:
        return None
