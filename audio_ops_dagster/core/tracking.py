"""
tracking.py — optional experiment tracking (MLflow) that never breaks training.

Training logs params/metrics/artifacts through this module. If MLflow is not
installed, is disabled (``ENABLE_MLFLOW=false``) or the tracking server is
unreachable, every call becomes a no-op and a single warning is logged — a
flaky tracking backend should never fail a training run.

Default backend is a local SQLite file (``sqlite:///mlruns/mlflow.db``); set
``MLFLOW_TRACKING_URI`` to point at a server (``http://mlflow:5000``) or any
other MLflow backend.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Dict, Iterator, Optional

from audio_ops_dagster._log import get_logger

logger = get_logger(__name__)

DEFAULT_TRACKING_URI = "sqlite:///mlruns/mlflow.db"


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI


def _ensure_sqlite_dir(uri: str) -> None:
    if uri.startswith("sqlite:///"):
        path = uri[len("sqlite:///"):]
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)


class Tracker:
    """Thin façade over MLflow's fluent API with graceful degradation."""

    def __init__(self) -> None:
        self._mlflow = None
        self._warned = False
        if os.environ.get("ENABLE_MLFLOW", "true").lower() in ("0", "false", "no"):
            return
        try:
            import mlflow

            uri = tracking_uri()
            _ensure_sqlite_dir(uri)
            mlflow.set_tracking_uri(uri)
            self._mlflow = mlflow
        except Exception as exc:  # noqa: BLE001
            self._warn(f"MLflow unavailable ({exc}); experiment tracking disabled.")

    @property
    def enabled(self) -> bool:
        return self._mlflow is not None

    def _warn(self, msg: str) -> None:
        if not self._warned:
            logger.warning(msg)
            self._warned = True

    def _call(self, fn_name: str, *args, **kwargs) -> Any:
        if self._mlflow is None:
            return None
        try:
            return getattr(self._mlflow, fn_name)(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self._warn(f"MLflow call {fn_name} failed ({exc}); continuing without tracking.")
            self._mlflow = None
            return None

    @contextlib.contextmanager
    def run(self, experiment: str, run_name: Optional[str] = None) -> Iterator[str]:
        """Start (or join) a run. Yields the run id ('' when tracking is off)."""
        if self._mlflow is None:
            yield ""
            return
        active = self._call("active_run")
        if active is not None:
            yield active.info.run_id
            return
        self._call("set_experiment", experiment)
        run = self._call("start_run", run_name=run_name)
        if run is None:
            yield ""
            return
        try:
            yield run.info.run_id
        finally:
            self._call("end_run")

    def log_params(self, params: Dict[str, Any]) -> None:
        self._call("log_params", {k: str(v)[:500] for k, v in params.items()})

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        clean = {}
        for k, v in metrics.items():
            try:
                clean[k] = float(v)
            except (TypeError, ValueError):
                continue
        if clean:
            self._call("log_metrics", clean, step=step)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        self._call("log_artifact", path, artifact_path=artifact_path)
