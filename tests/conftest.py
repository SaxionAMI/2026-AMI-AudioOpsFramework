import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Quiet, hermetic defaults for every test.
os.environ.setdefault("AUDIO_OPS_LOG_LEVEL", "WARNING")
os.environ.setdefault("ENABLE_MLFLOW", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("DEVICE", "cpu")
for var in ("STORAGE_BACKEND", "ENABLE_INFLUXDB", "API_KEY", "AUDIO_OPS_TASKS_DIR"):
    os.environ.pop(var, None)


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(pytest.mark.slow)


@pytest.fixture
def demo_project(tmp_path, monkeypatch):
    """A temp working dir with synthetic labelled audio for example_classifier."""
    from audio_ops_dagster.demo import write_dataset

    monkeypatch.chdir(tmp_path)
    write_dataset(tmp_path / "audio_input" / "example_classifier", files_per_class=6,
                  seconds=2.0)
    monkeypatch.setenv("DAGSTER_HOME", str(tmp_path / "dagster_home"))
    (tmp_path / "dagster_home").mkdir()
    return tmp_path
