import textwrap

import pytest

pytest.importorskip("torch")

from audio_ops_dagster import registry  # noqa: E402
from audio_ops_dagster.registry import TaskConfig  # noqa: E402


def test_taskconfig_defaults_and_extras():
    cfg = TaskConfig.from_dict({"num_classes": 2, "stream_buffer_seconds": 3.5}, task_name="t")
    assert cfg.input_dir == "audio_input/t" and cfg.output_dir == "storage/t"
    assert cfg.feature_type == "spectrogram" and cfg.optimizer == "adamw"
    assert cfg.stream_buffer_seconds == 3.5           # unknown keys survive as attributes
    assert cfg.to_dict()["stream_buffer_seconds"] == 3.5


def _write_external_task(root, name="ext_task", task_name=None):
    d = root / name
    d.mkdir(parents=True)
    (d / "__init__.py").write_text("")
    (d / "helpers.py").write_text("OUT = 3\n")
    (d / f"task_{name}.py").write_text(textwrap.dedent(f'''
        import torch.nn as nn
        from audio_ops_dagster.core.base_task import AudioTask
        from audio_ops_dagster.registry import register_task
        from .helpers import OUT            # relative import must work

        @register_task("{name}")
        class ExtTask(AudioTask):
            def build_model(self, input_shape):
                return nn.Linear(4, OUT)
            def compute_loss(self, model, batch, device):
                raise NotImplementedError
            def compute_metrics(self, y_true, y_pred):
                return {{}}
            def postprocess(self, raw_output):
                return {{}}
    '''))
    (d / "config.yaml").write_text(
        "archetype: pretrained_servable\n" + (f"task_name: {task_name}\n" if task_name else ""))
    return d


def test_external_task_dir_with_relative_import(tmp_path, monkeypatch):
    _write_external_task(tmp_path)
    monkeypatch.setenv("AUDIO_OPS_TASKS_DIR", str(tmp_path))
    registry.discover_tasks()
    assert "ext_task" in registry.list_tasks()
    task = registry.get_task("ext_task", {"num_classes": 1})
    assert task.build_model(None).out_features == 3
    names = [t.name for t in registry.iter_task_dirs()]
    assert "ext_task" in names and "example_classifier" in names


def test_builtin_tasks_can_be_hidden(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_OPS_DISABLE_BUILTIN_TASKS", "true")
    assert "example_classifier" not in [t.name for t in registry.iter_task_dirs()]


def test_check_task_flags_name_mismatch(tmp_path, monkeypatch):
    _write_external_task(tmp_path, name="ext_two", task_name="other_name")
    monkeypatch.setenv("AUDIO_OPS_TASKS_DIR", str(tmp_path))
    registry.discover_tasks()
    task = next(t for t in registry.iter_task_dirs() if t.name == "ext_two")
    errors, _ = registry.check_task(task)
    assert any("does not match its folder name" in e for e in errors)


def test_check_task_flags_missing_registration(tmp_path, monkeypatch):
    d = tmp_path / "orphan"
    d.mkdir()
    (d / "config.yaml").write_text("archetype: trainable_model\n")
    monkeypatch.setenv("AUDIO_OPS_TASKS_DIR", str(tmp_path))
    task = next(t for t in registry.iter_task_dirs() if t.name == "orphan")
    errors, _ = registry.check_task(task)
    assert any("no class registered" in e for e in errors)


def test_get_task_unknown_has_helpful_message():
    with pytest.raises(ValueError, match="register_task"):
        registry.get_task("definitely_missing", {})
