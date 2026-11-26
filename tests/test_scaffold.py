import importlib.util

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from audio_ops_dagster.core.task_config_schema import validate_task_config  # noqa: E402
from audio_ops_dagster.registry import TaskConfig, get_task  # noqa: E402
from audio_ops_dagster.scaffold import TYPES, create_task  # noqa: E402


def _load(task_dir, name):
    spec = importlib.util.spec_from_file_location(f"scaffold_{name}", task_dir / f"task_{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("ttype", TYPES)
def test_scaffold_creates_valid_task(tmp_path, ttype):
    import yaml

    name = f"t_{ttype}"
    task_dir = create_task(name, ttype=ttype, tasks_dir=tmp_path)
    cfg = yaml.safe_load((task_dir / "config.yaml").read_text())
    errors, _ = validate_task_config(cfg)
    assert errors == []
    _load(task_dir, name)       # registers the class
    task = get_task(name, TaskConfig.from_dict(cfg, task_name=name))
    assert task.task_type == ttype


@pytest.mark.parametrize("ttype", ["classification", "regression", "detection"])
def test_scaffolded_task_trains(tmp_path, ttype):
    """The trainer must use the task's own loss — regression/detection included."""
    import yaml
    from torch.utils.data import DataLoader, TensorDataset

    from audio_ops_dagster.core.model_trainer import ModelTrainer

    name = f"train_{ttype}"
    task_dir = create_task(name, ttype=ttype, tasks_dir=tmp_path)
    _load(task_dir, name)
    raw = yaml.safe_load((task_dir / "config.yaml").read_text())
    raw.update(epochs=2, batch_size=4)
    cfg = TaskConfig.from_dict(raw, task_name=name)
    task = get_task(name, cfg)

    x = torch.randn(12, 1, 128, 32)
    y = (torch.rand(12) if ttype == "regression" else torch.randint(0, 2, (12,)))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    trainer = ModelTrainer(task, cfg, checkpoint_dir=str(tmp_path / "ckpt"))
    ckpt = trainer.train(task.build_model(None), loader, loader)
    assert (tmp_path / "ckpt" / "best_model.pt").exists() and ckpt.endswith("best_model.pt")
    loss, metrics = trainer.test(task.build_model(None), loader)
    assert np.isfinite(loss) and metrics


def test_config_only_task_trains_and_predicts(tmp_path, monkeypatch):
    """A folder with only config.yaml is a complete task (ready-made base)."""
    from torch.utils.data import DataLoader, TensorDataset

    from audio_ops_dagster import registry
    from audio_ops_dagster.core.model_trainer import ModelTrainer
    from audio_ops_dagster.core.runtime import load_task_model

    task_dir = create_task("cfg_only", tasks_dir=tmp_path, config_only=True,
                           classes=["x", "y", "z"])
    assert [p.name for p in task_dir.iterdir()] == ["config.yaml"]
    monkeypatch.setenv("AUDIO_OPS_TASKS_DIR", str(tmp_path))
    registry.discover_tasks()
    found = next(t for t in registry.iter_task_dirs() if t.name == "cfg_only")
    assert registry.check_task(found)[0] == []
    cfg = TaskConfig.from_dict({**found.raw_config, "epochs": 1}, task_name="cfg_only")
    task = get_task("cfg_only", cfg)
    loader = DataLoader(TensorDataset(torch.randn(8, 1, 128, 32), torch.randint(0, 3, (8,))),
                        batch_size=4)
    ckpt = ModelTrainer(task, checkpoint_dir=str(tmp_path / "c")).train(
        task.build_model(None), loader, loader)
    # a config without labels adopts the checkpoint's class names
    bare = get_task("cfg_only", TaskConfig.from_dict({}, task_name="cfg_only"))
    model, meta = load_task_model(bare, ckpt)
    assert bare.config.class_labels == ["x", "y", "z"] and meta["num_classes"] == 3
    out = bare.postprocess(model(torch.randn(1, 1, 128, 32)))
    assert out["label"] in {"x", "y", "z"} and set(out["scores"]) == {"x", "y", "z"}


def test_bad_names_rejected(tmp_path):
    with pytest.raises(ValueError):
        create_task("Bad-Name", tasks_dir=tmp_path)
    create_task("ok_name", tasks_dir=tmp_path)
    with pytest.raises(FileExistsError):
        create_task("ok_name", tasks_dir=tmp_path)
