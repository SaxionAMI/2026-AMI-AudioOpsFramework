import pytest

from audio_ops_dagster.cli import main


def test_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip()


def test_new_task_then_validate(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    monkeypatch.setenv("AUDIO_OPS_TASKS_DIR", str(tmp_path))
    assert main(["new-task", "cli_task", "--classes", "yes,no"]) == 0
    assert (tmp_path / "cli_task" / "config.yaml").exists()
    assert "- \"yes\"" in (tmp_path / "cli_task" / "config.yaml").read_text()
    assert main(["validate", "cli_task"]) == 0
    assert main(["validate", "does_not_exist"]) == 1
    assert main(["list"]) == 0
    assert "cli_task" in capsys.readouterr().out


def test_validate_reports_errors(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    d = tmp_path / "broken"
    d.mkdir()
    (d / "config.yaml").write_text("archetype: not_a_thing\n")
    monkeypatch.setenv("AUDIO_OPS_TASKS_DIR", str(tmp_path))
    assert main(["validate", "broken"]) == 1


def test_demo_data(tmp_path):
    assert main(["demo-data", "--out", str(tmp_path / "d"), "--files", "2"]) == 0
    assert len(list((tmp_path / "d").rglob("*.wav"))) == 6


def test_doctor(capsys):
    assert main(["doctor"]) == 0
    assert "audio-ops-framework" in capsys.readouterr().out


def test_tasks_folder_is_found_without_env(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUDIO_OPS_TASKS_DIR", raising=False)
    assert main(["new-task", "local_task", "--config-only"]) == 0
    assert (tmp_path / "tasks" / "local_task" / "config.yaml").exists()
    assert main(["validate", "local_task"]) == 0
    out = capsys.readouterr().out
    assert "not searched" not in out


@pytest.mark.integration
def test_run_and_predict(demo_project, capsys):
    pytest.importorskip("dagster")
    assert main(["run", "example_classifier"]) == 0
    out = capsys.readouterr().out
    assert "succeeded" in out and "[example_classifier_trained_model] Epoch" in out
    wav = demo_project / "audio_input" / "example_classifier" / "class_b" / "class_b_00.wav"
    assert main(["predict", "example_classifier", str(wav)]) == 0
    assert "class_b" in capsys.readouterr().out
    assert main(["run", "example_classifier", "--job", "evaluate"]) == 1   # not an eval-only task
    assert main(["run", "no_such_task"]) == 1


def test_facade_exports():
    import audio_ops

    for name in audio_ops.__all__:
        assert getattr(audio_ops, name) is not None


def test_dev_uses_dagster_from_the_same_environment(tmp_path, monkeypatch):
    """`.venv/bin/audio-ops dev` must work without activating the venv."""
    import subprocess
    import sys
    from pathlib import Path

    pytest.importorskip("dagster")
    calls = []
    monkeypatch.setattr(subprocess, "call", lambda cmd, env=None: calls.append(cmd) or 0)
    monkeypatch.setenv("PATH", str(tmp_path))          # no dagster on PATH
    monkeypatch.setenv("DAGSTER_HOME", str(tmp_path / "dh"))
    assert main(["dev"]) == 0
    assert Path(calls[0][0]).parent == Path(sys.executable).parent
