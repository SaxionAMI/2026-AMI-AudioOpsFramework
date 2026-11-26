"""End-to-end Dagster pipeline tests on synthetic audio (CPU, a few seconds)."""

import json
import textwrap

import pytest

pytest.importorskip("dagster")
pytest.importorskip("torch")

from _helpers import run_job, train_example  # noqa: E402

pytestmark = pytest.mark.integration


def test_full_training_job_produces_artifacts(demo_project):
    defs, _ = train_example(demo_project)
    out = demo_project / "storage" / "example_classifier"
    assert (out / "checkpoints" / "best_model.pt").exists()
    assert (out / "dataset" / "train.csv").exists()
    report = json.loads((out / "outputs" / "evaluation_example_classifier.json").read_text())
    assert report["accuracy"] >= 0.6          # synthetic tones are easy
    results = json.loads((out / "outputs" / "inference_results_example_classifier.json")
                         .read_text())
    first = next(iter(results.values()))
    assert {"label", "confidence", "start_sec", "end_sec", "source"} <= set(first)
    assert (out / "outputs" / "inference_results_example_classifier.csv").exists()

    import torch
    ckpt = torch.load(out / "checkpoints" / "best_model.pt", weights_only=True)
    assert ckpt["feature_config"]["feature_type"] == "spectrogram"
    assert ckpt["class_names"] == ["class_a", "class_b", "class_c"]


def test_inference_job_with_launchpad_override(demo_project):
    from audio_ops_dagster.demo import write_dataset

    train_example(demo_project)
    write_dataset(demo_project / "new_batch", files_per_class=1, seconds=1.5, seed=9)
    run_job("example_classifier_inference_job", run_config={"ops": {
        "example_classifier_raw_audio_files": {"config": {
            "input_dir": str(demo_project / "new_batch"),
            "output_dir": str(demo_project / "new_batch_out")}}}})
    results = json.loads((demo_project / "new_batch_out" / "outputs" /
                          "inference_results_example_classifier.json").read_text())
    assert len(results) >= 3
    # the batch's features went to the override dir, not the task's storage
    assert (demo_project / "new_batch_out" / "features").is_dir()


def _write_variant(root, name, archetype, extra=""):
    d = root / name
    d.mkdir(parents=True)
    (d / f"task_{name}.py").write_text(textwrap.dedent(f'''
        from audio_ops_dagster.registry import register_task
        from audio_ops_dagster.tasks.example_classifier.task_example_classifier import (
            ExampleClassifierTask,
        )

        @register_task("{name}")
        class Variant(ExampleClassifierTask):
            pass
    '''))
    (d / "config.yaml").write_text(textwrap.dedent(f'''
        archetype: {archetype}
        input_dir: audio_input/example_classifier
        output_dir: storage/{name}
        num_classes: 3
        class_labels: [class_a, class_b, class_c]
        split_strategy: group
        {extra}
    '''))


def test_eval_only_and_pretrained_batch_use_existing_checkpoint(demo_project, monkeypatch):
    train_example(demo_project)
    ckpt = demo_project / "storage" / "example_classifier" / "checkpoints" / "best_model.pt"
    tasks = demo_project / "tasks"
    _write_variant(tasks, "eval_variant", "eval_only", f"checkpoint_path: {ckpt}")
    _write_variant(tasks, "batch_variant", "pretrained_batch", f"checkpoint_path: {ckpt}")
    monkeypatch.setenv("AUDIO_OPS_TASKS_DIR", str(tasks))

    defs, _ = run_job("eval_variant_evaluation_job")
    job_names = {j.name for j in defs.resolve_all_job_defs()}
    assert "eval_variant_full_training_job" not in job_names     # eval_only never trains
    assert not (demo_project / "storage" / "eval_variant" / "checkpoints").exists()
    report = json.loads((demo_project / "storage" / "eval_variant" / "outputs" /
                         "evaluation_eval_variant.json").read_text())
    assert report["checkpoint"] == str(ckpt)

    run_job("batch_variant_inference_job")
    assert (demo_project / "storage" / "batch_variant" / "outputs" /
            "inference_results_batch_variant.json").exists()


def test_broken_task_is_skipped_not_fatal(demo_project, monkeypatch):
    bad = demo_project / "tasks" / "broken"
    bad.mkdir(parents=True)
    (bad / "config.yaml").write_text("archetype: nope\n")
    monkeypatch.setenv("AUDIO_OPS_TASKS_DIR", str(demo_project / "tasks"))
    from audio_ops_dagster.definitions import build_definitions

    names = {j.name for j in build_definitions().resolve_all_job_defs()}
    assert "example_classifier_full_training_job" in names
    assert not any(n.startswith("broken_") for n in names)
