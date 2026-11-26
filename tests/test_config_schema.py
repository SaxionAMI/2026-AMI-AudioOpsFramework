import pytest

pytest.importorskip("pydantic")
from audio_ops_dagster.core.task_config_schema import validate_task_config  # noqa: E402


def test_valid_config():
    errors, _ = validate_task_config({
        "archetype": "trainable_model", "task_name": "t",
        "feature_type": "spectrogram", "num_classes": 2,
        "class_labels": ["a", "b"], "input_shape": [1, 128, 32],
    })
    assert errors == []


def test_label_count_mismatch_errors():
    errors, _ = validate_task_config({"num_classes": 3, "class_labels": ["a", "b"]})
    assert any("num_classes" in e for e in errors)


def test_bad_capability_errors():
    errors, _ = validate_task_config({"capabilities": ["trian"], "task_name": "t"})
    assert errors


def test_unknown_feature_warns_not_errors():
    errors, warnings = validate_task_config({"feature_type": "nope", "archetype": "trainable_model"})
    assert errors == []
    assert any("feature_type" in w for w in warnings)


def test_typo_suggestion():
    _, warnings = validate_task_config({"learnig_rate": 0.1})
    assert any("did you mean 'learning_rate'" in w for w in warnings)


@pytest.mark.parametrize("cfg", [
    {"split_strategy": "stratified"},
    {"optimizer": "adamax"},
    {"early_stop_metric": "val_f1"},
    {"segment_overlap": 1.0},
    {"epochs": 0},
    {"task_type": "segmentation"},
    {"split_ratios": [0.5, 0.5, 0.5]},
    {"class_labels": ["a", "a"]},
    {"num_classes": 2, "class_weights": [1.0]},
])
def test_invalid_values_error(cfg):
    errors, _ = validate_task_config(cfg)
    assert errors, cfg


def test_non_mapping():
    errors, _ = validate_task_config(["not", "a", "dict"])
    assert errors
