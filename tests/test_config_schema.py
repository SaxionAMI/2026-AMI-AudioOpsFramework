import pytest
pytest.importorskip("pydantic")
from audio_ops_dagster.core.task_config_schema import validate_task_config


def test_valid_config():
    errors, _ = validate_task_config({
        "archetype": "trainable_model", "task_name": "t",
        "feature_type": "spectrogram", "num_classes": 2,
        "class_labels": ["a", "b"], "input_shape": [1, 128, 87],
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
