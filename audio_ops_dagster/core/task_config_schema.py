"""
task_config_schema.py — validation for a task's config.yaml.

Goal: when someone starts a new task, mistakes surface immediately with a clear
message instead of failing deep inside a Dagster run. Validation is intentionally
lenient about *extra* keys (backends carry their own settings) but strict about
the fields the framework relies on.

Use:
    from audio_ops_dagster.core.task_config_schema import validate_task_config
    errors, warnings = validate_task_config(cfg_dict)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from audio_ops_dagster.capabilities import resolve_capabilities, CapabilityError
from audio_ops_dagster.core.features import has_feature, list_features


class TaskConfigSchema(BaseModel):
    """The framework-relevant subset of a task config. Extra keys are allowed."""

    model_config = ConfigDict(extra="allow")

    task_name: Optional[str] = None
    archetype: Optional[str] = None
    capabilities: Optional[List[str]] = None

    feature_type: str = "spectrogram"
    input_shape: Optional[List[int]] = None

    num_classes: Optional[int] = None
    class_labels: Optional[List[str]] = None

    split_strategy: str = "random"
    task_type: Optional[str] = None
    serve_wrapper: Optional[str] = None

    use_remote_storage: bool = False


def validate_task_config(cfg: dict) -> Tuple[List[str], List[str]]:
    """
    Return (errors, warnings). Errors should block the task from building;
    warnings are advisory. Never raises.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Shape / type validation via pydantic.
    try:
        model = TaskConfigSchema(**cfg)
    except Exception as e:  # pydantic ValidationError or otherwise
        return [f"config does not parse: {e}"], warnings

    # 2. Capabilities must resolve (explicit list / archetype / legacy flag).
    try:
        caps = resolve_capabilities(cfg)
    except CapabilityError as e:
        errors.append(str(e))
        caps = frozenset()

    # 3. Feature type should be registered (warn — a task module may register a
    #    custom one at import time, after this check runs).
    if not has_feature(model.feature_type):
        warnings.append(
            f"feature_type '{model.feature_type}' is not registered yet "
            f"(built-ins: {list_features()}). If you register it in your task "
            f"module with @register_feature, ignore this."
        )

    # 4. Classification consistency.
    is_classification = (model.task_type or "classification") == "classification"
    if is_classification and model.num_classes is not None and model.class_labels:
        if len(model.class_labels) != model.num_classes:
            errors.append(
                f"num_classes ({model.num_classes}) != len(class_labels) "
                f"({len(model.class_labels)})."
            )

    # 5. Trainable classification really wants an input_shape.
    if "train" in caps and is_classification and not model.input_shape:
        warnings.append("no input_shape given; the model must infer it or be shape-agnostic.")

    return errors, warnings
