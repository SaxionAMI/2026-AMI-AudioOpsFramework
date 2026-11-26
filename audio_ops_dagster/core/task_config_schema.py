"""
task_config_schema.py — validation for a task's config.yaml.

The schema itself is :class:`audio_ops_dagster.core.config.TaskConfig`; this
module turns its errors into readable messages and adds the checks that need
more context (capabilities, registered features, likely typos, training
advice). Never raises.

    from audio_ops_dagster.core.task_config_schema import validate_task_config
    errors, warnings = validate_task_config(cfg_dict)

or from the shell: ``audio-ops validate``.
"""

from __future__ import annotations

import difflib
from typing import List, Tuple

from pydantic import ValidationError

from audio_ops_dagster.capabilities import TRAIN, CapabilityError, resolve_capabilities
from audio_ops_dagster.core.config import TaskConfig
from audio_ops_dagster.core.features import has_feature, list_features

#: Backwards-compatible name.
TaskConfigSchema = TaskConfig

#: Keys that are legitimately used but not fields of TaskConfig.
_KNOWN_EXTRA = {"dagster", "use_amp"}


def _format_validation_error(exc: ValidationError) -> List[str]:
    out = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "config"
        msg = str(err.get("msg", "invalid value")).removeprefix("Value error, ")
        if err.get("type") in ("value_error", "assertion_error") and loc == "config":
            out.append(msg)
        else:
            out.append(f"{loc}: {msg} (got {err.get('input')!r})")
    return out


def validate_task_config(cfg: dict) -> Tuple[List[str], List[str]]:
    """Return ``(errors, warnings)``. Errors block the task; warnings are advisory."""
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(cfg, dict):
        return ["config.yaml must contain a mapping of keys to values"], warnings

    try:
        model = TaskConfig.from_dict(cfg)
    except ValidationError as e:
        return _format_validation_error(e), warnings

    known = set(TaskConfig.model_fields)
    for key in cfg:
        if key in known or key in _KNOWN_EXTRA:
            continue
        match = difflib.get_close_matches(key, known, n=1, cutoff=0.8)
        if match:
            warnings.append(f"unknown key '{key}' — did you mean '{match[0]}'?")

    try:
        caps = resolve_capabilities(cfg)
    except CapabilityError as e:
        errors.append(str(e))
        caps = frozenset()

    if not has_feature(model.feature_type):
        warnings.append(
            f"feature_type '{model.feature_type}' is not registered yet "
            f"(built-ins: {list_features()}). If your task module registers it with "
            f"@register_feature, ignore this."
        )

    labelled = model.task_type in ("classification", "detection")
    if TRAIN in caps and labelled and not model.class_labels:
        warnings.append(
            "no class_labels given; labels will be inferred from the folder names at "
            "training time. Listing them fixes the class order (= output index)."
        )
    if TRAIN in caps and model.task_type == "transcription":
        warnings.append(
            "transcription tasks are usually serve-only (archetype: pretrained_servable); "
            "the built-in trainer expects class-index or numeric targets."
        )
    return errors, warnings
