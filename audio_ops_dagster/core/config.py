"""
config.py — ``TaskConfig``: the one definition of a task's ``config.yaml``.

Every key the framework reads is declared here exactly once, with its type,
default and constraints. The Dagster assets, the trainer, the inference
service, the CLI and ``audio-ops validate`` all use this model, so there is a
single source of truth for what a config means.

Unknown keys are kept (``extra="allow"``) and readable as attributes, so
task-specific settings work too::

    cfg = TaskConfig.from_dict({"class_labels": ["a", "b"], "threshold": 0.7}, task_name="t")
    cfg.num_classes   # 2 — derived from class_labels
    cfg.threshold     # 0.7 — your own key
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TaskType = Literal[
    "classification", "regression", "detection", "transcription", "nonverbal_pretrained"
]
Criterion = Literal["cross_entropy", "mse", "l1", "mae", "bce", "bce_with_logits", "ctc"]


class TaskConfig(BaseModel):
    """A task's configuration, with defaults. See the docs' task-config reference."""

    model_config = ConfigDict(extra="allow", protected_namespaces=(), validate_assignment=False)

    # ── Identity & what the task does ────────────────────────────────────────
    task_name: str
    archetype: Optional[str] = None
    capabilities: Optional[List[str]] = None
    task_type: TaskType = "classification"
    serve_wrapper: Optional[str] = None

    # ── Data & features ──────────────────────────────────────────────────────
    input_dir: str = ""
    output_dir: str = ""
    feature_type: str = "spectrogram"
    sample_rate: int = Field(16000, gt=0)
    segment_duration: float = Field(1.0, gt=0)
    segment_overlap: float = Field(0.25, ge=0, lt=1)
    force_recompute: bool = False
    input_shape: Optional[List[int]] = None
    class_labels: Optional[List[str]] = None
    num_classes: Optional[int] = Field(None, ge=1)

    # ── Splits ───────────────────────────────────────────────────────────────
    split_strategy: Literal["group", "random"] = "random"
    split_ratios: List[float] = Field(default_factory=lambda: [0.7, 0.15, 0.15])
    group_pattern: Optional[str] = None
    seed: int = 42

    # ── Training ─────────────────────────────────────────────────────────────
    batch_size: int = Field(32, ge=1)
    epochs: int = Field(10, ge=1)
    learning_rate: float = Field(1e-3, gt=0)
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    weight_decay: float = Field(0.0, ge=0)
    criterion: Optional[Criterion] = None      # None → by task_type (CE / MSE / BCE)
    class_weights: Optional[List[float]] = None
    label_smoothing: float = Field(0.0, ge=0, lt=1)
    warmup_epochs: int = Field(0, ge=0)
    lr_min: float = Field(1e-6, ge=0)
    grad_clip: Optional[float] = Field(1.0, gt=0)
    grad_accumulation_steps: int = Field(1, ge=1)
    early_stop_metric: Literal["val_loss", "val_acc"] = "val_loss"
    early_stop_patience: int = Field(10, ge=1)
    mixed_precision: bool = True
    num_workers: int = Field(0, ge=0)
    resume: bool = True
    log_every_n_batches: int = Field(50, ge=1)
    model_architecture: str = "cnn"

    # ── Inference / serving ──────────────────────────────────────────────────
    checkpoint_path: Optional[str] = None
    transcription_chunk_seconds: float = Field(30.0, gt=0)
    stream_buffer_seconds: Optional[float] = Field(None, gt=0)
    silence_rms: Optional[float] = Field(None, ge=0)
    model_name: Optional[str] = None

    # ── Storage ──────────────────────────────────────────────────────────────
    use_remote_storage: bool = False
    storage_bucket_prefix: str = ""

    # ── Validation ───────────────────────────────────────────────────────────

    @model_validator(mode="before")
    @classmethod
    def _defaults_and_legacy(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        name = data.get("task_name") or "task"
        data["task_name"] = name
        if not data.get("input_dir"):
            data["input_dir"] = f"audio_input/{name}"
        if not data.get("output_dir"):
            data["output_dir"] = f"storage/{name}"
        if "use_amp" in data and "mixed_precision" not in data:      # legacy key
            data["mixed_precision"] = data.pop("use_amp")
        if data.get("task_type") is None:
            data.pop("task_type", None)
        return data

    @field_validator("class_labels", mode="before")
    @classmethod
    def _labels_are_strings(cls, v: Any) -> Any:
        if v is None:
            return v
        if not isinstance(v, (list, tuple)):
            raise ValueError("must be a list of label names")
        if any(isinstance(x, bool) for x in v):
            raise ValueError(
                'contains booleans — YAML reads yes/no/on/off/true/false as booleans; '
                'quote them, e.g. - "yes"')
        return [str(x) for x in v]

    @field_validator("group_pattern")
    @classmethod
    def _regex(cls, v):
        if v is not None:
            try:
                re.compile(v)
            except re.error as exc:
                raise ValueError(f"not a valid regular expression: {exc}") from None
        return v

    @model_validator(mode="after")
    def _consistency(self) -> TaskConfig:
        r = self.split_ratios
        if len(r) != 3 or any(x <= 0 for x in r) or abs(sum(r) - 1.0) > 1e-6:
            raise ValueError(f"split_ratios must be three positive numbers summing to 1 (got {r})")
        if self.class_labels is not None:
            if len(set(self.class_labels)) != len(self.class_labels):
                raise ValueError("class_labels contains duplicates")
            if self.num_classes is None:
                self.num_classes = len(self.class_labels)
            elif self.num_classes != len(self.class_labels) and self.task_type in (
                    "classification", "detection"):
                raise ValueError(f"num_classes ({self.num_classes}) != len(class_labels) "
                                 f"({len(self.class_labels)}) — you can drop num_classes, "
                                 f"it is derived from class_labels")
        if self.class_weights is not None and self.num_classes is not None \
                and len(self.class_weights) != self.num_classes:
            raise ValueError(f"class_weights has {len(self.class_weights)} entries, "
                             f"expected {self.num_classes}")
        return self

    # ── Helpers ──────────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]], task_name: Optional[str] = None
                  ) -> TaskConfig:
        """Validate a raw dict (e.g. a parsed config.yaml). Raises ``ValueError``."""
        data = dict(data or {})
        if task_name:
            data["task_name"] = task_name
        return cls.model_validate(data)

    @classmethod
    def field_names(cls) -> List[str]:
        return list(cls.model_fields)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def effective_bucket_prefix(self) -> str:
        return self.storage_bucket_prefix or self.task_name
