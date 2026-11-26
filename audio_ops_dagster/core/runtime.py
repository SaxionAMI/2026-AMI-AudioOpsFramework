"""
runtime.py — small helpers shared by training, batch inference and serving.

Keeping these in one place is what guarantees train/serve parity: the Dagster
assets, the trainer's dataset and the inference API all shape model inputs,
pick a device, name segments and load checkpoints through the same functions.
torch is imported lazily so this module stays importable without it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from audio_ops_dagster._log import get_logger

logger = get_logger(__name__)

#: File extensions treated as audio everywhere in the framework.
AUDIO_EXTENSIONS: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

_SEG_RE = re.compile(r"^(?P<stem>.+?)_(?:seg|chunk)_?(?P<idx>\d+)$")


def is_audio_file(path: os.PathLike | str) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXTENSIONS


def list_audio_files(root: os.PathLike | str) -> list[Path]:
    """All audio files under ``root`` (recursive), sorted for determinism."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and is_audio_file(p))


def segment_name(stem: str, index: int) -> str:
    """Canonical feature-file stem for segment ``index`` of recording ``stem``."""
    return f"{stem}_seg_{index:03d}"


def parse_segment_name(stem: str) -> Tuple[str, int]:
    """Inverse of :func:`segment_name`. Returns ``(recording_stem, index)``.

    Only a trailing ``_seg_NNN`` / ``_chunk_NNN`` suffix is stripped, so a
    recording called ``my_segment_a`` is not truncated to ``my``. Names without
    a suffix are returned unchanged with index 0.
    """
    m = _SEG_RE.match(stem)
    if not m:
        return stem, 0
    return m.group("stem"), int(m.group("idx"))


def resolve_device(preference: Optional[str] = None):
    """Pick a torch device.

    ``preference`` (or the ``DEVICE`` env var) may be ``auto`` (default),
    ``cpu``, ``cuda``, ``cuda:N`` or ``mps``. An unavailable accelerator falls
    back to CPU with a warning instead of crashing.
    """
    import torch

    pref = (preference or os.environ.get("DEVICE") or "auto").strip().lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("DEVICE=%s requested but CUDA is unavailable — using CPU.", pref)
        return torch.device("cpu")
    if pref == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            logger.warning("DEVICE=mps requested but MPS is unavailable — using CPU.")
            return torch.device("cpu")
    return torch.device(pref)


def add_channel_dim(feature: np.ndarray) -> np.ndarray:
    """(F, T) → (1, F, T). Other ranks are returned unchanged.

    This is the single rule the training dataset applies to every stored
    feature, so serving and batch inference must apply it too.
    """
    feature = np.asarray(feature)
    if feature.ndim == 2:
        return feature[np.newaxis, ...]
    return feature


def to_model_input(feature: Any, device=None):
    """Shape one feature exactly like a training sample, then add a batch dim.

    Training samples are ``add_channel_dim(np.load(path))`` and the DataLoader
    adds the batch dimension. Doing the same here means a raw-waveform model
    sees ``(1, T)`` and a spectrogram CNN sees ``(1, 1, F, T)`` — the same ranks
    it was trained on.
    """
    import torch

    if isinstance(feature, torch.Tensor):
        t = feature.detach().float()
        if t.dim() == 2:
            t = t.unsqueeze(0)
    else:
        t = torch.from_numpy(np.ascontiguousarray(add_channel_dim(feature), dtype=np.float32))
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(0)
    return t.to(device) if device is not None else t


# ── Checkpoints ──────────────────────────────────────────────────────────────


class CheckpointError(RuntimeError):
    """A checkpoint could not be read or does not fit the model."""


def _allow_unsafe() -> bool:
    return os.environ.get("ALLOW_UNSAFE_CHECKPOINTS", "false").lower() in ("1", "true", "yes")


def read_checkpoint(path: os.PathLike | str, map_location: Any = "cpu") -> Any:
    """Load a checkpoint file safely.

    Framework checkpoints (``best_model.pt``) contain only tensors and plain
    metadata, so they load with ``weights_only=True`` — no arbitrary code runs.
    Pickled full-model files are refused unless ``ALLOW_UNSAFE_CHECKPOINTS=true``
    is set, because unpickling executes code from the file.
    """
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:  # noqa: BLE001 — torch raises several types here
        if not _allow_unsafe():
            raise CheckpointError(
                f"Could not load {path} in safe (weights-only) mode: {exc}. If this is a "
                f"trusted pickled checkpoint, set ALLOW_UNSAFE_CHECKPOINTS=true."
            ) from exc
        logger.warning("Loading %s with weights_only=False (ALLOW_UNSAFE_CHECKPOINTS).", path)
        return torch.load(path, map_location=map_location, weights_only=False)


def state_dict_from(checkpoint: Any) -> Dict[str, Any]:
    """Extract a state dict from any supported checkpoint layout."""
    import torch.nn as nn

    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint
    raise CheckpointError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")


def load_checkpoint_into(model, path: os.PathLike | str, map_location: Any = "cpu") -> Dict[str, Any]:
    """Load weights from ``path`` into ``model`` (strict). Returns checkpoint metadata.

    Raises :class:`CheckpointError` with an actionable message on a mismatch,
    which almost always means the checkpoint came from a different
    architecture or ``num_classes``.
    """
    ckpt = read_checkpoint(path, map_location=map_location)
    state = state_dict_from(ckpt)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise CheckpointError(
            f"Checkpoint {path} does not match the model built by this task "
            f"(different architecture, input shape or num_classes?). Delete it and "
            f"retrain, or point checkpoint_path at a matching file. Details: {str(exc)[:300]}"
        ) from None
    if isinstance(ckpt, dict):
        return {k: v for k, v in ckpt.items() if k not in ("model_state_dict", "state_dict")}
    return {}


def load_task_model(task, path: Optional[os.PathLike | str] = None, *, device=None,
                    required: bool = True):
    """Build ``task``'s model and load its checkpoint. Returns ``(model, metadata)``.

    If the config has no ``class_labels``, the class names stored in the
    checkpoint are adopted first, so the network is built with the right size.
    With ``required=False`` a missing checkpoint returns an untrained model.
    """
    cfg = task.config
    path = str(path or default_checkpoint_path(cfg))
    meta: Dict[str, Any] = {}
    if os.path.exists(path):
        ckpt = read_checkpoint(path)
        if isinstance(ckpt, dict):
            meta = {k: v for k, v in ckpt.items() if k not in ("model_state_dict", "state_dict")}
        if not getattr(cfg, "class_labels", None) and meta.get("class_names"):
            cfg.class_labels = list(meta["class_names"])
            cfg.num_classes = len(cfg.class_labels)
        model = task.build_model(getattr(cfg, "input_shape", None))
        try:
            model.load_state_dict(state_dict_from(ckpt), strict=True)
        except RuntimeError as exc:
            raise CheckpointError(
                f"Checkpoint {path} does not match the model built by this task "
                f"(different architecture, input shape or number of classes?). Retrain, or "
                f"point checkpoint_path at a matching file. Details: {str(exc)[:300]}"
            ) from None
    elif required:
        raise CheckpointError(f"No checkpoint at {path}. Train the task first, or set "
                              f"checkpoint_path in its config.yaml.")
    else:
        model = task.build_model(getattr(cfg, "input_shape", None))
    device = device if device is not None else resolve_device()
    return model.to(device).eval(), meta


def default_checkpoint_path(cfg: Any) -> str:
    """Where a task's trained checkpoint lives unless configured otherwise.

    Priority: ``checkpoint_path`` in the task config → ``INFERENCE_CHECKPOINT``
    env var → ``<output_dir>/checkpoints/best_model.pt``.
    """
    explicit = getattr(cfg, "checkpoint_path", None)
    if explicit:
        return str(explicit)
    env = os.environ.get("INFERENCE_CHECKPOINT")
    if env:
        return env
    return os.path.join(str(getattr(cfg, "output_dir", ".")), "checkpoints", "best_model.pt")
