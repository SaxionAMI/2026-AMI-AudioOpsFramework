"""
Task Registry — auto-discovery and registration of audio processing tasks.

Discovery sources (both scanned):
  1. The framework's built-in tasks package: audio_ops_dagster/tasks/
  2. External task directories from the AUDIO_OPS_TASKS_DIR env var
     (os.pathsep- or comma-separated). This is how a *project repo* keeps its
     own tasks outside the installed framework package:

         AUDIO_OPS_TASKS_DIR=./tasks dagster dev -m audio_ops_dagster.definitions
         AUDIO_OPS_TASKS_DIR=./tasks audio-ops-serve

Each task is a directory containing `task_<name>.py` (or `task.py`) with a
class decorated by @register_task, plus a `config.yaml`.
"""

import importlib
import importlib.util
import os
import pkgutil
import sys
from pathlib import Path
from typing import Dict, Type, Optional, List
from dataclasses import dataclass, field


@dataclass
class TaskConfig:
    """Configuration for any audio task."""

    task_name: str
    input_dir: str
    output_dir: str
    feature_type: str

    num_classes: Optional[int] = None
    class_labels: Optional[List[str]] = None

    split_strategy: str = "random"

    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"
    criterion: str = "cross_entropy"
    model_architecture: str = "cnn"

    # ── Feature/segmentation knobs ───────────────────────────────────────────
    # These MUST live here (not only in the validation schema) or they never
    # reach AudioProcessor / the trainer: to_task_config() and the resource-field
    # filter both drop any key that isn't a declared field. Previously, omitting
    # them meant `segment_duration: 10.0` in config.yaml was silently ignored and
    # AST was fed 1-second windows instead of its ~10s pretraining window.
    sample_rate: int = 16000
    segment_duration: float = 1.0
    segment_overlap: float = 0.25
    force_recompute: bool = False

    # ── Training knobs (read directly by ModelTrainer) ───────────────────────
    weight_decay: float = 0.0
    warmup_epochs: int = 0
    grad_clip: Optional[float] = None
    early_stop_patience: int = 10
    early_stop_metric: str = "val_loss"
    class_weights: Optional[List[float]] = None

    # HuggingFace model identifier (for STT, etc.)
    model_name: Optional[str] = None

    # Storage configuration
    use_remote_storage: bool = False
    storage_bucket_prefix: str = ""

    def effective_bucket_prefix(self) -> str:
        return self.storage_bucket_prefix or self.task_name


_TASKS: Dict[str, Type] = {}


def register_task(name: str, task_class: Optional[Type] = None):
    """
    Register a task. Usable as decorator or direct call.

    Usage:
      @register_task("my_task")
      class MyTask(AudioTask): ...

      OR

      register_task("my_task", MyTaskClass)
    """
    def wrapper(cls: Type):
        _TASKS[name] = cls
        print(f"✓ Registered task: {name}")
        return cls

    if task_class is not None:
        return wrapper(task_class)
    return wrapper


def get_task(name: str, config: TaskConfig):
    """Get task instance by name."""
    if name not in _TASKS:
        available = list(_TASKS.keys())
        raise ValueError(f"Unknown task '{name}'. Available: {available}")
    return _TASKS[name](config)


def list_tasks() -> List[str]:
    """Return names of all registered tasks."""
    return list(_TASKS.keys())


# ── Discovery ────────────────────────────────────────────────────────────────

def external_task_dirs() -> List[Path]:
    """External task directories declared via AUDIO_OPS_TASKS_DIR."""
    raw = os.environ.get("AUDIO_OPS_TASKS_DIR", "")
    dirs: List[Path] = []
    for part in raw.replace(",", os.pathsep).split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        p = Path(part).expanduser()
        if p.is_dir():
            dirs.append(p)
        else:
            print(f"⚠ AUDIO_OPS_TASKS_DIR entry not found: {part}")
    return dirs


def _builtin_tasks_dir() -> Optional[Path]:
    try:
        import audio_ops_dagster.tasks as tasks_pkg
        return Path(tasks_pkg.__path__[0])
    except ImportError:
        return None


def task_search_dirs() -> List[Path]:
    """All directories to scan for tasks: built-in package + external repos."""
    dirs: List[Path] = []
    builtin = _builtin_tasks_dir()
    if builtin is not None:
        dirs.append(builtin)
    dirs.extend(external_task_dirs())
    # dedup preserving order
    seen, out = set(), []
    for d in dirs:
        r = d.resolve()
        if r not in seen:
            seen.add(r)
            out.append(d)
    return out


def _load_external_task(task_dir: Path) -> None:
    """Import a task module from an arbitrary directory (by file path)."""
    name = task_dir.name
    for file_name in (f"task_{name}.py", "task.py"):
        f = task_dir / file_name
        if not f.exists():
            continue
        mod_id = f"audio_ops_external.{name}.{file_name[:-3]}"
        if mod_id in sys.modules:
            return
        try:
            spec = importlib.util.spec_from_file_location(mod_id, f)
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_id] = module
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"❌ Error loading {f}: {e}")
        return


def discover_tasks():
    """Import all task modules so their @register_task decorators run."""
    # 1. Built-in package tasks (importable as audio_ops_dagster.tasks.<name>)
    builtin = _builtin_tasks_dir()
    if builtin is not None:
        for _, module_name, is_pkg in pkgutil.iter_modules([str(builtin)]):
            if not is_pkg:
                continue
            task_dir = builtin / module_name
            for file_name in (f"task_{module_name}.py", "task.py"):
                if (task_dir / file_name).exists():
                    module_path = f"audio_ops_dagster.tasks.{module_name}.{file_name[:-3]}"
                    try:
                        if module_path not in sys.modules:
                            importlib.import_module(module_path)
                    except Exception as e:
                        print(f"❌ Error loading {module_path}: {e}")
                    break

    # 2. External task directories from a project repo (loaded by file path)
    for ext in external_task_dirs():
        for sub in sorted(ext.iterdir()):
            if sub.is_dir() and not sub.name.startswith((".", "__")):
                _load_external_task(sub)


# Run auto-discovery on import
discover_tasks()
