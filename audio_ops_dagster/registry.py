"""
Task Registry — auto-discovery and registration of audio processing tasks.

v2: Added model_name to TaskConfig for HuggingFace model support.
"""

import importlib
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


def discover_tasks():
    """Discover tasks from the tasks package subdirectories."""
    try:
        import audio_ops_dagster.tasks as tasks_pkg
    except ImportError:
        return

    package_path = tasks_pkg.__path__[0]

    for _, module_name, is_pkg in pkgutil.iter_modules([package_path]):
        if is_pkg:
            task_dir = Path(package_path) / module_name
            possible_files = [f"task_{module_name}.py", "task.py"]

            for file_name in possible_files:
                if (task_dir / file_name).exists():
                    module_path = f"audio_ops_dagster.tasks.{module_name}.{file_name[:-3]}"
                    try:
                        if module_path not in sys.modules:
                            importlib.import_module(module_path)
                        break
                    except Exception as e:
                        print(f"❌ Error loading {module_path}: {e}")


# Run auto-discovery on import
discover_tasks()
