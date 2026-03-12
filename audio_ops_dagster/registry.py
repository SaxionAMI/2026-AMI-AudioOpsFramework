import importlib
import pkgutil
import inspect
import sys
from pathlib import Path
from typing import Dict, Type, Optional, List
from dataclasses import dataclass

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

_TASKS: Dict[str, Type] = {}

def register_task(name: str, task_class: Optional[Type] = None):
    """
    Registers a task. Can be used as a decorator or a direct function call.
    Usage:
      @register_task("my_task")
      class MyTask...
      
      OR
      register_task("my_task", MyTaskClass)
    """
    def wrapper(cls: Type):
        _TASKS[name] = cls
        print(f"✓ Registered task: {name}")
        return cls

    # If task_class is provided, we are calling it as a function
    if task_class is not None:
        return wrapper(task_class)
    
    # Otherwise, return the wrapper for decorator usage (@register_task("name"))
    return wrapper

def get_task(name: str, config: TaskConfig):
    """Get task instance by name."""
    if name not in _TASKS:
        available = list(_TASKS.keys())
        raise ValueError(f"Unknown task '{name}'. Available: {available}")
    return _TASKS[name](config)

def discover_tasks():
    """Discover tasks without screaming if a folder is empty or mid-development."""
    import audio_ops_dagster.tasks as tasks_pkg
    
    # Get the actual file system path of the tasks package
    package_path = tasks_pkg.__path__[0]
    
    for _, module_name, is_pkg in pkgutil.iter_modules([package_path]):
        if is_pkg:
            task_dir = Path(package_path) / module_name
            # Check for potential entry points
            possible_files = ["task_sentiment.py", "task.py", f"task_{module_name}.py"]
            
            found = False
            for file_name in possible_files:
                if (task_dir / file_name).exists():
                    module_path = f"audio_ops_dagster.tasks.{module_name}.{file_name[:-3]}"
                    try:
                        if module_path not in sys.modules:
                            importlib.import_module(module_path)
                        found = True
                        break 
                    except Exception as e:
                        print(f"❌ Error loading {module_path}: {e}")
            
            # If no task file was found, just skip it quietly unless it's the active task
            if not found:
                # Optional: print only if the directory isn't empty (avoids noise from __pycache__)
                pass
# Run auto-discovery
discover_tasks()