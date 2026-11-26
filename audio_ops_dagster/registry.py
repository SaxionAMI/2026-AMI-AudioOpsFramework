"""
Task registry — registration, configuration and auto-discovery of tasks.

Discovery sources (both scanned, in this order):
  1. The framework's built-in tasks package: ``audio_ops_dagster/tasks/``
  2. External task directories from the ``AUDIO_OPS_TASKS_DIR`` env var
     (``os.pathsep``- or comma-separated). This is how a *project repo* keeps
     its own tasks outside the installed framework package::

         AUDIO_OPS_TASKS_DIR=./tasks dagster dev -m audio_ops_dagster.definitions
         AUDIO_OPS_TASKS_DIR=./tasks audio-ops serve

A task is a directory ``<name>/`` containing ``config.yaml`` plus
``task_<name>.py`` (or ``task.py``) with a class decorated by
:func:`register_task`. The **directory name is the task's identity**; a
``task_name`` in the config, if present, must match it.

External task directories are imported as real packages
(``audio_ops_external.<name>``), so a task can split its code across modules
and use relative imports (``from .model import Net``).
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import pkgutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type

import yaml

from audio_ops_dagster._log import get_logger
from audio_ops_dagster.core.config import TaskConfig

logger = get_logger(__name__)

EXTERNAL_PACKAGE = "audio_ops_external"


# TaskConfig lives in core/config.py; re-exported here for backwards compatibility.


_TASKS: Dict[str, Type] = {}


def register_task(name: str, task_class: Optional[Type] = None):
    """
    Register a task class under ``name``. Usable as decorator or direct call::

        @register_task("my_task")
        class MyTask(AudioTask): ...

        register_task("my_task", MyTaskClass)
    """

    def wrapper(cls: Type):
        previous = _TASKS.get(name)
        if previous is not None and previous is not cls:
            logger.warning(
                "Task '%s' re-registered (%s.%s replaces %s.%s).",
                name, cls.__module__, cls.__qualname__, previous.__module__, previous.__qualname__,
            )
        _TASKS[name] = cls
        logger.debug("Registered task: %s", name)
        return cls

    if task_class is not None:
        return wrapper(task_class)
    return wrapper


def get_task(name: str, config: Any):
    """Instantiate the registered task ``name`` with ``config``.

    ``config`` may be a :class:`TaskConfig` or a plain dict (converted with
    :meth:`TaskConfig.from_dict`).
    """
    if name not in _TASKS:
        raise ValueError(
            f"Unknown task '{name}'. Registered: {sorted(_TASKS)}. Does "
            f"tasks/{name}/task_{name}.py define a class decorated with "
            f"@register_task('{name}')?"
        )
    if isinstance(config, dict):
        config = TaskConfig.from_dict(config, task_name=name)
    return _TASKS[name](config)


def get_task_class(name: str) -> Optional[Type]:
    return _TASKS.get(name)


def list_tasks() -> List[str]:
    """Names of all registered tasks."""
    return sorted(_TASKS)


# ── Discovery ────────────────────────────────────────────────────────────────


def external_task_dirs() -> List[Path]:
    """External task directories: ``AUDIO_OPS_TASKS_DIR``, else ``./tasks`` if it exists.

    The ``./tasks`` convention means a project repo works with no configuration:
    run ``audio-ops`` from the repo root and its tasks are found.
    """
    raw = os.environ.get("AUDIO_OPS_TASKS_DIR", "")
    if not raw.strip():
        local = Path.cwd() / "tasks"
        return [local] if local.is_dir() else []
    dirs: List[Path] = []
    for part in raw.replace(",", os.pathsep).split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        p = Path(part).expanduser()
        if p.is_dir():
            dirs.append(p)
        else:
            logger.warning("AUDIO_OPS_TASKS_DIR entry not found: %s", part)
    return dirs


def _builtin_tasks_dir() -> Optional[Path]:
    try:
        import audio_ops_dagster.tasks as tasks_pkg

        return Path(tasks_pkg.__path__[0])
    except ImportError:
        return None


def task_search_dirs() -> List[Path]:
    """All directories to scan for tasks: built-in package + external repos.

    Set ``AUDIO_OPS_DISABLE_BUILTIN_TASKS=true`` to hide the bundled
    ``example_classifier`` (useful in project repos).
    """
    dirs: List[Path] = []
    builtin = _builtin_tasks_dir()
    if builtin is not None and os.environ.get(
        "AUDIO_OPS_DISABLE_BUILTIN_TASKS", "false"
    ).lower() not in ("1", "true", "yes"):
        dirs.append(builtin)
    dirs.extend(external_task_dirs())
    seen, out = set(), []
    for d in dirs:
        r = d.resolve()
        if r not in seen:
            seen.add(r)
            out.append(d)
    return out


def _task_module_file(task_dir: Path) -> Optional[str]:
    for file_name in (f"task_{task_dir.name}.py", "task.py"):
        if (task_dir / file_name).exists():
            return file_name[:-3]
    return None


def _ensure_namespace(name: str, path: Optional[Path] = None) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
        spec.submodule_search_locations = [str(path)] if path else []
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
    return module


def _load_external_task(task_dir: Path) -> None:
    """Import an external task directory as package ``audio_ops_external.<name>``."""
    mod_name = _task_module_file(task_dir)
    if mod_name is None:
        return
    pkg = f"{EXTERNAL_PACKAGE}.{task_dir.name}"
    full = f"{pkg}.{mod_name}"
    if full in sys.modules:
        return
    _ensure_namespace(EXTERNAL_PACKAGE)
    try:
        init = task_dir / "__init__.py"
        if init.exists():
            spec = importlib.util.spec_from_file_location(
                pkg, init, submodule_search_locations=[str(task_dir)]
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[pkg] = module
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        else:
            _ensure_namespace(pkg, task_dir)
        importlib.import_module(full)
    except Exception as e:  # noqa: BLE001 — report, don't kill discovery
        logger.error("Error loading task module %s: %s", task_dir / f"{mod_name}.py", e)
        for key in (full, pkg):
            sys.modules.pop(key, None)


def _register_config_only(task_dir: Path) -> None:
    """A folder with only config.yaml gets a ready-made base chosen by task_type."""
    name = task_dir.name
    if name in _TASKS or _task_module_file(task_dir) is not None:
        return
    cfg_file = task_dir / "config.yaml"
    if not cfg_file.exists():
        return
    try:
        with open(cfg_file, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError:
        return
    from audio_ops_dagster.core.task_types import BASES_BY_TYPE

    base = BASES_BY_TYPE.get(str(raw.get("task_type") or "classification"))
    if base is not None:
        _TASKS[name] = base
        logger.debug("Config-only task '%s' uses %s", name, base.__name__)


def discover_tasks() -> None:
    """Import all task modules so their ``@register_task`` decorators run.

    Idempotent: modules already imported are skipped.
    """
    builtin = _builtin_tasks_dir()
    if builtin is not None:
        for _, module_name, is_pkg in pkgutil.iter_modules([str(builtin)]):
            if not is_pkg:
                continue
            mod_file = _task_module_file(builtin / module_name)
            if mod_file is None:
                _register_config_only(builtin / module_name)
                continue
            module_path = f"audio_ops_dagster.tasks.{module_name}.{mod_file}"
            if module_path in sys.modules:
                continue
            try:
                importlib.import_module(module_path)
            except Exception as e:  # noqa: BLE001
                logger.error("Error loading %s: %s", module_path, e)

    for ext in external_task_dirs():
        for sub in sorted(ext.iterdir()):
            if sub.is_dir() and not sub.name.startswith((".", "__")):
                _load_external_task(sub)
                _register_config_only(sub)


@dataclass
class DiscoveredTask:
    """A task directory found on disk, with its raw config."""

    name: str
    directory: Path
    raw_config: Dict[str, Any]

    @property
    def config_file(self) -> Path:
        return self.directory / "config.yaml"


def iter_task_dirs() -> Iterator[DiscoveredTask]:
    """Yield every task directory with a ``config.yaml``, first match wins.

    Used by both Dagster (``definitions.py``) and the inference service so
    they always agree on which tasks exist and what they are called.
    """
    seen: set = set()
    for root in task_search_dirs():
        if not root.exists():
            continue
        for task_dir in sorted(root.iterdir()):
            if not task_dir.is_dir() or task_dir.name.startswith((".", "__")):
                continue
            cfg_file = task_dir / "config.yaml"
            if not cfg_file.exists():
                continue
            name = task_dir.name
            if name in seen:
                logger.warning("Duplicate task '%s' in %s ignored (already loaded).", name, root)
                continue
            seen.add(name)
            try:
                with open(cfg_file, encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                logger.error("Task '%s': config.yaml is not valid YAML: %s", name, exc)
                raw = {"__yaml_error__": str(exc)}
            if not isinstance(raw, dict):
                raw = {"__yaml_error__": "config.yaml must be a mapping"}
            yield DiscoveredTask(name=name, directory=task_dir, raw_config=raw)


def check_task(task: DiscoveredTask) -> Tuple[List[str], List[str]]:
    """Validate a discovered task: config schema + registration consistency."""
    from audio_ops_dagster.core.task_config_schema import validate_task_config

    if "__yaml_error__" in task.raw_config:
        return [f"config.yaml could not be parsed: {task.raw_config['__yaml_error__']}"], []

    errors, warnings = validate_task_config(task.raw_config)
    declared = task.raw_config.get("task_name")
    if declared and declared != task.name:
        errors.append(
            f"task_name '{declared}' in config.yaml does not match its folder name "
            f"'{task.name}'. The folder name is the task's identity — rename one of them."
        )
    cls = get_task_class(task.name)
    if cls is None:
        errors.append(
            f"no class registered as '{task.name}'. Either add "
            f"{task.directory}/task_{task.name}.py with @register_task('{task.name}'), or — "
            f"for classification/regression/detection — keep the folder config-only."
        )
    else:
        cfg_type = task.raw_config.get("task_type")
        cls_type = getattr(cls, "TASK_TYPE", None)
        if cfg_type and cls_type and cfg_type != cls_type and cfg_type != "nonverbal_pretrained":
            warnings.append(
                f"task_type '{cfg_type}' in config.yaml differs from {cls.__name__}.TASK_TYPE "
                f"'{cls_type}'; the class attribute wins."
            )
    return errors, warnings


# Run auto-discovery on import (historical behaviour; idempotent).
discover_tasks()
