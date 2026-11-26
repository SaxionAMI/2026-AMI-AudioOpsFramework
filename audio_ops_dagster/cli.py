"""
cli.py — the ``audio-ops`` command.

    audio-ops new-task NAME [--type T] [--archetype A] [--feature F] [--dir D]
    audio-ops list                      tasks, capabilities, checkpoint status
    audio-ops validate [NAME ...]       check config.yaml files (exit 1 on errors)
    audio-ops run TASK [--job J]        train / evaluate / batch-infer without the UI
    audio-ops predict TASK FILE...      score audio files (no server)
    audio-ops demo-data [--out DIR]     synthetic labelled audio for the example task
    audio-ops serve [--host H] [--port P] [--reload]
    audio-ops dev                       dagster dev for the framework's definitions
    audio-ops doctor                    environment / optional-dependency report
    audio-ops version
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _cmd_new_task(args) -> int:
    from audio_ops_dagster.scaffold import create_task

    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
    try:
        task_dir = create_task(args.name, ttype=args.type, archetype=args.archetype,
                               feature=args.feature,
                               tasks_dir=Path(args.dir) if args.dir else None, classes=classes,
                               config_only=args.config_only)
    except (ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Created {args.type} task '{args.name}' at {task_dir}")
    from audio_ops_dagster.registry import task_search_dirs

    searched = {d.resolve() for d in task_search_dirs()}
    if task_dir.parent.resolve() not in searched:
        print(f"Note: {task_dir.parent} is not searched from here — set "
              f"AUDIO_OPS_TASKS_DIR={task_dir.parent} (or run from its parent folder).")
    if args.type != "transcription":
        print(f"Next: put audio under audio_input/{args.name}/<label>/*.wav, then "
              f"`audio-ops run {args.name}`  (or `audio-ops dev` for the UI).")
    else:
        print("Next: implement predict() with your STT runtime, then `audio-ops serve`.")
    return 0


def _discover():
    from audio_ops_dagster.registry import check_task, discover_tasks, iter_task_dirs

    discover_tasks()
    return [(t, *check_task(t)) for t in iter_task_dirs()]


def _cmd_list(args) -> int:
    from audio_ops_dagster.capabilities import CapabilityError, describe, resolve_capabilities
    from audio_ops_dagster.core.runtime import default_checkpoint_path
    from audio_ops_dagster.registry import TaskConfig

    rows = []
    for task, errors, _warnings in _discover():
        try:
            caps = describe(resolve_capabilities(task.raw_config))
        except CapabilityError:
            caps = "?"
        try:
            cfg = TaskConfig.from_dict(task.raw_config, task_name=task.name)
            ckpt = "yes" if os.path.exists(default_checkpoint_path(cfg)) else "-"
            ttype = cfg.task_type
        except Exception:  # noqa: BLE001
            ckpt, ttype = "?", "?"
        status = "ok" if not errors else f"{len(errors)} error(s)"
        rows.append((task.name, ttype, caps, ckpt, status, str(task.directory)))
    if not rows:
        print("No tasks found. Create one with `audio-ops new-task NAME` or set "
              "AUDIO_OPS_TASKS_DIR.")
        return 0
    header = ("TASK", "TYPE", "CAPABILITIES", "CHECKPOINT", "CONFIG", "PATH")
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    for row in [header, *rows]:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, widths)).rstrip())
    return 0


def _cmd_validate(args) -> int:
    found = _discover()
    if args.names:
        missing = set(args.names) - {t.name for t, *_ in found}
        for m in sorted(missing):
            print(f"✗ {m}: no such task (searched built-in tasks and AUDIO_OPS_TASKS_DIR)")
        found = [f for f in found if f[0].name in args.names]
        if missing:
            return 1
    n_err = 0
    for task, errors, warnings in found:
        mark = "✗" if errors else "✓"
        print(f"{mark} {task.name}  ({task.config_file})")
        for e in errors:
            print(f"    error:   {e}")
        for w in warnings:
            print(f"    warning: {w}")
        n_err += bool(errors)
    if not found:
        print("No tasks found.")
    return 1 if n_err else 0


def _cmd_demo_data(args) -> int:
    from audio_ops_dagster.demo import write_dataset, write_stream_demo

    n = write_dataset(args.out, files_per_class=args.files)
    print(f"Wrote {n} synthetic clips to {args.out}/<class_a|class_b|class_c>/")
    if args.stream_demo:
        p = write_stream_demo(args.stream_demo)
        print(f"Wrote streaming demo audio to {p}")
    return 0


def _cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("The inference service needs the serve extra: "
              "pip install 'audio-ops-framework[serve]'", file=sys.stderr)
        return 1
    reload = args.reload or os.environ.get("INFERENCE_RELOAD") == "1"
    reload_dirs = None
    if reload:
        import audio_ops_dagster
        import audio_ops_inference

        reload_dirs = [str(Path(audio_ops_inference.__file__).parent),
                       str(Path(audio_ops_dagster.__file__).parent)]
        reload_dirs += [p for p in os.environ.get("AUDIO_OPS_TASKS_DIR", "").split(os.pathsep)
                        if p]
    uvicorn.run(
        "audio_ops_inference.api:app",
        host=args.host or os.environ.get("INFERENCE_HOST", "0.0.0.0"),
        port=int(args.port or os.environ.get("INFERENCE_PORT", "8000")),
        workers=1 if reload else int(os.environ.get("INFERENCE_WORKERS", "1")),
        reload=reload,
        reload_dirs=reload_dirs,
    )
    return 0


def _cmd_dev(args) -> int:
    # Prefer the dagster next to this Python, so `.venv/bin/audio-ops dev` works
    # without activating the virtual environment.
    local = Path(sys.executable).parent / ("dagster.exe" if os.name == "nt" else "dagster")
    exe = str(local) if local.exists() else shutil.which("dagster")
    if exe is None:
        print("Dagster is not installed: pip install 'audio-ops-framework[train]'",
              file=sys.stderr)
        return 1
    cmd = [exe, "dev", "-m", "audio_ops_dagster.definitions", *(a for a in args.extra if a != "--")]
    env = dict(os.environ)
    if not env.get("DAGSTER_HOME"):
        # Keep run history between sessions instead of a throwaway temp dir.
        home = Path.cwd() / "dagster_home"
        home.mkdir(exist_ok=True)
        env["DAGSTER_HOME"] = str(home)
    print(f"$ DAGSTER_HOME={env['DAGSTER_HOME']} {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


_JOB_SUFFIX = {"train": "full_training_job", "evaluate": "evaluation_job",
               "infer": "inference_job", "prep": "data_prep_job"}


def _compact_logger():
    """A Dagster logger that prints ``[step] message`` instead of the full prefix."""
    import logging

    from dagster import Field, logger

    class _Fmt(logging.Formatter):
        def format(self, record):
            meta = getattr(record, "dagster_meta", None) or {}
            msg = meta.get("orig_message") or record.getMessage()
            step = (meta.get("step_key") or "").replace(f"{meta.get('job_name', '')}_", "")
            prefix = f"[{step}] " if step else ""
            level = "" if record.levelno == logging.INFO else f"{record.levelname}: "
            return f"{prefix}{level}{msg}"

    @logger(config_schema={"log_level": Field(str, is_required=False, default_value="INFO")})
    def compact(init_context):
        lg = logging.Logger("audio_ops.run", level=init_context.logger_config["log_level"])
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_Fmt())
        lg.addHandler(handler)
        return lg

    return compact


def _cmd_run(args) -> int:
    """Run a task's pipeline headlessly (no UI, ephemeral Dagster instance)."""
    try:
        from audio_ops_dagster.definitions import build_definitions
    except ImportError:
        print("Running pipelines needs the train extra: "
              "pip install 'audio-ops-framework[train]'", file=sys.stderr)
        return 1
    defs = build_definitions(loggers={"compact": _compact_logger()})
    available = {j.name for j in defs.resolve_all_job_defs()}
    order = [args.job] if args.job else ["train", "evaluate", "infer"]
    job = next((f"{args.task}_{_JOB_SUFFIX[k]}" for k in order
                if f"{args.task}_{_JOB_SUFFIX[k]}" in available), None)
    if job is None:
        mine = sorted(j for j in available if j.startswith(f"{args.task}_"))
        print(f"error: no {args.job or 'runnable'} job for '{args.task}'. "
              f"Available: {mine or 'none (unknown task, config error, or serve-only)'} — "
              f"see `audio-ops list` / `audio-ops validate`.", file=sys.stderr)
        return 1

    run_config: dict = {"loggers": {"compact": {"config": {"log_level": args.log_level}}}}
    overrides = {k: v for k, v in (("input_dir", args.input_dir),
                                   ("output_dir", args.output_dir)) if v}
    if overrides:
        run_config["ops"] = {f"{args.task}_raw_audio_files": {"config": overrides}}
    print(f"Running {job} …")
    result = defs.resolve_job_def(job).execute_in_process(run_config=run_config,
                                                          raise_on_error=False)
    if not result.success:
        for ev in result.all_events:
            if ev.is_step_failure:
                err = getattr(ev.event_specific_data, "error", None)
                print(f"✗ {ev.step_key}: {err.message.strip() if err else ev.message}",
                      file=sys.stderr)
        return 1

    from audio_ops_dagster.registry import TaskConfig, iter_task_dirs

    task = next(t for t in iter_task_dirs() if t.name == args.task)
    out = Path(args.output_dir or TaskConfig.from_dict(task.raw_config,
                                                       task_name=args.task).output_dir)
    print(f"✓ {job} succeeded. Artifacts in {out}/")
    for f in sorted((out / "outputs").glob("*")) if (out / "outputs").is_dir() else []:
        print(f"  {f}")
    ckpt = out / "checkpoints" / "best_model.pt"
    if ckpt.exists():
        print(f"  {ckpt}")
    return 0


def _cmd_predict(args) -> int:
    """Score audio files with a trained/servable task — no server needed."""
    import json as _json

    try:
        import librosa

        from audio_ops_dagster.registry import TaskConfig, discover_tasks, get_task, iter_task_dirs
        from audio_ops_inference.wrappers import get_wrapper_factory, resolve_wrapper_type
    except ImportError as exc:
        print(f"predict needs torch: pip install 'audio-ops-framework[serve]' ({exc})",
              file=sys.stderr)
        return 1
    discover_tasks()
    found = next((t for t in iter_task_dirs() if t.name == args.task), None)
    if found is None:
        print(f"error: no task '{args.task}' (see `audio-ops list`)", file=sys.stderr)
        return 1
    cfg = TaskConfig.from_dict(found.raw_config, task_name=args.task)
    if args.checkpoint:
        cfg.checkpoint_path = args.checkpoint
    task = get_task(args.task, cfg)
    wrapper = get_wrapper_factory(resolve_wrapper_type(task, cfg))(task, args.task, cfg)
    if wrapper is None:
        print(f"error: '{args.task}' has no checkpoint yet — run `audio-ops run {args.task}` "
              f"or pass --checkpoint.", file=sys.stderr)
        return 1

    status = 0
    for path in args.files:
        try:
            audio, _ = librosa.load(path, sr=16000, mono=True)
        except Exception as exc:  # noqa: BLE001
            print(f"✗ {path}: cannot read audio ({exc})", file=sys.stderr)
            status = 1
            continue
        result = wrapper.predict_clip(audio)
        if args.json:
            print(_json.dumps({"file": path, **result}, default=str))
            continue
        if "transcription" in result:
            summary = result["transcription"]
        elif "label" in result:
            conf = result.get("confidence")
            summary = f"{result['label']}" + (f"  ({conf:.1%})" if conf is not None else "")
        elif "value" in result:
            summary = f"{result['value']:.4g}"
        elif "labels" in result:
            summary = ", ".join(result["labels"]) or "(none)"
        else:
            summary = _json.dumps(result, default=str)[:200]
        print(f"{path}: {summary}")
        if args.segments:
            for seg in result.get("segments", []):
                val = seg.get("label", seg.get("value", seg.get("labels")))
                c = seg.get("confidence")
                print(f"    {seg['start_sec']:7.2f}–{seg['end_sec']:<7.2f} {val}"
                      + (f"  ({c:.1%})" if c is not None else ""))
    return status


def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _cmd_doctor(args) -> int:
    from audio_ops_dagster import __version__

    print(f"audio-ops-framework {__version__} on Python {sys.version.split()[0]}")
    groups = {
        "core": ["numpy", "yaml", "pydantic", "librosa", "soundfile"],
        "train": ["torch", "dagster", "mlflow", "sklearn", "pandas"],
        "serve": ["torch", "fastapi", "uvicorn", "multipart"],
        "ast": ["transformers", "torchaudio"],
        "s3": ["boto3"],
        "metrics": ["influxdb_client"],
    }
    for group, mods in groups.items():
        status = ", ".join(f"{m}{'' if _have(m) else ' (missing)'}" for m in mods)
        ok = all(_have(m) for m in mods)
        print(f"  [{'x' if ok else ' '}] {group:<8} {status}")
    if _have("torch"):
        import torch

        accel = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu only")
        print(f"  torch {torch.__version__} — accelerator: {accel}")
    print(f"  ffmpeg: {shutil.which('ffmpeg') or 'not found (needed for RTSP/stream pulls)'}")
    keys = ["AUDIO_OPS_TASKS_DIR", "STORAGE_BACKEND", "ENABLE_INFLUXDB", "MLFLOW_TRACKING_URI",
            "ENABLE_LLM", "ENABLE_FUSION", "DEVICE", "API_KEY"]
    print("  environment:")
    for k in keys:
        v = os.environ.get(k)
        if k == "API_KEY" and v:
            v = "(set)"
        print(f"    {k}={v if v is not None else ''}")
    return 0


def _cmd_version(args) -> int:
    from audio_ops_dagster import __version__

    print(__version__)
    return 0


def build_parser() -> argparse.ArgumentParser:
    from audio_ops_dagster.scaffold import ARCHETYPES, TYPES

    p = argparse.ArgumentParser(
        prog="audio-ops",
        description="Audio Ops Framework — build audio ML pipelines from small tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""typical workflow:
  audio-ops new-task my_task --config-only --classes a,b   # tasks/my_task/config.yaml
  # put audio in audio_input/my_task/<label>/*.wav   (or: audio-ops demo-data)
  audio-ops validate my_task
  audio-ops run my_task                                    # train, evaluate, score
  audio-ops predict my_task clip.wav
  audio-ops serve                                          # HTTP / RTSP / WebSocket API

docs: https://saxionami.github.io/2026-AMI-AudioOpsFramework/""")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    n = sub.add_parser("new-task", help="scaffold a new task")
    n.add_argument("name")
    n.add_argument("--type", default="classification", choices=TYPES)
    n.add_argument("--archetype", default=None, choices=ARCHETYPES)
    n.add_argument("--feature", default="spectrogram")
    n.add_argument("--classes", default=None, help="comma-separated class labels")
    n.add_argument("--config-only", action="store_true",
                   help="only write config.yaml (uses the ready-made base for the type)")
    n.add_argument("--dir", default=None,
                   help="tasks directory (default: first AUDIO_OPS_TASKS_DIR entry, the "
                        "framework checkout's tasks/, or ./tasks)")
    n.set_defaults(func=_cmd_new_task)

    sub.add_parser("list", help="list discovered tasks").set_defaults(func=_cmd_list)

    r = sub.add_parser("run", help="run a task's pipeline without the UI")
    r.add_argument("task")
    r.add_argument("--job", choices=sorted(_JOB_SUFFIX),
                   help="default: train, else evaluate, else infer — whatever the task has")
    r.add_argument("--input-dir", default=None, help="score/train this folder instead")
    r.add_argument("--output-dir", default=None, help="write features/outputs here instead")
    r.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    r.set_defaults(func=_cmd_run)

    pr = sub.add_parser("predict", help="score audio files with a task (no server)")
    pr.add_argument("task")
    pr.add_argument("files", nargs="+")
    pr.add_argument("--checkpoint", default=None, help="use this checkpoint file")
    pr.add_argument("--segments", action="store_true", help="also print per-segment results")
    pr.add_argument("--json", action="store_true", help="one JSON object per file")
    pr.set_defaults(func=_cmd_predict)

    v = sub.add_parser("validate", help="validate task configs")
    v.add_argument("names", nargs="*")
    v.set_defaults(func=_cmd_validate)

    d = sub.add_parser("demo-data", help="write synthetic training audio")
    d.add_argument("--out", default="audio_input/example_classifier")
    d.add_argument("--files", type=int, default=12, help="clips per class")
    d.add_argument("--stream-demo", default=None, metavar="WAV",
                   help="also write a streaming demo file (e.g. mediamtx/test_audio/demo.wav)")
    d.set_defaults(func=_cmd_demo_data)

    s = sub.add_parser("serve", help="run the inference API")
    s.add_argument("--host", default=None)
    s.add_argument("--port", default=None)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=_cmd_serve)

    dv = sub.add_parser("dev", help="run `dagster dev` on the framework definitions")
    dv.add_argument("extra", nargs=argparse.REMAINDER, help="extra args for dagster dev")
    dv.set_defaults(func=_cmd_dev)

    sub.add_parser("doctor", help="environment report").set_defaults(func=_cmd_doctor)
    sub.add_parser("version", help="print the version").set_defaults(func=_cmd_version)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "dev" and not {"-h", "--help"} & set(argv[1:2]):
        # Everything after `dev` goes to `dagster dev` verbatim.
        return int(_cmd_dev(argparse.Namespace(extra=argv[1:])) or 0)
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
