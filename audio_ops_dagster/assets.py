"""
assets.py — Dagster asset definitions for the Audio Ops pipeline.

``build_audio_assets`` returns, per task, only the assets its capabilities
need (see ``_CAPABILITY_ASSETS``)::

    raw_audio_files → segmented_features → dataset_splits → trained_model → model_evaluation
                                         ↘ inference_results

Every asset optionally writes time-series metrics through the metrics
resource (a no-op unless InfluxDB is enabled) and syncs artifacts to object
storage when the task sets ``use_remote_storage: true`` and a backend is on.
"""


import csv as _csv
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from dagster import AssetIn, AssetKey, Config, MetadataValue, Nothing, Output, asset

from audio_ops_dagster.capabilities import BATCH_INFER, EVALUATE, TRAIN
from audio_ops_dagster.core.audio_processor import AudioProcessor
from audio_ops_dagster.core.runtime import (
    default_checkpoint_path,
    list_audio_files,
    load_task_model,
    parse_segment_name,
    resolve_device,
    to_model_input,
)
from audio_ops_dagster.core.storage import ArtifactType
from audio_ops_dagster.registry import get_task


class InferenceRunConfig(Config):
    """
    Per-run overrides for the inference path, set in the Dagster Launchpad.

    Launch a task's ``<task>_inference_job`` and set these once on the
    ``<task>_raw_audio_files`` asset to score an arbitrary folder::

        ops:
          <task>_raw_audio_files:
            config:
              input_dir: /data/new_batch        # audio to score
              output_dir: /data/new_batch_out   # where features + results go

    Leave them unset to use the task's configured dirs (normal pipeline runs).
    The trained checkpoint is read from ``checkpoint_path`` in the task config,
    else ``INFERENCE_CHECKPOINT``, else ``<task output_dir>/checkpoints/best_model.pt``.
    """

    input_dir: Optional[str] = None
    output_dir: Optional[str] = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_results_csv(path, results: Dict[str, Dict[str, Any]]) -> None:
    """Flatten per-segment predictions (scalar fields only) into a CSV."""
    field_keys: list = []
    for v in results.values():
        for k, val in v.items():
            if isinstance(val, (str, int, float, bool)) and k not in field_keys:
                field_keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["segment"] + field_keys)
        for seg, v in results.items():
            w.writerow([seg] + [v.get(k, "") for k in field_keys])


def _normalize_prediction(pred: Any) -> Dict[str, Any]:
    if isinstance(pred, list) and pred:
        pred = pred[0]
    if isinstance(pred, dict):
        return dict(pred)
    return {"result": pred if isinstance(pred, (str, int, float, bool)) else str(pred)}


def _label_to_idx(task_config) -> Optional[Dict[str, int]]:
    labels = getattr(task_config, "class_labels", None)
    return {str(label): i for i, label in enumerate(labels)} if labels else None


# Which assets each capability requires, including prerequisites.
# `serve` is absent on purpose — it produces no Dagster assets.
# `evaluate` without `train` scores an existing checkpoint (eval_only).
_CAPABILITY_ASSETS = {
    TRAIN:       {"raw_audio_files", "segmented_features", "dataset_splits", "trained_model"},
    EVALUATE:    {"raw_audio_files", "segmented_features", "dataset_splits", "model_evaluation"},
    BATCH_INFER: {"raw_audio_files", "segmented_features", "inference_results"},
}

# Canonical materialization order (respects the dependency chain).
_ASSET_ORDER = [
    "raw_audio_files",
    "segmented_features",
    "dataset_splits",
    "trained_model",
    "model_evaluation",
    "inference_results",
]


def wanted_assets(capabilities) -> list:
    caps = frozenset(capabilities)
    wanted = set().union(*(_CAPABILITY_ASSETS.get(c, set()) for c in caps)) if caps else set()
    return [k for k in _ASSET_ORDER if k in wanted]


def build_audio_assets(task_name: str, resource_key: str, capabilities):
    """
    Factory creating the assets for one task, namespaced by ``task_name``.

    Only the assets required by the task's declared ``capabilities`` are
    returned. A serve-only task returns an empty list.
    """
    prefix = task_name
    _required = {resource_key, "minio_storage", "influxdb_metrics"}

    caps = frozenset(capabilities)
    wanted = set(wanted_assets(caps))
    if not wanted:
        return []
    trains = "trained_model" in wanted

    def _task_and_config(context):
        res = getattr(context.resources, resource_key)
        task_config = res.to_task_config()
        return res, task_config, get_task(task_name, task_config)

    def _processor(task_config) -> AudioProcessor:
        return AudioProcessor(
            feature_type=task_config.feature_type,
            sample_rate=int(task_config.sample_rate),
            segment_duration=float(task_config.segment_duration),
            overlap=float(task_config.segment_overlap),
        )

    def _build_loaded_model(context, task, checkpoint_path, required=True):
        if os.path.exists(checkpoint_path):
            context.log.info(f"Loading checkpoint: {checkpoint_path}")
        elif not required:
            context.log.warning(f"No checkpoint at {checkpoint_path} — using an untrained model.")
        device = resolve_device()
        model, _meta = load_task_model(task, checkpoint_path, device=device, required=required)
        return model, device

    def _sync(context, task_config, fn, *args):
        storage = context.resources.minio_storage
        if task_config.use_remote_storage and storage.enabled:
            try:
                return fn(task_config.effective_bucket_prefix(), *args)
            except Exception as e:  # noqa: BLE001
                context.log.warning(f"Object storage upload failed (non-fatal): {e}")
        return None

    # ── 1. Raw Audio Files ───────────────────────────────────────────────────

    @asset(
        name=f"{prefix}_raw_audio_files",
        required_resource_keys=_required,
        group_name=prefix,
        description="Audio files found under the task's input_dir (<input_dir>/<label>/*).",
    )
    def raw_audio_files(context, config: InferenceRunConfig):
        res = getattr(context.resources, resource_key)
        metrics = context.resources.influxdb_metrics

        # Per-run overrides (Launchpad) fall back to the task's configured dirs.
        input_dir = config.input_dir or res.input_dir
        output_dir = config.output_dir or res.output_dir

        metrics.write_pipeline_event(task_name=task_name, run_id=context.run_id,
                                     event="raw_audio_scan_start")
        Path(input_dir).mkdir(parents=True, exist_ok=True)
        files = [str(p) for p in list_audio_files(input_dir)]
        if not files:
            context.log.warning(
                f"No audio found in {os.path.abspath(input_dir)}. Expected "
                f"<input_dir>/<label>/*.wav (try `audio-ops demo-data` for a sample set)."
            )
        else:
            context.log.info(f"Found {len(files)} raw audio files in {input_dir}")

        metrics.write_pipeline_event(task_name=task_name, run_id=context.run_id,
                                     event="raw_audio_scan_complete",
                                     fields={"file_count": len(files)})

        # Thread resolved dirs downstream so the whole path honors the override.
        return Output(
            value={"files": files, "input_dir": input_dir, "output_dir": output_dir},
            metadata={"file_count": MetadataValue.int(len(files)),
                      "input_dir": MetadataValue.path(os.path.abspath(input_dir))},
        )

    # ── 2. Segmented Features ────────────────────────────────────────────────

    @asset(
        name=f"{prefix}_segmented_features",
        ins={"raw_audio_files": AssetIn(key=AssetKey(f"{prefix}_raw_audio_files"))},
        required_resource_keys=_required,
        group_name=prefix,
        description="Fixed-length segments → one .npy feature file each.",
    )
    def segmented_features(context, raw_audio_files):
        res, task_config, _task = _task_and_config(context)
        storage = context.resources.minio_storage
        metrics = context.resources.influxdb_metrics

        t0 = time.time()
        metrics.write_pipeline_event(task_name=task_name, run_id=context.run_id,
                                     event="feature_extraction_start")

        input_dir = raw_audio_files.get("input_dir", res.input_dir)
        base_out = raw_audio_files.get("output_dir", res.output_dir)

        processor = _processor(task_config)
        feature_out = os.path.abspath(os.path.join(base_out, "features"))
        stats = processor.process_directory(
            input_dir, feature_out,
            force_recompute=bool(task_config.force_recompute),
            progress=context.log.info,
        )

        # Shape sanity check: the most common cause of a model that "trains"
        # but serves garbage is an input_shape that doesn't match the features.
        shape = stats.get("feature_shape")
        expected = task_config.input_shape
        if shape and expected:
            effective = [1, *shape] if len(shape) == 2 else list(shape)
            if list(expected) != effective:
                context.log.warning(
                    f"input_shape {list(expected)} in config.yaml differs from the actual "
                    f"feature shape {effective} (feature_type={task_config.feature_type}, "
                    f"segment_duration={task_config.segment_duration}s). Shape-agnostic "
                    f"models are fine; otherwise set input_shape: {effective}."
                )

        elapsed = time.time() - t0
        metrics.write_pipeline_event(
            task_name=task_name, run_id=context.run_id, event="feature_extraction_complete",
            fields={"total_files": stats.get("total_files", 0),
                    "total_segments": stats.get("total_segments", 0),
                    "duration_sec": round(elapsed, 2)},
        )
        _sync(context, task_config, storage.upload_task_directory, ArtifactType.FEATURES,
              feature_out)

        return Output(
            value={**stats, "feature_dir": feature_out, "output_dir": base_out,
                   "input_dir": input_dir, "files": raw_audio_files.get("files", []),
                   "hop_seconds": processor.hop_seconds},
            metadata={
                "feature_dir": MetadataValue.path(feature_out),
                "total_segments": MetadataValue.int(stats.get("total_segments", 0)),
                "total_files": MetadataValue.int(stats.get("total_files", 0)),
                "cached_files": MetadataValue.int(stats.get("skipped_files", 0)),
                "errored_files": MetadataValue.int(stats.get("errored_files", 0)),
                "feature_shape": MetadataValue.text(str(shape)),
            },
        )

    # ── 3. Dataset Splits ────────────────────────────────────────────────────

    @asset(
        name=f"{prefix}_dataset_splits",
        ins={"segmented_features": AssetIn(key=AssetKey(f"{prefix}_segmented_features"))},
        required_resource_keys=_required,
        group_name=prefix,
        description="train/val/test CSVs (group strategy = no recording leaks across splits).",
    )
    def dataset_splits(context, segmented_features):
        import pandas as pd

        from audio_ops_dagster.core.audio_dataset import AudioDataset

        res, task_config, task = _task_and_config(context)
        storage = context.resources.minio_storage
        metrics = context.resources.influxdb_metrics

        feature_dir = segmented_features.get("feature_dir") or os.path.abspath(
            os.path.join(res.output_dir, "features"))
        dataset_dir = os.path.abspath(os.path.join(res.output_dir, "dataset"))

        if not os.path.isdir(feature_dir) or not any(Path(feature_dir).rglob("*.npy")):
            raise ValueError(
                f"No .npy feature files under {feature_dir}. Check that the input "
                f"directory {res.input_dir!r} contains audio in <label>/ folders."
            )

        split_paths = AudioDataset.create_splits_csv(
            feature_dir=feature_dir,
            output_dir=dataset_dir,
            ratios=task_config.split_ratios,
            strategy=task_config.split_strategy,
            seed=int(task_config.seed),
            stratify=task.task_type != "regression",
            group_pattern=task_config.group_pattern,
        )

        counts = {}
        for split_name, csv_path in split_paths.items():
            counts[split_name] = len(pd.read_csv(csv_path))
            metrics.write_pipeline_event(task_name=task_name, run_id=context.run_id,
                                         event=f"dataset_split_{split_name}",
                                         fields={"sample_count": counts[split_name]})
            _sync(context, task_config, storage.upload_task_file, ArtifactType.DATASET, csv_path)

        return Output(
            value={**split_paths, "feature_dir": feature_dir},
            metadata={f"{k}_segments": MetadataValue.int(v) for k, v in counts.items()},
        )

    # ── 4. Trained Model ─────────────────────────────────────────────────────

    @asset(
        name=f"{prefix}_trained_model",
        ins={"dataset_splits": AssetIn(key=AssetKey(f"{prefix}_dataset_splits"))},
        required_resource_keys=_required,
        group_name=prefix,
        description="Trains the task's model; writes checkpoints/best_model.pt.",
    )
    def trained_model(context, dataset_splits):
        from torch.utils.data import DataLoader

        from audio_ops_dagster.core.audio_dataset import AudioDataset, target_type_for
        from audio_ops_dagster.core.model_trainer import ModelTrainer

        res, task_config, task = _task_and_config(context)
        storage = context.resources.minio_storage
        metrics = context.resources.influxdb_metrics

        feature_root = dataset_splits.get("feature_dir") or os.path.join(res.output_dir,
                                                                          "features")
        target_type = target_type_for(task.task_type)
        if target_type == "index" and not task_config.class_labels:
            # No class_labels configured: use every label folder, sorted. They are
            # stored in the checkpoint, so evaluation and serving agree.
            import pandas as pd

            found = set()
            for split in ("train", "val", "test"):
                found |= set(pd.read_csv(dataset_splits[split])["label"].astype(str))
            task_config.class_labels = sorted(found)
            task_config.num_classes = len(found)
            context.log.info(f"class_labels inferred from folders: {task_config.class_labels}")
        label_to_idx = _label_to_idx(task_config) if target_type == "index" else None

        def _ds(split):
            return AudioDataset(csv_path=dataset_splits[split], feature_dir=feature_root,
                                label_to_idx=label_to_idx, target_type=target_type)

        train_ds, val_ds = _ds("train"), _ds("val")
        model = task.build_model(input_shape=task_config.input_shape)
        workers = int(task_config.num_workers)
        train_loader = DataLoader(train_ds, batch_size=task_config.batch_size, shuffle=True,
                                  num_workers=workers)
        val_loader = DataLoader(val_ds, batch_size=task_config.batch_size, num_workers=workers)

        checkpoint_dir = os.path.abspath(os.path.join(res.output_dir, "checkpoints"))
        metrics.write_pipeline_event(
            task_name=task_name, run_id=context.run_id, event="training_start",
            fields={"epochs": task_config.epochs, "batch_size": task_config.batch_size},
        )

        trainer = ModelTrainer(task, task_config, checkpoint_dir=checkpoint_dir,
                               metrics=metrics, run_id=context.run_id, context=context)
        checkpoint_path = trainer.train(model, train_loader, val_loader)

        metrics.write_pipeline_event(task_name=task_name, run_id=context.run_id,
                                     event="training_complete")
        model_uri = _sync(context, task_config, storage.upload_task_file, ArtifactType.MODELS,
                          checkpoint_path)

        return Output(
            value={"checkpoint_path": checkpoint_path, "model_uri": model_uri},
            metadata={
                "checkpoint_path": MetadataValue.path(checkpoint_path),
                "best_epoch": MetadataValue.int(trainer.best_epoch),
                "train_segments": MetadataValue.int(len(train_ds)),
                "val_segments": MetadataValue.int(len(val_ds)),
                "model_uri": MetadataValue.text(model_uri or "local only"),
            },
        )

    # ── 5. Model Evaluation ──────────────────────────────────────────────────

    def _evaluate(context, dataset_splits, trained_model_info=None):
        from torch.utils.data import DataLoader

        from audio_ops_dagster.core.audio_dataset import AudioDataset, target_type_for
        from audio_ops_dagster.core.model_trainer import ModelTrainer

        res, task_config, task = _task_and_config(context)
        metrics = context.resources.influxdb_metrics

        checkpoint_path = (trained_model_info or {}).get("checkpoint_path") or \
            default_checkpoint_path(task_config)
        model, _device = _build_loaded_model(context, task, checkpoint_path)

        target_type = target_type_for(task.task_type)
        label_to_idx = _label_to_idx(task_config) if target_type == "index" else None
        feature_root = dataset_splits.get("feature_dir") or os.path.join(res.output_dir,
                                                                          "features")
        test_ds = AudioDataset(csv_path=dataset_splits["test"], feature_dir=feature_root,
                               label_to_idx=label_to_idx, target_type=target_type)
        test_loader = DataLoader(test_ds, batch_size=task_config.batch_size)

        test_loss, eval_metrics = ModelTrainer(task, task_config).test(model, test_loader)
        report = {"task": task_name, "checkpoint": checkpoint_path,
                  "test_segments": len(test_ds), "test_loss": test_loss, **eval_metrics,
                  "evaluated_at": datetime.now().isoformat(timespec="seconds")}

        out_dir = Path(res.output_dir) / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"evaluation_{prefix}.json"
        report_path.write_text(json.dumps(_jsonable(report), indent=2))
        context.log.info(f"Evaluation: loss={test_loss:.4f} " +
                         " ".join(f"{k}={v:.4f}" for k, v in eval_metrics.items()))

        metrics.write_pipeline_event(task_name=task_name, run_id=context.run_id,
                                     event="evaluation_complete",
                                     fields={"test_loss": test_loss, **eval_metrics})

        md = {k: MetadataValue.float(float(v)) for k, v in eval_metrics.items()
              if isinstance(v, (int, float)) and not np.isnan(v)}
        md["test_loss"] = MetadataValue.float(float(test_loss))
        md["report"] = MetadataValue.path(str(report_path))
        return Output(value=eval_metrics, metadata=md)

    _eval_kwargs = dict(
        name=f"{prefix}_model_evaluation",
        required_resource_keys=_required,
        group_name=prefix,
        description="Scores the test split with the task's compute_metrics.",
    )
    _splits_in = {"dataset_splits": AssetIn(key=AssetKey(f"{prefix}_dataset_splits"))}

    if trains:
        @asset(ins={**_splits_in,
                    "trained_model_info": AssetIn(key=AssetKey(f"{prefix}_trained_model"))},
               **_eval_kwargs)
        def model_evaluation(context, dataset_splits, trained_model_info):
            return _evaluate(context, dataset_splits, trained_model_info)
    else:
        # eval_only: score an existing checkpoint (checkpoint_path / default location).
        @asset(ins=_splits_in, **_eval_kwargs)
        def model_evaluation(context, dataset_splits):
            return _evaluate(context, dataset_splits, None)

    # ── 6. Inference Results ─────────────────────────────────────────────────

    infer_ins = {"segmented_features": AssetIn(key=AssetKey(f"{prefix}_segmented_features"))}
    if trains:
        # Ordering edge only: in the full training job inference waits for the
        # fresh checkpoint. In the inference-only job trained_model is not
        # selected, so the existing checkpoint on disk is used.
        infer_ins["trained_model"] = AssetIn(key=AssetKey(f"{prefix}_trained_model"),
                                             dagster_type=Nothing)

    @asset(
        name=f"{prefix}_inference_results",
        ins=infer_ins,
        required_resource_keys=_required,
        group_name=prefix,
        description="Scores every segment; writes outputs/inference_results_<task>.{json,csv}.",
    )
    def inference_results(context, segmented_features):
        import torch

        res, task_config, task = _task_and_config(context)
        storage = context.resources.minio_storage
        metrics = context.resources.influxdb_metrics

        feature_root = segmented_features.get("feature_dir") or os.path.join(res.output_dir,
                                                                             "features")
        base_out = segmented_features.get("output_dir", res.output_dir)
        hop = float(segmented_features.get("hop_seconds")
                    or task_config.segment_duration * (1 - task_config.segment_overlap))
        seg_dur = float(task_config.segment_duration)
        is_transcription = task.task_type == "transcription"

        results: Dict[str, Dict[str, Any]] = {}
        influx_records = []
        metrics.write_pipeline_event(task_name=task_name, run_id=context.run_id,
                                     event="inference_start")

        if is_transcription:
            # ── Transcription path: transcribe each source recording directly,
            #    in transcription_chunk_seconds windows (Whisper-style models are
            #    far more efficient on long windows than on 1 s segments).
            import librosa

            files = segmented_features.get("files") or [
                str(p) for p in list_audio_files(segmented_features.get("input_dir",
                                                                        res.input_dir))]
            sr = int(task_config.sample_rate)
            chunk_s = float(task_config.transcription_chunk_seconds)
            task.warmup()
            context.log.info(f"Transcribing {len(files)} files in {chunk_s:.0f}s windows...")
            t_start = time.time()
            for fi, f in enumerate(files, start=1):
                y, _ = librosa.load(f, sr=sr, mono=True)
                step = max(1, int(chunk_s * sr))
                for ci, start in enumerate(range(0, max(1, len(y)), step)):
                    window = y[start:start + step]
                    t_offset = start / sr
                    pred = _normalize_prediction(task.transcribe(window, t_offset=t_offset))
                    key = f"{Path(f).parent.name}/{Path(f).stem}_chunk_{ci:03d}"
                    pred.update({"source": str(f), "start_sec": round(t_offset, 3),
                                 "end_sec": round(t_offset + len(window) / sr, 3)})
                    results[key] = pred
                    rec = task.format_influx_record(segment_id=key, prediction=pred,
                                                    audio_offset_sec=t_offset)
                    rec.setdefault("tags", {})["audio_source"] = Path(f).parent.name
                    influx_records.append(rec)
                elapsed = time.time() - t_start
                eta = elapsed / fi * (len(files) - fi)
                context.log.info(f"[{fi}/{len(files)}] {Path(f).name} — ETA ~{eta / 60:.1f} min")
        else:
            # ── Model path: every segment through build_model + postprocess.
            checkpoint_path = default_checkpoint_path(task_config)
            model, device = _build_loaded_model(context, task, checkpoint_path,
                                                required=False)
            feature_files = sorted(Path(feature_root).rglob("*.npy"))
            total = len(feature_files)
            context.log.info(f"Running inference on {total} segments...")
            log_every = max(1, total // 10)
            per_file = defaultdict(int)
            for idx, feat_path in enumerate(feature_files):
                rel = str(feat_path.relative_to(feature_root))
                stem, seg_idx = parse_segment_name(feat_path.stem)
                start = seg_idx * hop
                x = to_model_input(np.load(feat_path), device=device)
                with torch.no_grad():
                    out = model(x)
                out = out.logits if hasattr(out, "logits") else out
                pred = _normalize_prediction(task.postprocess(out))
                pred.update({"source": f"{feat_path.parent.name}/{stem}",
                             "segment_index": seg_idx, "start_sec": round(start, 3),
                             "end_sec": round(start + seg_dur, 3)})
                results[rel] = pred
                per_file[pred["source"]] += 1

                if (idx + 1) % log_every == 0 or idx == total - 1:
                    context.log.info(f"  scored {idx + 1}/{total} segments")

                rec = task.format_influx_record(segment_id=feat_path.stem, prediction=pred,
                                                audio_offset_sec=start)
                rec.setdefault("tags", {})["audio_source"] = feat_path.parent.name
                influx_records.append(rec)

        if results and not is_transcription:
            from collections import Counter

            counts = Counter(str(r.get("label", r.get("value", "?"))) for r in results.values())
            if len(counts) <= 20:
                context.log.info("Predicted: " + ", ".join(f"{k} ×{v}"
                                                           for k, v in counts.most_common()))
        if influx_records:
            metrics.write_points(measurement="audio_inference", records=influx_records,
                                 common_tags={"task_name": task_name, "run_id": context.run_id})
        metrics.write_pipeline_event(task_name=task_name, run_id=context.run_id,
                                     event="inference_complete",
                                     fields={"total_segments": len(results)})

        out_dir = Path(base_out) / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / f"inference_results_{prefix}.json"
        serializable = _jsonable(results)
        results_path.write_text(json.dumps(serializable, indent=2, default=str))
        csv_path = out_dir / f"inference_results_{prefix}.csv"
        try:
            _write_results_csv(csv_path, serializable)
        except Exception as e:  # noqa: BLE001
            context.log.warning(f"CSV write failed (non-fatal): {e}")
        context.log.info(f"Wrote results: {results_path} and {csv_path}")

        results_uri = _sync(context, task_config, storage.upload_task_file,
                            ArtifactType.OUTPUTS, results_path)
        return Output(
            value=serializable,
            metadata={
                "total_segments": MetadataValue.int(len(results)),
                "results_json": MetadataValue.path(str(results_path)),
                "results_csv": MetadataValue.path(str(csv_path)),
                "storage_uri": MetadataValue.text(results_uri or "local only"),
            },
        )

    _by_key = {
        "raw_audio_files": raw_audio_files,
        "segmented_features": segmented_features,
        "dataset_splits": dataset_splits,
        "trained_model": trained_model,
        "model_evaluation": model_evaluation,
        "inference_results": inference_results,
    }
    return [_by_key[k] for k in _ASSET_ORDER if k in wanted]
