"""
assets.py — Dagster asset definitions for the Audio Ops pipeline.

v2 changes:
  • Every asset writes time-series metrics to InfluxDB via the influxdb_metrics resource
  • Inference results are logged per-segment with task-specific schemas
  • Training metrics are streamed epoch-by-epoch
  • Pipeline lifecycle events are tracked
  • LLM post-processing is wired in when enabled
"""

import os
import glob
import json
import time
import csv as _csv
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from dagster import asset, AssetKey, AssetIn, Output, MetadataValue, Config

from audio_ops_dagster.core.audio_processor import AudioProcessor
from audio_ops_dagster.core.audio_dataset import AudioDataset, InferenceDataset
from audio_ops_dagster.core.model_trainer import ModelTrainer
from audio_ops_dagster.core.storage import ArtifactType
from audio_ops_dagster.registry import get_task
from audio_ops_dagster.capabilities import TRAIN, EVALUATE, BATCH_INFER
from torch.utils.data import DataLoader


class InferenceRunConfig(Config):
    """
    Per-run overrides for the inference path, set in the Dagster Launchpad.

    Launch a task's `<task>_inference_job` and set these once on the
    `<task>_raw_audio_files` asset to score an arbitrary folder:

        ops:
          <task>_raw_audio_files:
            config:
              input_dir: /data/new_batch        # audio to score
              output_dir: /data/new_batch_out   # where features + results go

    Leave them unset to use the task's configured dirs (normal pipeline runs).
    The trained checkpoint is always read from the task's configured
    output_dir/checkpoints (override with the INFERENCE_CHECKPOINT env var).
    """
    input_dir: Optional[str] = None
    output_dir: Optional[str] = None


def _write_results_csv(path, serializable: dict) -> None:
    """Flatten per-segment predictions (scalar fields only) into a CSV."""
    field_keys = []
    for v in serializable.values():
        if isinstance(v, dict):
            for k, val in v.items():
                if isinstance(val, (str, int, float, bool)) and k not in field_keys:
                    field_keys.append(k)
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["segment"] + field_keys)
        for seg, v in serializable.items():
            if isinstance(v, dict):
                w.writerow([seg] + [v.get(k, "") for k in field_keys])
            else:
                w.writerow([seg, v])


# Which assets each capability requires, including prerequisites.
# Logical (unprefixed) asset keys; the builder maps them to the namespaced
# closures below. `serve` is absent on purpose — it produces no Dagster assets.
_CAPABILITY_ASSETS = {
    TRAIN:       {"raw_audio_files", "segmented_features", "dataset_splits", "trained_model"},
    EVALUATE:    {"raw_audio_files", "segmented_features", "dataset_splits", "trained_model", "model_evaluation"},
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


def build_audio_assets(task_name: str, resource_key: str, capabilities):
    """
    Factory to create assets namespaced by task_name.

    Only the assets required by the task's declared `capabilities` are returned.
    A serve-only task returns an empty list — it contributes nothing to the
    Dagster graph and lives entirely in the inference service.
    """
    prefix = task_name
    _required = {resource_key, "minio_storage", "influxdb_metrics"}

    caps = frozenset(capabilities)
    wanted = set().union(*(_CAPABILITY_ASSETS.get(c, set()) for c in caps)) if caps else set()
    if not wanted:
        return []

    # ── 1. Raw Audio Files ───────────────────────────────────────────────────

    @asset(
        name=f"{prefix}_raw_audio_files",
        required_resource_keys=_required,
        group_name=prefix,
    )
    def raw_audio_files(context, config: InferenceRunConfig):
        res = getattr(context.resources, resource_key)
        metrics = context.resources.influxdb_metrics

        # Per-run overrides (Launchpad) fall back to the task's configured dirs.
        input_dir = config.input_dir or res.input_dir
        output_dir = config.output_dir or res.output_dir

        metrics.write_pipeline_event(
            task_name=task_name,
            run_id=context.run_id,
            event="raw_audio_scan_start",
        )

        Path(input_dir).mkdir(parents=True, exist_ok=True)
        files = glob.glob(f"{input_dir}/**/*.wav", recursive=True)
        files += glob.glob(f"{input_dir}/**/*.mp3", recursive=True)
        context.log.info(f"Found {len(files)} raw audio files in {input_dir}")

        metrics.write_pipeline_event(
            task_name=task_name,
            run_id=context.run_id,
            event="raw_audio_scan_complete",
            fields={"file_count": len(files)},
        )

        # Thread resolved dirs downstream so the whole inference path honors the
        # per-run override without configuring every asset.
        return {"files": files, "input_dir": input_dir, "output_dir": output_dir}

    # ── 2. Segmented Features ────────────────────────────────────────────────

    @asset(
        name=f"{prefix}_segmented_features",
        ins={"raw_audio_files": AssetIn(key=AssetKey(f"{prefix}_raw_audio_files"))},
        required_resource_keys=_required,
        group_name=prefix,
    )
    def segmented_features(context, raw_audio_files):
        config = getattr(context.resources, resource_key)
        storage = context.resources.minio_storage
        metrics = context.resources.influxdb_metrics
        task_config = config.to_task_config()

        t0 = time.time()
        metrics.write_pipeline_event(
            task_name=task_name, run_id=context.run_id, event="feature_extraction_start",
        )

        # Honor per-run overrides threaded from raw_audio_files; fall back to
        # the task's configured dirs for normal pipeline runs.
        if isinstance(raw_audio_files, dict):
            input_dir = raw_audio_files.get("input_dir", config.input_dir)
            base_out = raw_audio_files.get("output_dir", config.output_dir)
        else:
            input_dir, base_out = config.input_dir, config.output_dir

        processor = AudioProcessor(feature_type=config.feature_type)
        feature_out = os.path.abspath(os.path.join(base_out, "features"))
        stats = processor.process_directory(input_dir, feature_out)

        elapsed = time.time() - t0
        metrics.write_pipeline_event(
            task_name=task_name,
            run_id=context.run_id,
            event="feature_extraction_complete",
            fields={
                "total_files": stats.get("total_files", 0),
                "total_segments": stats.get("total_segments", 0),
                "duration_sec": round(elapsed, 2),
            },
        )

        if task_config.use_remote_storage and storage.enabled:
            try:
                storage.upload_task_directory(
                    task_name=task_config.effective_bucket_prefix(),
                    artifact_type=ArtifactType.FEATURES,
                    local_dir=feature_out,
                )
            except Exception as e:
                context.log.warning(f"MinIO upload failed (non-fatal): {e}")

        return Output(
            value={**stats, "feature_dir": feature_out, "output_dir": base_out},
            metadata={
                "feature_dir": MetadataValue.path(feature_out),
                "total_segments": MetadataValue.int(stats.get("total_segments", 0)),
            },
        )

    # ── 3. Dataset Splits ────────────────────────────────────────────────────

    @asset(
        name=f"{prefix}_dataset_splits",
        ins={"segmented_features": AssetIn(key=AssetKey(f"{prefix}_segmented_features"))},
        required_resource_keys=_required,
        group_name=prefix,
    )
    def dataset_splits(context, segmented_features):
        config = getattr(context.resources, resource_key)
        storage = context.resources.minio_storage
        metrics = context.resources.influxdb_metrics
        task_config = config.to_task_config()

        feature_dir = os.path.join(config.output_dir, "features")
        dataset_dir = os.path.join(config.output_dir, "dataset")
        strategy = getattr(config, "split_strategy", "random")

        split_paths = AudioDataset.create_splits_csv(
            feature_dir=feature_dir,
            output_dir=dataset_dir,
            strategy=strategy,
        )

        for split_name, csv_path in split_paths.items():
            import pandas as pd
            df = pd.read_csv(csv_path)
            metrics.write_pipeline_event(
                task_name=task_name,
                run_id=context.run_id,
                event=f"dataset_split_{split_name}",
                fields={"sample_count": len(df)},
            )

        if task_config.use_remote_storage and storage.enabled:
            bucket_prefix = task_config.effective_bucket_prefix()
            for split_name, csv_path in split_paths.items():
                try:
                    storage.upload_task_file(
                        task_name=bucket_prefix,
                        artifact_type=ArtifactType.DATASET,
                        local_path=csv_path,
                    )
                except Exception as e:
                    context.log.warning(f"MinIO upload failed (non-fatal): {e}")

        return Output(value=split_paths)

    # ── 4. Trained Model ─────────────────────────────────────────────────────

    @asset(
        name=f"{prefix}_trained_model",
        ins={"dataset_splits": AssetIn(key=AssetKey(f"{prefix}_dataset_splits"))},
        required_resource_keys=_required,
        group_name=prefix,
    )
    def trained_model(context, dataset_splits):
        import mlflow
        import torch

        config = getattr(context.resources, resource_key)
        storage = context.resources.minio_storage
        metrics = context.resources.influxdb_metrics
        task_config = config.to_task_config()

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment(f"Audio_Ops_{prefix.capitalize()}")

        feature_root = os.path.join(config.output_dir, "features")

        task = get_task(config.task_name, task_config)
        input_shape = getattr(task_config, "input_shape", None)
        model = task.build_model(input_shape=input_shape)

        # Generic: a task is "sequence" when it declares transcription, not by name.
        is_seq = getattr(task, "TASK_TYPE", "classification") == "transcription"
        # Generic: derive the label->index map once from the task's declared
        # class_labels so train/val/test share an identical, order-stable mapping
        # that also matches what postprocess() uses at inference time. Falls back
        # to per-CSV inference when class_labels aren't provided.
        label_to_idx = None
        if not is_seq and getattr(task_config, "class_labels", None):
            label_to_idx = {label: i for i, label in enumerate(task_config.class_labels)}

        train_ds = AudioDataset(
            csv_path=dataset_splits["train"], feature_dir=feature_root,
            label_to_idx=label_to_idx, is_sequence=is_seq,
        )
        val_ds = AudioDataset(
            csv_path=dataset_splits["val"], feature_dir=feature_root,
            label_to_idx=label_to_idx, is_sequence=is_seq,
        )

        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size)

        checkpoint_dir = os.path.abspath(os.path.join(config.output_dir, "checkpoints"))
        run_name = f"{prefix.upper()}_{datetime.now().strftime('%H%M%S')}"

        metrics.write_pipeline_event(
            task_name=task_name, run_id=context.run_id, event="training_start",
            fields={"epochs": config.epochs, "batch_size": config.batch_size},
        )

        with mlflow.start_run(run_name=run_name):
            trainer = ModelTrainer(
                task, task_config, checkpoint_dir=checkpoint_dir,
                influxdb_metrics=metrics, dagster_run_id=context.run_id,
            )
            checkpoint_path = trainer.train(
                model=model, train_loader=train_loader, val_loader=val_loader
            )

        metrics.write_pipeline_event(
            task_name=task_name, run_id=context.run_id, event="training_complete",
        )

        model_uri = None
        if task_config.use_remote_storage and storage.enabled:
            try:
                model_uri = storage.upload_task_file(
                    task_name=task_config.effective_bucket_prefix(),
                    artifact_type=ArtifactType.MODELS,
                    local_path=checkpoint_path,
                )
            except Exception as e:
                context.log.warning(f"MinIO upload failed (non-fatal): {e}")

        return Output(value={"checkpoint_path": checkpoint_path, "model_uri": model_uri})

    # ── 5. Model Evaluation ──────────────────────────────────────────────────

    @asset(
        name=f"{prefix}_model_evaluation",
        ins={
            "trained_model_info": AssetIn(key=AssetKey(f"{prefix}_trained_model")),
            "dataset_splits": AssetIn(key=AssetKey(f"{prefix}_dataset_splits")),
        },
        required_resource_keys=_required,
        group_name=prefix,
    )
    def model_evaluation(context, trained_model_info, dataset_splits):
        import mlflow
        import torch

        config = getattr(context.resources, resource_key)
        metrics = context.resources.influxdb_metrics
        task_config = config.to_task_config()

        checkpoint_path = trained_model_info["checkpoint_path"]
        task = get_task(config.task_name, task_config)
        input_shape = getattr(task_config, "input_shape", None)
        model = task.build_model(input_shape=input_shape)
        _ckpt = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(_ckpt["model_state_dict"] if isinstance(_ckpt, dict) and "model_state_dict" in _ckpt else _ckpt)

        is_seq = getattr(task, "TASK_TYPE", "classification") == "transcription"
        label_to_idx = None
        if not is_seq and getattr(task_config, "class_labels", None):
            label_to_idx = {label: i for i, label in enumerate(task_config.class_labels)}

        feature_root = os.path.join(config.output_dir, "features")
        test_ds = AudioDataset(
            csv_path=dataset_splits["test"], feature_dir=feature_root,
            label_to_idx=label_to_idx, is_sequence=is_seq,
        )
        test_loader = DataLoader(test_ds, batch_size=config.batch_size)

        trainer = ModelTrainer(task, task_config)
        test_loss, eval_metrics = trainer.test(model=model, test_loader=test_loader)

        metrics.write_pipeline_event(
            task_name=task_name,
            run_id=context.run_id,
            event="evaluation_complete",
            fields={"test_loss": test_loss, **eval_metrics},
        )

        return Output(value=eval_metrics)

    # ── 6. Inference Results ─────────────────────────────────────────────────

    @asset(
        name=f"{prefix}_inference_results",
        ins={"segmented_features": AssetIn(key=AssetKey(f"{prefix}_segmented_features"))},
        required_resource_keys=_required,
        group_name=prefix,
    )
    def inference_results(context, segmented_features):
        import torch

        config = getattr(context.resources, resource_key)
        storage = context.resources.minio_storage
        metrics = context.resources.influxdb_metrics
        task_config = config.to_task_config()

        # Resolve dirs: per-run overrides thread through segmented_features.
        # Features are read from where they were just extracted (the run's
        # output_dir); the checkpoint comes from the task's *trained* location.
        if isinstance(segmented_features, dict):
            feature_root = segmented_features.get("feature_dir") or os.path.join(config.output_dir, "features")
            base_out = segmented_features.get("output_dir", config.output_dir)
        else:
            feature_root = os.path.join(config.output_dir, "features")
            base_out = config.output_dir

        task = get_task(config.task_name, task_config)
        input_shape = getattr(task_config, "input_shape", None)
        model = task.build_model(input_shape=input_shape)

        is_transcription = getattr(task, "TASK_TYPE", "") == "transcription"

        if not is_transcription:
            checkpoint_path = os.environ.get("INFERENCE_CHECKPOINT") or os.path.join(
                config.output_dir, "checkpoints", "best_model.pt"
            )
            if os.path.exists(checkpoint_path):
                _ckpt = torch.load(checkpoint_path, weights_only=True)
                model.load_state_dict(
                    _ckpt["model_state_dict"] if isinstance(_ckpt, dict) and "model_state_dict" in _ckpt else _ckpt
                )
                context.log.info(f"Loaded checkpoint: {checkpoint_path}")
            else:
                context.log.warning(f"No checkpoint at {checkpoint_path} — using an untrained model.")
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

        all_feature_files = sorted(Path(feature_root).rglob("*.npy"))
        total_segments = len(all_feature_files)
        segment_duration = 1.0  # seconds per segment

        results = {}
        influx_records = []

        metrics.write_pipeline_event(
            task_name=task_name,
            run_id=context.run_id,
            event="inference_start",
            fields={"segment_count": total_segments},
        )

        context.log.info(f"Running inference on {total_segments} segments...")

        if is_transcription:
            # ── Transcription path ───────────────────────────────────────────
            # Whisper is highly inefficient on 1s chunks — each model call has
            # fixed overhead regardless of audio length. Concatenate segments
            # into 30s chunks (Whisper's full receptive field) to reduce model
            # calls from N to N/30, cutting runtime by ~20-25x.
            CHUNK_SECONDS = 30
            all_feature_files_list = list(all_feature_files)
            chunks = [
                all_feature_files_list[i:i + CHUNK_SECONDS]
                for i in range(0, total_segments, CHUNK_SECONDS)
            ]
            total_chunks = len(chunks)

            context.log.info(
                f"Transcription: concatenating {total_segments} x 1s segments "
                f"into {total_chunks} x {CHUNK_SECONDS}s chunks for efficient inference."
            )

            # Warm up model before loop so first-chunk timing isn't skewed
            _ = task._backend._fw_model
            context.log.info("Whisper model loaded, starting transcription...")

            t_start = time.time()

            for chunk_idx, chunk_files in enumerate(chunks):
                audio_arrays = [np.load(f).astype(np.float32) for f in chunk_files]
                combined_audio = np.concatenate(audio_arrays)
                t_offset = chunk_idx * CHUNK_SECONDS

                prediction = task._backend.transcribe(combined_audio, t_offset=t_offset)

                # Progress logging every chunk
                elapsed = time.time() - t_start
                chunks_done = chunk_idx + 1
                rate = chunks_done / elapsed if elapsed > 0 else 0
                eta_s = (total_chunks - chunks_done) / rate if rate > 0 else 0
                eta_m = round(eta_s / 60, 1)
                pct = round(chunks_done / total_chunks * 100, 1)
                preview = prediction.get("transcription", "")[:80]

                context.log.info(
                    f"[{chunks_done}/{total_chunks}] {pct}% — "
                    f"ETA ~{eta_m}min — \"{preview}\""
                )

                # Map result back to each individual segment file
                for fi, feat_path in enumerate(chunk_files):
                    seg_offset = t_offset + fi * segment_duration
                    results[feat_path.name] = prediction
                    influx_rec = task.format_influx_record(
                        segment_id=feat_path.stem,
                        prediction=prediction,
                        audio_offset_sec=seg_offset,
                    )
                    influx_rec["tags"]["audio_source"] = feat_path.parent.name
                    influx_records.append(influx_rec)

        else:
            # ── Classifier path ──────────────────────────────────────────────
            log_every = max(1, total_segments // 20)  # log every 5%

            for seg_idx, feat_path in enumerate(all_feature_files):
                raw_feat = np.load(feat_path)
                audio_offset = seg_idx * segment_duration

                x = torch.tensor(raw_feat, dtype=torch.float32)
                while x.dim() < 4:
                    x = x.unsqueeze(0)
                x = x.to(device)
                with torch.no_grad():
                    logits = model(x)
                prediction = task.postprocess(logits)

                results[feat_path.name] = prediction

                if seg_idx % log_every == 0 or seg_idx == total_segments - 1:
                    pct = round((seg_idx + 1) / total_segments * 100, 1)
                    context.log.info(
                        f"[{seg_idx + 1}/{total_segments}] {pct}% — "
                        f"label: {prediction.get('label', '?')} "
                        f"({round(prediction.get('confidence', 0) * 100, 1)}%)"
                    )

                influx_rec = task.format_influx_record(
                    segment_id=feat_path.stem,
                    prediction=prediction,
                    audio_offset_sec=audio_offset,
                )
                influx_rec["tags"]["audio_source"] = feat_path.parent.name
                influx_records.append(influx_rec)

        # ── Write to InfluxDB ────────────────────────────────────────────────
        if influx_records:
            metrics.write_points(
                measurement="audio_inference",
                records=influx_records,
                common_tags={
                    "task_name": task_name,
                    "run_id": context.run_id,
                },
            )

        metrics.write_pipeline_event(
            task_name=task_name,
            run_id=context.run_id,
            event="inference_complete",
            fields={"total_segments": len(results)},
        )

        # ── Save results (JSON + CSV) ────────────────────────────────────────
        out_dir = Path(base_out) / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / f"inference_results_{prefix}.json"

        serializable = {}
        for k, v in results.items():
            if isinstance(v, dict):
                serializable[k] = {
                    kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                    for kk, vv in v.items()
                }
            else:
                serializable[k] = str(v)
        results_path.write_text(json.dumps(serializable, indent=2, default=str))

        csv_path = out_dir / f"inference_results_{prefix}.csv"
        try:
            _write_results_csv(csv_path, serializable)
        except Exception as e:
            context.log.warning(f"CSV write failed (non-fatal): {e}")

        context.log.info(f"Wrote results: {results_path} and {csv_path}")

        results_uri = None
        if task_config.use_remote_storage and storage.enabled:
            try:
                results_uri = storage.upload_task_file(
                    task_name=task_config.effective_bucket_prefix(),
                    artifact_type=ArtifactType.OUTPUTS,
                    local_path=results_path,
                )
            except Exception as e:
                context.log.warning(f"MinIO upload failed (non-fatal): {e}")

        return Output(
            value=results,
            metadata={
                "total_files_processed": MetadataValue.int(len(results)),
                "minio_uri": MetadataValue.text(results_uri or "local only"),
            },
        )

    # Map logical keys → the namespaced asset closures defined above, then
    # return only the ones the declared capabilities require, in canonical order.
    _by_key = {
        "raw_audio_files": raw_audio_files,
        "segmented_features": segmented_features,
        "dataset_splits": dataset_splits,
        "trained_model": trained_model,
        "model_evaluation": model_evaluation,
        "inference_results": inference_results,
    }
    return [_by_key[k] for k in _ASSET_ORDER if k in wanted]