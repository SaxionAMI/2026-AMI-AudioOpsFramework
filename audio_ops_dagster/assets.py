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
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from dagster import asset, AssetKey, AssetIn, Output, MetadataValue

from audio_ops_dagster.core.audio_processor import AudioProcessor
from audio_ops_dagster.core.audio_dataset import AudioDataset, InferenceDataset
from audio_ops_dagster.core.model_trainer import ModelTrainer
from audio_ops_dagster.core.storage import ArtifactType
from audio_ops_dagster.registry import get_task
from torch.utils.data import DataLoader


def build_audio_assets(task_name: str, resource_key: str):
    """Factory to create assets namespaced by task_name."""
    prefix = task_name
    _required = {resource_key, "minio_storage", "influxdb_metrics"}

    # ── 1. Raw Audio Files ───────────────────────────────────────────────────

    @asset(
        name=f"{prefix}_raw_audio_files",
        required_resource_keys=_required,
        group_name=prefix,
    )
    def raw_audio_files(context):
        config = getattr(context.resources, resource_key)
        metrics = context.resources.influxdb_metrics

        metrics.write_pipeline_event(
            task_name=task_name,
            run_id=context.run_id,
            event="raw_audio_scan_start",
        )

        Path(config.input_dir).mkdir(parents=True, exist_ok=True)
        files = glob.glob(f"{config.input_dir}/**/*.wav", recursive=True)
        files += glob.glob(f"{config.input_dir}/**/*.mp3", recursive=True)
        context.log.info(f"Found {len(files)} raw audio files in {config.input_dir}")

        metrics.write_pipeline_event(
            task_name=task_name,
            run_id=context.run_id,
            event="raw_audio_scan_complete",
            fields={"file_count": len(files)},
        )

        return files

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

        processor = AudioProcessor(feature_type=config.feature_type)
        feature_out = os.path.abspath(os.path.join(config.output_dir, "features"))
        stats = processor.process_directory(config.input_dir, feature_out)

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
            storage.upload_task_directory(
                task_name=task_config.effective_bucket_prefix(),
                artifact_type=ArtifactType.FEATURES,
                local_dir=feature_out,
            )

        return Output(
            value=stats,
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

        # Log split sizes to InfluxDB
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
                storage.upload_task_file(
                    task_name=bucket_prefix,
                    artifact_type=ArtifactType.DATASET,
                    local_path=csv_path,
                )

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
        is_seq = task_name in ("speech_to_text",)
        train_ds = AudioDataset(
            csv_path=dataset_splits["train"], feature_dir=feature_root, is_sequence=is_seq
        )
        val_ds = AudioDataset(
            csv_path=dataset_splits["val"], feature_dir=feature_root, is_sequence=is_seq
        )

        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size)

        task = get_task(config.task_name, task_config)
        input_shape = getattr(task_config, "input_shape", None)
        model = task.build_model(input_shape=input_shape)

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
            model_uri = storage.upload_task_file(
                task_name=task_config.effective_bucket_prefix(),
                artifact_type=ArtifactType.MODELS,
                local_path=checkpoint_path,
            )

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
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        feature_root = os.path.join(config.output_dir, "features")
        test_ds = AudioDataset(csv_path=dataset_splits["test"], feature_dir=feature_root)
        test_loader = DataLoader(test_ds, batch_size=config.batch_size)

        trainer = ModelTrainer(task, task_config)
        test_loss, eval_metrics = trainer.test(model=model, test_loader=test_loader)

        # Log evaluation metrics to InfluxDB
        metrics.write_pipeline_event(
            task_name=task_name,
            run_id=context.run_id,
            event="evaluation_complete",
            fields={"test_loss": test_loss, **eval_metrics},
        )

        return Output(value=eval_metrics)

    # ── 6. Inference Results (with InfluxDB per-segment logging) ─────────────

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

        task = get_task(config.task_name, task_config)
        input_shape = getattr(task_config, "input_shape", None)
        model = task.build_model(input_shape=input_shape)

        # Try to load best checkpoint if it exists
        checkpoint_path = os.path.join(config.output_dir, "checkpoints", "best_model.pt")
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            context.log.info(f"Loaded checkpoint: {checkpoint_path}")

        feature_root = os.path.join(config.output_dir, "features")
        all_feature_files = sorted(Path(feature_root).rglob("*.npy"))

        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        results = {}
        influx_records = []
        segment_duration = 1.0  # seconds per segment (from AudioProcessor default)

        metrics.write_pipeline_event(
            task_name=task_name, run_id=context.run_id, event="inference_start",
            fields={"segment_count": len(all_feature_files)},
        )

        context.log.info(f"Running inference on {len(all_feature_files)} segments...")

        with torch.no_grad():
            for seg_idx, feat_path in enumerate(all_feature_files):
                raw_feat = np.load(feat_path)
                x = torch.tensor(raw_feat, dtype=torch.float32)

                # --- NEW TASK-SPECIFIC SHAPING ---
                if "sentiment" in config.task_name:
                    # ResNet (Sentiment) MUST be 4D: [Batch, Channel, Height, Width]
                    while x.dim() < 4:
                        x = x.unsqueeze(0)
                
                elif "speech_to_text" in config.task_name:
                    # Wav2Vec2 (STT) MUST be 2D: [Batch, Length]
                    # If it's 1D [16000], make it 2D [1, 16000]
                    while x.dim() < 2:
                        x = x.unsqueeze(0)
                    # If it's 3D or 4D, flatten it back down to [Batch, Length]
                    if x.dim() > 2:
                        x = x.squeeze() 
                        if x.dim() == 1: # Don't accidentally squeeze the batch!
                            x = x.unsqueeze(0)
                # ---------------------------------

                x = x.to(device)
                logits = model(x)
                prediction = task.postprocess(logits)
                results[feat_path.name] = prediction

                # Write per-segment result to InfluxDB
                audio_offset = seg_idx * segment_duration
                influx_rec = task.format_influx_record(
                    segment_id=feat_path.stem,
                    prediction=prediction,
                    audio_offset_sec=audio_offset,
                )
                influx_rec["tags"]["audio_source"] = feat_path.parent.name
                influx_records.append(influx_rec)

        # Batch write all inference results to InfluxDB
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
            task_name=task_name, run_id=context.run_id, event="inference_complete",
            fields={"total_segments": len(results)},
        )

        # Save results JSON
        out_dir = Path(config.output_dir) / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / f"inference_results_{prefix}.json"

        # Serialize results (handle non-serializable types)
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

        results_uri = None
        if task_config.use_remote_storage and storage.enabled:
            results_uri = storage.upload_task_file(
                task_name=task_config.effective_bucket_prefix(),
                artifact_type=ArtifactType.OUTPUTS,
                local_path=results_path,
            )

        return Output(
            value=results,
            metadata={
                "total_files_processed": MetadataValue.int(len(results)),
                "minio_uri": MetadataValue.text(results_uri or "local only"),
            },
        )

    return [
        raw_audio_files,
        segmented_features,
        dataset_splits,
        trained_model,
        model_evaluation,
        inference_results,
    ]
