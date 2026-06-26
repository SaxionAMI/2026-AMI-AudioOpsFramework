from dagster import define_asset_job, AssetSelection

from audio_ops_dagster.capabilities import TRAIN, EVALUATE, BATCH_INFER


def build_audio_jobs(task_name: str, capabilities):
    """
    Creates named jobs for a specific audio task, gated by capabilities.

    A job is only emitted when the assets it selects actually exist:
      • full_training_job / data_prep_job  → require train or evaluate
      • inference_job                       → requires batch_infer
    A serve-only task yields no jobs.
    """
    prefix = task_name
    caps = frozenset(capabilities)
    produces_model = bool(caps & {TRAIN, EVALUATE})

    jobs = []

    # 1. Full Pipeline (Everything in the group)
    if produces_model:
        jobs.append(define_asset_job(
            name=f"{prefix}_full_training_job",
            selection=AssetSelection.groups(prefix),
            description=f"Full pipeline for {task_name} including training.",
        ))

    # 2. Data Preparation Only (No ML) — needs the split asset (train/evaluate)
    if produces_model:
        jobs.append(define_asset_job(
            name=f"{prefix}_data_prep_job",
            selection=(
                AssetSelection.assets(f"{prefix}_raw_audio_files") |
                AssetSelection.assets(f"{prefix}_segmented_features") |
                AssetSelection.assets(f"{prefix}_dataset_splits")
            ),
            description=f"Feature extraction and splitting for {task_name}.",
        ))

    # 3. Inference Only Job (The Fast Path) — needs the batch inference asset.
    # Skips dataset_splits, trained_model, and model_evaluation.
    if BATCH_INFER in caps:
        jobs.append(define_asset_job(
            name=f"{prefix}_inference_job",
            selection=(
                AssetSelection.assets(f"{prefix}_raw_audio_files") |
                AssetSelection.assets(f"{prefix}_segmented_features") |
                AssetSelection.assets(f"{prefix}_inference_results")
            ),
            description=(
                f"Run pre-trained model inference for {task_name}. "
                f"Set input_dir/output_dir on {prefix}_raw_audio_files in the "
                f"Launchpad to score an arbitrary folder; results (JSON+CSV) are "
                f"written under <output_dir>/outputs."
            ),
        ))

    return jobs