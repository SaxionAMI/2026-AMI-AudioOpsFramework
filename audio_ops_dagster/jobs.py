from dagster import define_asset_job, AssetSelection

def build_audio_jobs(task_name: str):
    """Creates named jobs for a specific audio task."""
    prefix = task_name

    # 1. Full Pipeline (Everything)
    full_training_job = define_asset_job(
        name=f"{prefix}_full_training_job",
        selection=AssetSelection.groups(prefix),
        description=f"Full pipeline for {task_name} including training."
    )

    # 2. Data Preparation Only (No ML)
    data_prep_job = define_asset_job(
        name=f"{prefix}_data_prep_job",
        selection=(
            AssetSelection.assets(f"{prefix}_raw_audio_files") |
            AssetSelection.assets(f"{prefix}_segmented_features") |
            AssetSelection.assets(f"{prefix}_dataset_splits")
        ),
        description=f"Feature extraction and splitting for {task_name}."
    )

    # 3. Inference Only Job (The Fast Path)
    # This skips dataset_splits, trained_model, and model_evaluation
    inference_job = define_asset_job(
        name=f"{prefix}_inference_job",
        selection=(
            AssetSelection.assets(f"{prefix}_raw_audio_files") |
            AssetSelection.assets(f"{prefix}_segmented_features") |
            AssetSelection.assets(f"{prefix}_inference_results")
        ),
        description=f"Run pre-trained model inference on input audio for {task_name}."
    )

    return [full_training_job, data_prep_job, inference_job]