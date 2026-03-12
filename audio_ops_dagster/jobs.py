from dagster import define_asset_job, AssetSelection

def build_audio_jobs(task_name: str):
    """Creates named jobs for a specific audio task."""
    prefix = task_name

    full_training_job = define_asset_job(
        name=f"{prefix}_full_training_job",
        selection=AssetSelection.groups(prefix),
        description=f"Full pipeline for {task_name}."
    )

    data_prep_job = define_asset_job(
        name=f"{prefix}_data_prep_job",
        selection=(
            AssetSelection.assets(f"{prefix}_raw_audio_files") |
            AssetSelection.assets(f"{prefix}_segmented_features") |
            AssetSelection.assets(f"{prefix}_dataset_splits")
        ),
        description=f"Feature extraction and splitting for {task_name}."
    )

    return [full_training_job, data_prep_job]