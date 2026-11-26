"""Per-task Dagster jobs, emitted only when the assets they select exist."""

from dagster import AssetSelection, define_asset_job

from audio_ops_dagster.capabilities import BATCH_INFER, EVALUATE, TRAIN


def build_audio_jobs(task_name: str, capabilities):
    """
    Create the named jobs for one task, gated by its capabilities:

      ``<task>_full_training_job``  everything in the task's group   (train)
      ``<task>_evaluation_job``     prep + evaluate a checkpoint      (evaluate, no train)
      ``<task>_data_prep_job``      raw → features → splits           (train or evaluate)
      ``<task>_inference_job``      raw → features → inference_results (batch_infer)

    A serve-only task yields no jobs.
    """
    prefix = task_name
    caps = frozenset(capabilities)
    jobs = []

    def _assets(*names):
        sel = AssetSelection.assets(f"{prefix}_{names[0]}")
        for n in names[1:]:
            sel = sel | AssetSelection.assets(f"{prefix}_{n}")
        return sel

    if TRAIN in caps:
        jobs.append(define_asset_job(
            name=f"{prefix}_full_training_job",
            selection=AssetSelection.groups(prefix),
            description=f"Full pipeline for {task_name}: features, splits, training"
                        f"{', evaluation' if EVALUATE in caps else ''}"
                        f"{', batch inference' if BATCH_INFER in caps else ''}.",
        ))
    elif EVALUATE in caps:
        jobs.append(define_asset_job(
            name=f"{prefix}_evaluation_job",
            selection=_assets("raw_audio_files", "segmented_features", "dataset_splits",
                              "model_evaluation"),
            description=f"Evaluate an existing {task_name} checkpoint on the test split.",
        ))

    if caps & {TRAIN, EVALUATE}:
        jobs.append(define_asset_job(
            name=f"{prefix}_data_prep_job",
            selection=_assets("raw_audio_files", "segmented_features", "dataset_splits"),
            description=f"Feature extraction and splitting for {task_name}.",
        ))

    if BATCH_INFER in caps:
        jobs.append(define_asset_job(
            name=f"{prefix}_inference_job",
            selection=_assets("raw_audio_files", "segmented_features", "inference_results"),
            description=(
                f"Score audio with a trained {task_name} model (no training). "
                f"Set input_dir/output_dir on {prefix}_raw_audio_files in the "
                f"Launchpad to score an arbitrary folder; results (JSON+CSV) are "
                f"written under <output_dir>/outputs."
            ),
        ))

    return jobs
