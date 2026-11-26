"""Shared helpers for the integration tests."""

import os


def run_job(job_name, run_config=None):
    from audio_ops_dagster.definitions import build_definitions

    defs = build_definitions()
    result = defs.resolve_job_def(job_name).execute_in_process(run_config=run_config or {},
                                                               raise_on_error=True)
    assert result.success
    return defs, result


def train_example(project_dir):
    """Train the bundled example task inside ``project_dir`` (cwd must be it)."""
    assert os.getcwd() == str(project_dir)
    return run_job("example_classifier_full_training_job")
