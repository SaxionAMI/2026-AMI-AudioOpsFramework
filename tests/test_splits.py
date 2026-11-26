import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from audio_ops_dagster.core.audio_dataset import AudioDataset  # noqa: E402


def _make_features(root, labels=("class_a", "class_b", "class_c"), recs=6, segs=5,
                   prefix=None):
    feat = os.path.join(root, "features")
    for lbl in labels:
        os.makedirs(os.path.join(feat, lbl), exist_ok=True)
        for rec in range(recs):
            name = prefix or f"{lbl}rec"
            for seg in range(segs):
                np.save(os.path.join(feat, lbl, f"{name}{rec}_seg_{seg:03d}.npy"),
                        np.random.rand(128, 32).astype(np.float32))
    return feat


def test_group_split_no_leakage_and_stratified(tmp_path):
    feat = _make_features(tmp_path)
    paths = AudioDataset.create_splits_csv(feature_dir=feat, output_dir=tmp_path / "ds",
                                           strategy="group")
    dfs = {k: pd.read_csv(v) for k, v in paths.items()}
    g = {k: set(v["group_id"]) for k, v in dfs.items()}
    assert not (g["train"] & g["val"]) and not (g["train"] & g["test"]) and not (g["val"] & g["test"])
    for v in dfs.values():
        assert set(v["label"]) == {"class_a", "class_b", "class_c"}


def test_same_filename_in_two_classes_is_two_groups(tmp_path):
    feat = _make_features(tmp_path, prefix="clip")   # identical stems in every class
    df = AudioDataset.scan_features(feat)
    assert df["group_id"].nunique() == 3 * 6


def test_splits_are_deterministic(tmp_path):
    feat = _make_features(tmp_path)
    a = AudioDataset.create_splits_csv(feat, tmp_path / "a", strategy="group", seed=7)
    b = AudioDataset.create_splits_csv(feat, tmp_path / "b", strategy="group", seed=7)
    for k in a:
        assert pd.read_csv(a[k]).equals(pd.read_csv(b[k]))


def test_too_few_recordings_is_a_clear_error(tmp_path):
    feat = _make_features(tmp_path, labels=("x",), recs=2, segs=2)
    with pytest.raises(ValueError, match="at least 3"):
        AudioDataset.create_splits_csv(feat, tmp_path / "ds", strategy="group")


def test_unknown_label_is_a_clear_error(tmp_path):
    feat = _make_features(tmp_path)
    paths = AudioDataset.create_splits_csv(feat, tmp_path / "ds", strategy="group")
    with pytest.raises(ValueError, match="not in class_labels"):
        AudioDataset(paths["train"], feature_dir=feat, label_to_idx={"class_a": 0})


def test_regression_targets_from_numeric_folders(tmp_path):
    feat = _make_features(tmp_path, labels=("0.25", "0.5", "1.0"))
    paths = AudioDataset.create_splits_csv(feat, tmp_path / "ds", strategy="group",
                                           stratify=False)
    ds = AudioDataset(paths["train"], feature_dir=feat, target_type="float")
    x, y = ds[0]
    assert tuple(x.shape) == (1, 128, 32)
    assert float(y) in (0.25, 0.5, 1.0)


def test_group_pattern_keeps_clips_of_one_match_together(tmp_path):
    feat = os.path.join(tmp_path, "features")
    for lbl in ("approval", "disapproval", "neutral"):
        os.makedirs(os.path.join(feat, lbl))
        for match in range(6):
            for clip in range(3):
                np.save(os.path.join(feat, lbl, f"match{match}_{clip:03d}_seg_000.npy"),
                        np.zeros((8, 8), np.float32))
    df = AudioDataset.scan_features(feat, group_pattern=r"^(.+)_\d+$")
    assert df["group_id"].nunique() == 6          # one group per match, across labels
    paths = AudioDataset.create_splits_csv(feat, tmp_path / "ds", strategy="group",
                                           group_pattern=r"^(.+)_\d+$")
    g = {k: set(pd.read_csv(v)["group_id"]) for k, v in paths.items()}
    assert not (g["train"] & g["val"]) and not (g["train"] & g["test"]) and not (g["val"] & g["test"])


def test_group_pattern_must_be_a_valid_regex():
    from audio_ops_dagster.core.config import TaskConfig

    with pytest.raises(ValueError, match="regular expression"):
        TaskConfig.from_dict({"group_pattern": "(unclosed"}, task_name="t")
