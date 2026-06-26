import os, tempfile
import pytest
pytest.importorskip("torch")
pytest.importorskip("sklearn")
import numpy as np, pandas as pd
from audio_ops_dagster.core.audio_dataset import AudioDataset


def test_group_split_no_leakage_and_stratified():
    root = tempfile.mkdtemp(); feat = os.path.join(root, "features"); out = os.path.join(root, "dataset")
    labels = ["class_a", "class_b", "class_c"]
    for lbl in labels:
        os.makedirs(os.path.join(feat, lbl))
        for rec in range(6):
            for seg in range(5):
                np.save(os.path.join(feat, lbl, f"{lbl}rec{rec}_seg_{seg:03d}.npy"),
                        np.random.rand(128, 87).astype(np.float32))
    paths = AudioDataset.create_splits_csv(feature_dir=feat, output_dir=out, strategy="group")
    dfs = {k: pd.read_csv(v) for k, v in paths.items()}
    g = {k: set(v["group_id"]) for k, v in dfs.items()}
    assert not (g["train"] & g["val"]) and not (g["train"] & g["test"]) and not (g["val"] & g["test"])
    for v in dfs.values():
        assert set(v["label"]) == set(labels)
