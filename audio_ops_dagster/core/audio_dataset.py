"""
audio_dataset.py — CSV + ``.npy`` datasets and leak-safe splits.

``AudioDataset.create_splits_csv`` scans a feature directory written by
``AudioProcessor`` and writes ``train/val/test.csv``. The **group** strategy
keeps every segment of one source recording in the same split (no leakage)
and stratifies the split over recordings so class balance is preserved.

Targets (``AudioDataset.target_type``):
  ``index``  class index from ``label_to_idx`` (classification / detection)
  ``float``  the label folder name parsed as a number (regression — name the
             folders by their target value, e.g. ``audio_input/loudness/0.75/``)
  ``raw``    the label string itself (sequence / transcription tasks)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from audio_ops_dagster._log import get_logger
from audio_ops_dagster.core.runtime import add_channel_dim, parse_segment_name

logger = get_logger(__name__)

TARGET_TYPES = ("index", "float", "raw")


def target_type_for(task_type: str) -> str:
    return {"regression": "float", "transcription": "raw"}.get(task_type, "index")


class AudioDataset(Dataset):
    def __init__(self, csv_path, feature_dir=None, label_to_idx=None, is_sequence=False,
                 target_type: Optional[str] = None):
        self.df = pd.read_csv(csv_path)
        self.feature_dir = Path(feature_dir) if feature_dir else Path.cwd()
        self.target_type = target_type or ("raw" if is_sequence else "index")
        if self.target_type not in TARGET_TYPES:
            raise ValueError(f"target_type must be one of {TARGET_TYPES}")
        self.is_sequence = self.target_type == "raw"

        if self.target_type == "index":
            if label_to_idx is None:
                unique_labels = sorted(self.df["label"].astype(str).unique())
                label_to_idx = {label: i for i, label in enumerate(unique_labels)}
            unknown = set(self.df["label"].astype(str)) - set(label_to_idx)
            if unknown:
                raise ValueError(
                    f"{csv_path}: labels {sorted(unknown)} are not in class_labels "
                    f"{list(label_to_idx)}. Add them to class_labels in config.yaml or "
                    f"remove those folders from the input directory."
                )
        elif self.target_type == "float":
            try:
                self.df["label"].astype(float)
            except ValueError:
                raise ValueError(
                    f"{csv_path}: regression targets come from numeric label folder names "
                    f"(e.g. audio_input/<task>/0.75/clip.wav); got {sorted(self.df['label'].unique())[:5]}"
                ) from None
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature = add_channel_dim(np.load(self.feature_dir / row["path"]))
        label = row["label"]
        if self.target_type == "index":
            target = self.label_to_idx[str(label)]
        elif self.target_type == "float":
            target = torch.tensor(float(label), dtype=torch.float32)
        else:
            target = label
        return torch.tensor(feature, dtype=torch.float32), target

    @staticmethod
    def scan_features(feature_dir: str, group_pattern: Optional[str] = None) -> pd.DataFrame:
        """One row per segment with its label and recording group.

        By default every source file is its own recording, scoped by label folder
        so two files with the same name in different classes are not merged.
        ``group_pattern`` is a regex searched in the file stem; its first capture
        group (or the whole match) is the recording id, shared across label
        folders. ``^(.+)_\\d+$`` groups ``match3_001.wav`` and
        ``match3_002.wav`` as ``match3``. Stems that don't match keep the default.
        """
        path = Path(feature_dir)
        rx = re.compile(group_pattern) if group_pattern else None
        rows = []
        for npy_file in sorted(path.rglob("*.npy")):
            stem, _ = parse_segment_name(npy_file.stem)
            label = npy_file.parent.name
            group_id = f"{label}/{stem}"
            if rx is not None:
                m = rx.search(stem)
                if m:
                    group_id = m.group(1) if m.groups() else m.group(0)
            rows.append({
                "path": str(npy_file.relative_to(path)),
                "label": label,
                "group_id": group_id,
            })
        return pd.DataFrame(rows, columns=["path", "label", "group_id"])

    @staticmethod
    def create_splits_csv(feature_dir: str, output_dir: str,
                          ratios: Sequence[float] = (0.7, 0.15, 0.15),
                          strategy: str = "random", seed: int = 42,
                          stratify: bool = True,
                          group_pattern: Optional[str] = None) -> Dict[str, str]:
        """Write train/val/test CSVs. Returns ``{"train": path, "val": ..., "test": ...}``.

        ``stratify`` should be False for regression (continuous targets).
        """
        from sklearn.model_selection import train_test_split

        df = AudioDataset.scan_features(feature_dir, group_pattern=group_pattern)
        if df.empty:
            raise ValueError(f"No .npy files found in {feature_dir}.")

        r_train, r_val, r_test = (float(r) for r in ratios)
        val_share = r_val / (r_val + r_test)
        logger.info("Found %d segments across %d source recordings.",
                    len(df), df["group_id"].nunique())

        def _split(frame, train_size, strat_col):
            strat = frame[strat_col] if (stratify and strat_col) else None
            try:
                return train_test_split(frame, train_size=train_size, stratify=strat,
                                        random_state=seed)
            except ValueError as exc:
                if strat is None:
                    raise
                logger.warning("Stratified split not possible (%s); using an unstratified split.",
                               exc)
                return train_test_split(frame, train_size=train_size, random_state=seed)

        unit = df
        if strategy == "group":
            unit = df[["group_id", "label"]].drop_duplicates("group_id").reset_index(drop=True)
        elif strategy != "random":
            raise ValueError(f"split_strategy must be 'group' or 'random', got {strategy!r}")

        if len(unit) < 3:
            raise ValueError(
                f"Need at least 3 {'recordings' if strategy == 'group' else 'segments'} "
                f"to make train/val/test splits; found {len(unit)} in {feature_dir}."
            )

        train_u, rem_u = _split(unit, r_train, "label")
        val_u, test_u = _split(rem_u, val_share, "label")

        if strategy == "group":
            train_df = df[df["group_id"].isin(train_u["group_id"])]
            val_df = df[df["group_id"].isin(val_u["group_id"])]
            test_df = df[df["group_id"].isin(test_u["group_id"])]
        else:
            train_df, val_df, test_df = train_u, val_u, test_u

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
            p = out / f"{name}.csv"
            frame.sort_values("path").to_csv(p, index=False)
            paths[name] = str(p)
            logger.info("%s: %d segments, class balance %s", name, len(frame),
                        frame["label"].value_counts(normalize=True).round(3).to_dict())
        return paths


class InferenceDataset(Dataset):
    """Every ``.npy`` under ``feature_dir`` (sorted), yielding ``(tensor, relpath)``."""

    def __init__(self, feature_dir):
        self.path = Path(feature_dir)
        self.files = sorted(self.path.rglob("*.npy"))
        self.file_names = [str(f.relative_to(self.path)) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        feature = add_channel_dim(np.load(self.files[idx]))
        return torch.tensor(feature, dtype=torch.float32), self.file_names[idx]
