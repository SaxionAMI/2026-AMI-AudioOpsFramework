import torch
import pandas as pd
import numpy as np
import re
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class AudioDataset(Dataset):
    def __init__(self, csv_path, feature_dir=None, label_to_idx=None, is_sequence=False):
        self.df = pd.read_csv(csv_path)
        self.feature_dir = Path(feature_dir) if feature_dir else Path.cwd()
    
        self.is_sequence = is_sequence # Set this to True for STT
        
        # Only build label_to_idx for classification tasks
        if not is_sequence and label_to_idx is None:
            unique_labels = sorted(self.df['label'].unique())
            self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature = np.load(self.feature_dir / row['path'])
        if feature.ndim == 2:
            feature = np.expand_dims(feature, axis=0) 
        
        label = row['label'] if self.is_sequence else self.label_to_idx[row['label']]
        return torch.tensor(feature, dtype=torch.float32), label

    @staticmethod
    def create_splits_csv(feature_dir: str, output_dir: str, ratios=(0.7, 0.15, 0.15), strategy="random"):
        path = Path(feature_dir)
        data = []

        for npy_file in path.rglob("*.npy"):
            filename = npy_file.stem
            
            # THE FIX: Simply split at '_seg' or '_chunk' and take the left side.
            # 'appn0001_seg_000' -> 'appn0001'
            group_id = re.split(r'_seg|_chunk', filename)[0]
            
            data.append({
                "path": str(npy_file.relative_to(path)), 
                "label": npy_file.parent.name,
                "group_id": group_id
            })
        
        if not data:
            raise ValueError(f"No .npy files found in {feature_dir}.")
            
        df = pd.DataFrame(data)
        
        unique_groups = df['group_id'].nunique()
        print(f"DEBUG: Found {len(df)} segments across {unique_groups} unique source files.")

        if strategy == "group":
            try:
                # 1. Create a summary dataframe of just the unique files
                # This assumes all segments of a file share the same label
                group_df = df[['group_id', 'label']].drop_duplicates().reset_index(drop=True)
                
                # 2. Perform a STRATIFIED split on the summary groups
                train_groups, rem_groups = train_test_split(
                    group_df, 
                    train_size=ratios[0], 
                    stratify=group_df['label'], 
                    random_state=42
                )
                
                val_groups, test_groups = train_test_split(
                    rem_groups, 
                    train_size=0.5, 
                    stratify=rem_groups['label'], 
                    random_state=42
                )
                
                # 3. Map the split groups back to the raw segments
                train_df = df[df['group_id'].isin(train_groups['group_id'])]
                val_df = df[df['group_id'].isin(val_groups['group_id'])]
                test_df = df[df['group_id'].isin(test_groups['group_id'])]
                
            except ValueError as e:
                # If a class has too few groups to stratify, we catch it and fallback
                print(f"WARNING: Group stratification failed ({e}). Falling back to random split.")
                strategy = "random"

        if strategy != "group":
            # Stratified random split (fallback for non-chunked datasets)
            train_df, rem_df = train_test_split(df, train_size=ratios[0], stratify=df['label'], random_state=42)
            val_df, test_df = train_test_split(rem_df, train_size=0.5, stratify=rem_df['label'], random_state=42)

        # Save files
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(out / "train.csv", index=False)
        val_df.to_csv(out / "val.csv", index=False)
        test_df.to_csv(out / "test.csv", index=False)

        # Print the balancing stats so you can verify in Dagster logs
        print(f"Train Balance:\n{train_df['label'].value_counts(normalize=True)}")
        print(f"Val Balance:\n{val_df['label'].value_counts(normalize=True)}")

        return {
            "train": str(out / "train.csv"),
            "val": str(out / "val.csv"),
            "test": str(out / "test.csv")
        }
        
class InferenceDataset(Dataset):
    def __init__(self, feature_dir):
        self.path = Path(feature_dir)
        self.files = list(self.path.rglob("*.npy"))
        self.file_names = [f.name for f in self.files]

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        feature = np.load(self.files[idx])
        if feature.ndim == 2:
            feature = np.expand_dims(feature, axis=0)
        return torch.tensor(feature, dtype=torch.float32), self.file_names[idx]
