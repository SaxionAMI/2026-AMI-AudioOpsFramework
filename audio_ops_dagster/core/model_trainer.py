"""
Model Trainer (v2)
==================
Improved training pipeline with:
  - Mixed-precision (AMP) training
  - Cosine-annealing + warm-up LR schedule
  - Early stopping with patience
  - Per-batch AND per-epoch metrics streamed to InfluxDB
  - MLflow run management with artifact logging
  - Gradient clipping & accumulation
  - Class-weighted loss for imbalanced datasets
  - Reproducible seeding
  - Self-describing checkpoints: saves full model object so inference
    never needs to reconstruct architecture — works for any custom head.

InfluxDB adapter
  - Accepts both InfluxDBMetricsResource (Dagster) and InfluxWriter
    (standalone) — detects which one at runtime.
"""

from __future__ import annotations

import logging
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# InfluxDB adapter — works with both resource types                   #
# ------------------------------------------------------------------ #

class _InfluxAdapter:
    """
    Routes InfluxDB calls to whichever object was passed in:
      - InfluxDBMetricsResource (Dagster)  → write_training_metric()
      - InfluxWriter (standalone)          → write(Metrics.training_epoch())
      - None                               → no-op
    """

    def __init__(self, influx, run_id: str = ""):
        self._influx      = influx
        self._run_id      = run_id
        self._is_resource = hasattr(influx, "write_training_metric")
        self._is_writer   = hasattr(influx, "write") and not self._is_resource

    def log_epoch(self, task: str, model: str, epoch: int, total_epochs: int,
                  train_loss: float, val_loss: float, train_acc: float,
                  val_acc: float, lr: float, duration_s: float) -> None:
        if self._influx is None:
            return
        try:
            if self._is_resource:
                self._influx.write_training_metric(
                    task_name=task,
                    run_id=self._run_id,
                    epoch=epoch,
                    fields={
                        "train_loss":   train_loss,
                        "val_loss":     val_loss,
                        "train_acc":    train_acc,
                        "val_acc":      val_acc,
                        "learning_rate": lr,
                        "duration_s":   duration_s,
                        "total_epochs": total_epochs,
                        "progress_pct": 100.0 * epoch / max(total_epochs, 1),
                    },
                    tags={"model": model},
                )
            elif self._is_writer:
                from audio_ops_dagster.monitoring.influx_writer import Metrics
                self._influx.write(Metrics.training_epoch(
                    task=task, model=model, epoch=epoch,
                    total_epochs=total_epochs, train_loss=train_loss,
                    val_loss=val_loss, train_acc=train_acc, val_acc=val_acc,
                    learning_rate=lr, epoch_duration_s=duration_s,
                ))
        except Exception as exc:
            logger.warning("InfluxDB epoch log failed: %s", exc)

    def log_batch(self, task: str, model: str, epoch: int,
                  step: int, loss: float, lr: float) -> None:
        if self._influx is None:
            return
        try:
            if self._is_resource:
                self._influx.write_training_metric(
                    task_name=task,
                    run_id=self._run_id,
                    epoch=epoch,
                    fields={"batch_loss": loss, "learning_rate": lr, "step": step},
                    tags={"model": model, "granularity": "batch"},
                )
            elif self._is_writer:
                from audio_ops_dagster.monitoring.influx_writer import Metrics
                self._influx.write(Metrics.training_batch(
                    task=task, model=model, epoch=epoch,
                    step=step, loss=loss, learning_rate=lr,
                ))
        except Exception as exc:
            logger.warning("InfluxDB batch log failed: %s", exc)


# ------------------------------------------------------------------ #
# Configuration                                                       #
# ------------------------------------------------------------------ #

@dataclass
class TrainConfig:
    task_name: str
    model_name: str

    # Data
    data_dir: str = ""
    batch_size: int = 32
    num_workers: int = 4
    val_split: float = 0.2
    sample_rate: int = 16_000
    clip_duration_s: float = 3.0
    augment: bool = True

    # Optimisation
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    grad_accumulation_steps: int = 1
    mixed_precision: bool = True
    label_smoothing: float = 0.1

    # LR schedule
    warmup_epochs: int = 3
    lr_min: float = 1e-6

    # Early stopping
    early_stop_patience: int = 10
    early_stop_metric: str = "val_loss"

    # Logging
    log_every_n_batches: int = 10
    checkpoint_dir: str = "/app/checkpoints"
    mlflow_experiment: str = ""

    # Reproducibility
    seed: int = 42

    # Model metadata
    class_names: list[str] = field(default_factory=list)
    class_weights: Optional[list[float]] = None
    input_shape: Optional[list[int]] = None


# ------------------------------------------------------------------ #
# Trainer                                                             #
# ------------------------------------------------------------------ #

class ModelTrainer:
    """
    Generic audio classification trainer.

    Supports two calling conventions:

    New style:
        trainer = ModelTrainer(config, model, train_loader, val_loader, influx_writer)
        checkpoint_path = trainer.train()

    Old style (assets.py):
        trainer = ModelTrainer(task, task_config, checkpoint_dir=...,
                               influxdb_metrics=..., dagster_run_id=...)
        checkpoint_path = trainer.train(model=model, train_loader=..., val_loader=...)
    """

    def __init__(
        self,
        task_or_config,
        task_config=None,
        model=None,
        train_loader=None,
        val_loader=None,
        influx_writer=None,
        checkpoint_dir: str = "/app/checkpoints",
        influxdb_metrics=None,
        dagster_run_id: str = "",
    ):
        import torch

        if isinstance(task_or_config, TrainConfig):
            # New style
            self.cfg          = task_or_config
            self.model        = task_config
            self.train_loader = model
            self.val_loader   = train_loader
            raw_influx        = val_loader or influx_writer or influxdb_metrics
        else:
            # Old style — task_or_config may be a string or task object
            raw_task  = task_or_config
            task_name = (
                raw_task if isinstance(raw_task, str)
                else getattr(raw_task, "task_name", None)
                  or getattr(raw_task, "name", None)
                  or type(raw_task).__name__.lower().replace("task", "")
            )
            old_cfg           = task_config
            self.model        = model
            self.train_loader = train_loader
            self.val_loader   = val_loader
            raw_influx        = influxdb_metrics or influx_writer

            self.cfg = TrainConfig(
                task_name         = str(task_name),
                model_name        = str(getattr(old_cfg, "model_architecture", "resnet18")),
                data_dir          = str(getattr(old_cfg, "input_dir", "")),
                batch_size        = int(getattr(old_cfg, "batch_size", 32)),
                epochs            = int(getattr(old_cfg, "epochs", 10)),
                learning_rate     = float(getattr(old_cfg, "learning_rate", 1e-3)),
                checkpoint_dir    = str(checkpoint_dir),
                class_names       = list(getattr(old_cfg, "class_labels", [])),
                input_shape       = list(getattr(old_cfg, "input_shape", None) or []),
                mlflow_experiment = str(task_name),
            )

        self.dagster_run_id = str(dagster_run_id)
        self._influx        = _InfluxAdapter(raw_influx, run_id=self.dagster_run_id)
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._seed_everything(self.cfg.seed)
        logger.info("Trainer on device=%s task=%s model=%s",
                    self.device, self.cfg.task_name, self.cfg.model_name)

    def train(self, model=None, train_loader=None, val_loader=None) -> str:
        """Run training. Returns path to best checkpoint."""
        if model is not None:
            self.model = model
        if train_loader is not None:
            self.train_loader = train_loader
        if val_loader is not None:
            self.val_loader = val_loader
        return self._run_training()

    # ------------------------------------------------------------------ #
    # Core training loop                                                   #
    # ------------------------------------------------------------------ #

    def _run_training(self) -> str:
        import torch
        import torch.nn as nn
        import mlflow

        cfg   = self.cfg
        model = self.model.to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        def lr_lambda(epoch: int) -> float:
            if epoch < cfg.warmup_epochs:
                return (epoch + 1) / max(cfg.warmup_epochs, 1)
            progress = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
            cosine   = 0.5 * (1 + np.cos(np.pi * progress))
            return max(cfg.lr_min / cfg.learning_rate, cosine)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if cfg.class_weights:
            weights   = torch.tensor(cfg.class_weights, dtype=torch.float, device=self.device)
            criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        scaler = None
        if cfg.mixed_precision and self.device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed-precision (AMP) enabled")

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        _active = mlflow.active_run()
        if _active is None:
            mlflow.set_experiment(cfg.mlflow_experiment or cfg.task_name)
            _mlflow_ctx = mlflow.start_run(run_name=f"{cfg.model_name}_seed{cfg.seed}")
        else:
            _mlflow_ctx = nullcontext(_active)

        best_metric      = float("inf")
        best_epoch       = 0
        patience_counter = 0
        history          = []

        with _mlflow_ctx as run:
            run_id = run.info.run_id
            mlflow.log_params({
                "task":            cfg.task_name,
                "model":           cfg.model_name,
                "epochs":          cfg.epochs,
                "batch_size":      cfg.batch_size,
                "lr":              cfg.learning_rate,
                "weight_decay":    cfg.weight_decay,
                "warmup_epochs":   cfg.warmup_epochs,
                "label_smoothing": cfg.label_smoothing,
                "mixed_precision": cfg.mixed_precision,
                "device":          str(self.device),
                "seed":            cfg.seed,
                "class_names":     ",".join(cfg.class_names),
                "input_shape":     str(cfg.input_shape),
            })

            for epoch in range(cfg.epochs):
                epoch_start = time.time()
                current_lr  = scheduler.get_last_lr()[0] \
                              if hasattr(scheduler, "get_last_lr") else cfg.learning_rate

                train_loss, train_acc = self._train_epoch(
                    model, optimizer, criterion, scaler, epoch, current_lr
                )
                val_loss, val_acc, _ = self._validate(model, criterion)
                scheduler.step()
                epoch_duration = time.time() - epoch_start

                mlflow.log_metrics(
                    {"train_loss": train_loss, "train_acc": train_acc,
                     "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr},
                    step=epoch,
                )
                self._influx.log_epoch(
                    task=cfg.task_name, model=cfg.model_name,
                    epoch=epoch + 1, total_epochs=cfg.epochs,
                    train_loss=train_loss, val_loss=val_loss,
                    train_acc=train_acc, val_acc=val_acc,
                    lr=current_lr, duration_s=epoch_duration,
                )
                logger.info(
                    "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | "
                    "val_acc=%.3f | lr=%.2e | %.1fs",
                    epoch + 1, cfg.epochs, train_loss, val_loss,
                    val_acc, current_lr, epoch_duration,
                )
                history.append({
                    "epoch": epoch + 1, "train_loss": train_loss,
                    "val_loss": val_loss, "train_acc": train_acc,
                    "val_acc": val_acc, "lr": current_lr, "duration_s": epoch_duration,
                })

                monitor = val_loss if cfg.early_stop_metric == "val_loss" else -val_acc
                if monitor < best_metric:
                    best_metric      = monitor
                    best_epoch       = epoch + 1
                    patience_counter = 0
                    self._save_checkpoint(model, epoch, val_loss, val_acc, run_id)
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.early_stop_patience:
                        logger.info("Early stopping at epoch %d (best=%d)",
                                    epoch + 1, best_epoch)
                        break

            best_ckpt = os.path.join(cfg.checkpoint_dir, "best_model.pt")
            if os.path.exists(best_ckpt):
                try:
                    mlflow.log_artifact(best_ckpt, artifact_path="checkpoints")
                except Exception as exc:
                    logger.warning("MLflow artifact log failed: %s", exc)

        return best_ckpt

    # ------------------------------------------------------------------ #
    # Training / validation steps                                          #
    # ------------------------------------------------------------------ #

    def _train_epoch(self, model, optimizer, criterion, scaler,
                     epoch: int, current_lr: float) -> tuple[float, float]:
        import torch

        model.train()
        total_loss    = 0.0
        total_correct = 0
        total_samples = 0
        optimizer.zero_grad()

        for step, (inputs, targets) in enumerate(self.train_loader):
            inputs  = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss    = criterion(outputs, targets) / self.cfg.grad_accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(inputs)
                loss    = criterion(outputs, targets) / self.cfg.grad_accumulation_steps
                loss.backward()

            total_loss    += loss.item() * self.cfg.grad_accumulation_steps
            preds          = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += len(targets)

            if (step + 1) % self.cfg.grad_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                    optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % self.cfg.log_every_n_batches == 0:
                self._influx.log_batch(
                    task=self.cfg.task_name, model=self.cfg.model_name,
                    epoch=epoch + 1, step=step + 1,
                    loss=total_loss / (step + 1), lr=current_lr,
                )

        return (
            total_loss / max(len(self.train_loader), 1),
            total_correct / max(total_samples, 1),
        )

    def _validate(self, model, criterion) -> tuple[float, float, Any]:
        import torch

        model.eval()
        total_loss   = 0.0
        all_preds:   list[int] = []
        all_targets: list[int] = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs  = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                outputs = model(inputs)
                total_loss += criterion(outputs, targets).item()
                all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
                all_targets.extend(targets.cpu().tolist())

        avg_loss = total_loss / max(len(self.val_loader), 1)
        accuracy = sum(p == t for p, t in zip(all_preds, all_targets)) / max(len(all_targets), 1)

        n_classes = len(self.cfg.class_names) or (max(all_targets) + 1 if all_targets else 1)
        cm        = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(all_targets, all_preds):
            if 0 <= t < n_classes and 0 <= p < n_classes:
                cm[t][p] += 1

        return avg_loss, accuracy, cm

    def _save_checkpoint(self, model, epoch: int, val_loss: float,
                         val_acc: float, run_id: str) -> None:
        """
        Save a self-describing checkpoint.

        Saves the FULL MODEL OBJECT (not just state dict) so the inference
        service can load it with torch.load() regardless of what custom
        head architecture was used. Works for any task, any model variant.

        Also saves metadata alongside for inspection and future use.
        """
        import torch

        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.cfg.checkpoint_dir, "best_model.pt")

        # Attach metadata directly to the model object
        model.class_names  = self.cfg.class_names
        model.input_shape  = self.cfg.input_shape or [1, 128, 87]
        model.architecture = self.cfg.model_name
        model.task_name    = self.cfg.task_name

        checkpoint = {
            # Full model — load with torch.load(), call .eval(), done.
            # No architecture reconstruction needed.
            "model":              model,

            # State dict kept for transfer learning / inspection
            "model_state_dict":   model.state_dict(),

            # Metadata
            "model_architecture": self.cfg.model_name,
            "class_names":        self.cfg.class_names,
            "num_classes":        len(self.cfg.class_names),
            "input_shape":        self.cfg.input_shape or [1, 128, 87],
            "task_name":          self.cfg.task_name,
            "epoch":              epoch + 1,
            "val_loss":           val_loss,
            "val_acc":            val_acc,
            "mlflow_run_id":      run_id,
            "saved_at":           time.time(),
            "feature_config": {                          # ← ADD THIS
                "feature_type": self.cfg.feature_type,       # from task config
                "sample_rate": self.cfg.sample_rate,
                "n_mels": 128,
                "n_fft": 512,
                "hop_length": 160,
                "clip_duration_s": self.cfg.clip_duration_s,
            },
        }

        torch.save(checkpoint, path)
        logger.info(
            "Checkpoint saved: %s (epoch=%d val_loss=%.4f val_acc=%.3f "
            "arch=%s classes=%s input=%s)",
            path, epoch + 1, val_loss, val_acc,
            self.cfg.model_name, self.cfg.class_names, self.cfg.input_shape,
        )

    @staticmethod
    def _seed_everything(seed: int) -> None:
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ------------------------------------------------------------------ #
# Audio dataset with augmentation                                     #
# ------------------------------------------------------------------ #

class AudioDataset:
    """
    PyTorch Dataset for audio classification.
    Expects directory structure:
        data_dir/
            class_a/  file1.wav  file2.wav ...
            class_b/  ...
    """

    def __init__(
        self,
        data_dir: str,
        class_names: list[str],
        sample_rate: int = 16_000,
        clip_duration_s: float = 3.0,
        augment: bool = False,
        transform: Optional[Callable] = None,
    ):
        from pathlib import Path

        self.sr           = sample_rate
        self.clip_samples = int(sample_rate * clip_duration_s)
        self.augment      = augment
        self.transform    = transform
        self.class_names  = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

        self.samples: list[tuple[str, int]] = []
        root = Path(data_dir)
        for cls in class_names:
            for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"):
                for p in (root / cls).glob(ext):
                    self.samples.append((str(p), self.class_to_idx[cls]))

        logger.info("AudioDataset: %d samples, %d classes",
                    len(self.samples), len(class_names))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch
        import soundfile as sf

        path, label = self.samples[idx]
        audio, sr   = sf.read(path, dtype="float32", always_2d=False)

        if sr != self.sr:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
            except ImportError:
                pass

        if len(audio) < self.clip_samples:
            audio = np.pad(audio, (0, self.clip_samples - len(audio)))
        else:
            start = random.randint(0, len(audio) - self.clip_samples) \
                    if self.augment else (len(audio) - self.clip_samples) // 2
            audio = audio[start: start + self.clip_samples]

        if self.augment:
            audio = self._augment(audio)

        features = self._extract_features(audio)
        if self.transform:
            features = self.transform(features)

        return torch.from_numpy(features), label

    def _augment(self, audio: np.ndarray) -> np.ndarray:
        if random.random() < 0.4:
            audio = audio + random.uniform(0.001, 0.015) * \
                    np.random.randn(*audio.shape).astype(np.float32)
        if random.random() < 0.5:
            audio = audio * random.uniform(0.5, 1.5)
        if random.random() < 0.3:
            audio = np.roll(audio, int(random.uniform(-0.1, 0.1) * self.sr))
        if random.random() < 0.2:
            try:
                import librosa
                audio = librosa.effects.pitch_shift(
                    audio, sr=self.sr, n_steps=random.uniform(-2, 2)
                )
            except ImportError:
                pass
        return np.clip(audio, -1.0, 1.0)

    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Log-mel spectrogram, shape (1, n_mels, time)."""
        try:
            import librosa
            mel     = librosa.feature.melspectrogram(
                y=audio, sr=self.sr, n_mels=128, n_fft=512, hop_length=160
            )
            log_mel = librosa.power_to_db(mel, ref=np.max)
            return log_mel[np.newaxis, :, :].astype(np.float32)
        except ImportError:
            return audio[np.newaxis, :].astype(np.float32)


def build_data_loaders(cfg: TrainConfig) -> tuple:
    """Build train and validation DataLoaders from a directory."""
    import torch
    from torch.utils.data import DataLoader, Subset

    dataset = AudioDataset(
        data_dir=cfg.data_dir, class_names=cfg.class_names,
        sample_rate=cfg.sample_rate, clip_duration_s=cfg.clip_duration_s,
        augment=True,
    )

    indices = list(range(len(dataset)))
    random.Random(cfg.seed).shuffle(indices)
    split     = int(len(indices) * (1 - cfg.val_split))
    train_idx = indices[:split]
    val_idx   = indices[split:]

    val_dataset = AudioDataset(
        data_dir=cfg.data_dir, class_names=cfg.class_names,
        sample_rate=cfg.sample_rate, clip_duration_s=cfg.clip_duration_s,
        augment=False,
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader