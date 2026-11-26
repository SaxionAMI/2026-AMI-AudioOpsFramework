"""
model_trainer.py — the training loop, driven by the task contract.

* loss from ``task.compute_loss`` (tensor or ``(loss, outputs)``), so
  classification, regression and multi-label detection all train correctly
* optimizer from ``task.get_optimizer`` (``optimizer``/``learning_rate``/``weight_decay``)
* warm-up + cosine LR schedule, gradient clipping and accumulation, AMP on CUDA
* early stopping on ``val_loss`` or ``val_acc``
* resumable runs (``last_checkpoint.pt``) and a weights-only ``best_model.pt``
* metrics to the Dagster UI (live log lines + AssetObservations), MLflow and
  the metrics resource (InfluxDB or no-op) — all optional

Every knob comes from :class:`~audio_ops_dagster.core.config.TaskConfig`::

    trainer = ModelTrainer(task, checkpoint_dir="storage/t/checkpoints")
    best = trainer.train(model, train_loader, val_loader)
    loss, metrics = trainer.test(model, test_loader)
"""

from __future__ import annotations

import os
import random
import time
from typing import Tuple

import numpy as np

from audio_ops_dagster._log import get_logger
from audio_ops_dagster.core.base_task import effective_task_type
from audio_ops_dagster.core.runtime import resolve_device

logger = get_logger(__name__)


def _fmt_eta(seconds: float) -> str:
    """Human-friendly duration: 45s, 3m 12s, 1h 04m."""
    if seconds is None or seconds != seconds or seconds < 0:
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h {rem // 60:02d}m"


def _finite(v: float) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v))


class ModelTrainer:
    """Train / evaluate a task's model. See the module docstring."""

    def __init__(self, task, config=None, *, checkpoint_dir: str = "checkpoints",
                 metrics=None, run_id: str = "", context=None):
        self.task = task
        self.cfg = config if config is not None else task.config
        self.checkpoint_dir = str(checkpoint_dir)
        self.metrics = metrics            # InfluxDBMetricsResource / NullMetricsResource / None
        self.run_id = str(run_id)
        self.task_type = effective_task_type(task)
        self.device = resolve_device()
        self.best_epoch = 0
        # Dagster context (optional): live log lines + per-epoch observations.
        self._ctx = context
        self._log = context.log.info if context is not None and hasattr(context, "log") \
            else logger.info
        if self.cfg.early_stop_metric == "val_acc" and self.task_type != "classification":
            raise ValueError("early_stop_metric 'val_acc' needs a classification task; "
                             "use early_stop_metric: val_loss.")
        _seed_everything(int(self.cfg.seed))

    # ── Public API ───────────────────────────────────────────────────────────

    def train(self, model, train_loader, val_loader=None) -> str:
        """Train ``model``; return the path to ``best_model.pt``."""
        if train_loader is None or len(train_loader) == 0:
            raise ValueError(f"No training batches for '{self.cfg.task_name}'. Add more audio "
                             f"or lower batch_size (the train split is empty).")
        return self._run(model.to(self.device), train_loader, val_loader)

    def test(self, model, loader) -> Tuple[float, dict]:
        """Score ``loader``. Returns ``(avg_loss, task.compute_metrics(y_true, y_pred))``."""
        import torch

        model.to(self.device).eval()
        outs, targets, total, n = [], [], 0.0, 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                loss, out = self._step(model, x, y.to(self.device) if hasattr(y, "to") else y)
                out = self._logits(out if out is not None else model(x))
                total += float(loss.item())
                n += 1
                outs.append(out.detach().float().cpu().numpy())
                targets.append(np.atleast_1d(y.cpu().numpy() if hasattr(y, "cpu") else y))
        y_pred = np.concatenate(outs) if outs else np.zeros((0, 1))
        y_true = np.concatenate(targets) if targets else np.zeros((0,))
        try:
            metrics = self.task.compute_metrics(y_true, y_pred)
        except Exception as exc:  # noqa: BLE001
            logger.warning("compute_metrics failed (%s) — reporting accuracy only", exc)
            metrics = {"accuracy": float((y_pred.argmax(1) == y_true).mean())
                       if y_pred.ndim == 2 and len(y_true) else float("nan")}
        return total / max(n, 1), {k: float(v) for k, v in metrics.items()}

    # ── Steps ────────────────────────────────────────────────────────────────

    def _step(self, model, x, y):
        """``(loss, outputs | None)`` from the task's compute_loss."""
        result = self.task.compute_loss(model, (x, y), self.device)
        if isinstance(result, tuple):
            return result[0], (result[1] if len(result) > 1 else None)
        return result, None

    @staticmethod
    def _logits(out):
        return out.logits if hasattr(out, "logits") else out

    def _correct(self, out, y) -> Tuple[int, int]:
        """(correct, total) for class-index targets, else (0, 0)."""
        import torch

        if out is None or self.task_type != "classification":
            return 0, 0
        logits = self._logits(out)
        if not isinstance(y, torch.Tensor) or y.dim() != 1 or logits.dim() != 2:
            return 0, 0
        return int((logits.argmax(1) == y.to(logits.device)).sum()), int(y.numel())

    def _run(self, model, train_loader, val_loader) -> str:
        import torch

        from audio_ops_dagster.core.tracking import Tracker

        cfg = self.cfg
        optimizer = self.task.get_optimizer(model)
        base_lr = float(cfg.learning_rate)

        def lr_lambda(epoch: int) -> float:
            if epoch < cfg.warmup_epochs:
                return (epoch + 1) / max(cfg.warmup_epochs, 1)
            progress = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
            return max(cfg.lr_min / base_lr, 0.5 * (1 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = (torch.amp.GradScaler("cuda")
                  if cfg.mixed_precision and self.device.type == "cuda" else None)

        best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        start, best_metric, best_epoch, patience = self._resume(model, optimizer, scheduler)
        if start >= cfg.epochs:
            if os.path.exists(best_path):
                self._log(f"Training already complete for this config ({start}/{cfg.epochs} "
                          f"epochs). Using {best_path}")
                self.best_epoch = best_epoch
                return best_path
            start = cfg.epochs - 1

        n_val = len(val_loader) if val_loader is not None else 0
        self._log(f"Training '{cfg.task_name}' ({self.task_type}) on {self.device}: epochs "
                  f"{start + 1}→{cfg.epochs}, {len(train_loader)} train / {n_val} val batches, "
                  f"{type(optimizer).__name__} lr={base_lr:g}, early stop on "
                  f"{cfg.early_stop_metric} (patience {cfg.early_stop_patience})")
        if not n_val:
            self._log("Validation split is empty — 'best' falls back to the training loss.")

        tracker = Tracker()
        t0 = time.time()
        with tracker.run(f"audio_ops/{cfg.task_name}", run_name=f"{cfg.model_architecture}"):
            tracker.log_params({k: cfg.to_dict().get(k) for k in (
                "task_type", "feature_type", "epochs", "batch_size", "learning_rate",
                "optimizer", "weight_decay", "criterion", "seed", "input_shape",
                "class_labels", "segment_duration")} | {"device": str(self.device)})

            for epoch in range(start, cfg.epochs):
                e0 = time.time()
                lr = scheduler.get_last_lr()[0]
                train_loss, train_acc = self._train_epoch(model, train_loader, optimizer,
                                                          scaler, epoch, lr)
                val_loss, val_acc = (self._validate(model, val_loader) if n_val
                                     else (train_loss, train_acc))
                scheduler.step()

                dur = time.time() - e0
                eta = (time.time() - t0) / (epoch - start + 1) * (cfg.epochs - epoch - 1)
                row = {"train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc,
                       "val_acc": val_acc, "lr": lr, "epoch_duration_s": dur}
                tracker.log_metrics(row, step=epoch)
                self._write_metrics(epoch + 1, row)
                self._observe(epoch + 1, row)
                acc = f" | val_acc={val_acc:.3f}" if _finite(val_acc) else ""
                self._log(f"Epoch {epoch + 1}/{cfg.epochs} | train_loss={train_loss:.4f} | "
                          f"val_loss={val_loss:.4f}{acc} | lr={lr:.2e} | {dur:.1f}s | "
                          f"ETA {_fmt_eta(eta)}")

                monitor = val_loss if cfg.early_stop_metric == "val_loss" else -val_acc
                if _finite(monitor) and monitor < best_metric:
                    best_metric, best_epoch, patience = monitor, epoch + 1, 0
                    self._save_best(model, epoch, val_loss, val_acc)
                else:
                    patience += 1
                self._save_resume(model, optimizer, scheduler, epoch, best_metric,
                                  best_epoch, patience)
                if patience >= cfg.early_stop_patience:
                    self._log(f"Early stopping at epoch {epoch + 1} (best = {best_epoch}).")
                    break

            if not os.path.exists(best_path):   # metric never finite → keep final weights
                logger.warning("Validation metric never improved; saving the final weights.")
                self._save_best(model, cfg.epochs - 1, float("nan"), float("nan"))
            tracker.log_artifact(best_path, artifact_path="checkpoints")

        self.best_epoch = best_epoch
        return best_path

    def _train_epoch(self, model, loader, optimizer, scaler, epoch, lr) -> Tuple[float, float]:
        import torch

        cfg = self.cfg
        model.train()
        accum = int(cfg.grad_accumulation_steps)
        total, correct, count = 0.0, 0, 0
        n = len(loader)
        e0 = time.time()
        optimizer.zero_grad()
        for i, (x, y) in enumerate(loader):
            x = x.to(self.device, non_blocking=True)
            if isinstance(y, torch.Tensor):
                y = y.to(self.device, non_blocking=True)
            if scaler is not None:
                with torch.autocast(device_type="cuda"):
                    loss, out = self._step(model, x, y)
                scaler.scale(loss / accum).backward()
            else:
                loss, out = self._step(model, x, y)
                (loss / accum).backward()

            if (i + 1) % accum == 0 or i + 1 == n:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            total += float(loss.item())
            c, k = self._correct(out, y)
            correct, count = correct + c, count + k
            if (i + 1) % cfg.log_every_n_batches == 0:
                eta = (time.time() - e0) / (i + 1) * (n - i - 1)
                self._log(f"  epoch {epoch + 1}/{cfg.epochs} batch {i + 1}/{n} "
                          f"loss={total / (i + 1):.4f} (epoch ETA {_fmt_eta(eta)})")
                self._write_metrics(epoch + 1, {"batch_loss": total / (i + 1), "lr": lr,
                                                "step": i + 1}, granularity="batch")
        return total / max(n, 1), (correct / count if count else float("nan"))

    def _validate(self, model, loader) -> Tuple[float, float]:
        import torch

        model.eval()
        total, correct, count = 0.0, 0, 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                if isinstance(y, torch.Tensor):
                    y = y.to(self.device, non_blocking=True)
                loss, out = self._step(model, x, y)
                if out is None and self.task_type == "classification":
                    out = model(x)
                total += float(loss.item())
                c, k = self._correct(out, y)
                correct, count = correct + c, count + k
        return total / max(len(loader), 1), (correct / count if count else float("nan"))

    # ── Side channels ────────────────────────────────────────────────────────

    def _write_metrics(self, epoch: int, fields: dict, granularity: str = "epoch") -> None:
        if self.metrics is None:
            return
        try:
            self.metrics.write_training_metric(
                task_name=self.cfg.task_name, run_id=self.run_id, epoch=epoch,
                fields={k: float(v) for k, v in fields.items() if _finite(v)},
                tags={"model": self.cfg.model_architecture, "granularity": granularity})
        except Exception as exc:  # noqa: BLE001
            logger.debug("metrics write failed: %s", exc)

    def _observe(self, epoch: int, metrics: dict) -> None:
        """AssetObservation per epoch → a live metric curve in the Dagster UI."""
        ctx = self._ctx
        if ctx is None or getattr(ctx, "asset_key", None) is None:
            return
        try:
            from dagster import AssetObservation, MetadataValue

            md = {k: MetadataValue.float(float(v)) for k, v in metrics.items() if _finite(v)}
            md["epoch"] = MetadataValue.int(epoch)
            ctx.log_event(AssetObservation(asset_key=ctx.asset_key, metadata=md))
        except Exception as exc:  # noqa: BLE001
            logger.debug("AssetObservation skipped: %s", exc)

    # ── Checkpoints ──────────────────────────────────────────────────────────

    def _fingerprint(self) -> str:
        c = self.cfg
        return "|".join(str(x) for x in (c.task_name, c.model_architecture, c.feature_type,
                                         c.input_shape, c.epochs, c.class_labels))

    def _resume(self, model, optimizer, scheduler):
        """(start_epoch, best_metric, best_epoch, patience) — fresh unless a compatible
        ``last_checkpoint.pt`` exists."""
        import torch

        fresh = (0, float("inf"), 0, 0)
        path = os.path.join(self.checkpoint_dir, "last_checkpoint.pt")
        if not self.cfg.resume or not os.path.exists(path):
            return fresh
        try:
            # Written only by this trainer into the task's own output_dir; optimizer
            # state can hold non-tensor objects, hence weights_only=False.
            state = torch.load(path, map_location=self.device, weights_only=False)
            if state.get("cfg_fingerprint") != self._fingerprint():
                self._log("Ignoring last_checkpoint.pt: the training setup changed.")
                return fresh
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            if state.get("scheduler_state_dict"):
                scheduler.load_state_dict(state["scheduler_state_dict"])
        except Exception as exc:  # noqa: BLE001
            self._log(f"Could not resume from {path} ({str(exc)[:120]}) — starting fresh.")
            return fresh
        start = int(state.get("epoch", -1)) + 1
        self._log(f"Resuming at epoch {start + 1}/{self.cfg.epochs} "
                  f"(best epoch so far: {state.get('best_epoch', 0)}).")
        return (start, float(state.get("best_metric", float("inf"))),
                int(state.get("best_epoch", 0)), int(state.get("patience_counter", 0)))

    def _save_resume(self, model, optimizer, scheduler, epoch, best_metric, best_epoch,
                     patience) -> None:
        import torch

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, "last_checkpoint.pt")
        try:
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch, "best_metric": best_metric, "best_epoch": best_epoch,
                        "patience_counter": patience, "cfg_fingerprint": self._fingerprint(),
                        "saved_at": time.time()}, path + ".tmp")
            os.replace(path + ".tmp", path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not write resume checkpoint: %s", exc)

    def _save_best(self, model, epoch: int, val_loss: float, val_acc: float) -> None:
        """Weights-only, self-describing checkpoint (loadable with weights_only=True)."""
        import torch

        c = self.cfg
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, "best_model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "format_version": 2,
            "task_name": c.task_name,
            "task_type": self.task_type,
            "model_architecture": c.model_architecture,
            "class_names": list(c.class_labels or []),
            "num_classes": int(c.num_classes or 0),
            "input_shape": list(c.input_shape or []),
            "epoch": int(epoch) + 1,
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "saved_at": time.time(),
            "feature_config": {"feature_type": c.feature_type, "sample_rate": int(c.sample_rate),
                               "segment_duration": float(c.segment_duration),
                               "segment_overlap": float(c.segment_overlap)},
        }, path + ".tmp")
        os.replace(path + ".tmp", path)
        logger.info("Saved %s (epoch %d, val_loss=%.4f)", path, epoch + 1, val_loss)


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


__all__ = ["ModelTrainer"]
