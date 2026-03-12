import torch
import logging
import numpy as np
import mlflow
import mlflow.pytorch
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, task, config, checkpoint_dir="checkpoints"):
        self.task = task
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self, model, train_loader, val_loader=None):
        # 1. Convert object to dict for MLflow logging
        if hasattr(self.config, "__dict__"):
            # For standard objects/dataclasses
            params_to_log = self.config.__dict__
        elif hasattr(self.config, "dict"):
            # For Pydantic models (common in Dagster)
            params_to_log = self.config.dict()
        else:
            params_to_log = self.config # Fallback if it is already a dict
            
        mlflow.log_params(params_to_log)
        
        model = model.to(self.device)
        criterion = self.task.get_loss_function()
        optimizer = self.task.get_optimizer(model)
        
        best_loss = float('inf')

        epochs = getattr(self.config, 'epochs', 10)
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            
            # 3. Log Training Metric
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            logger.info(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f}")

            if val_loader:
                val_loss, metrics = self._validate(model, val_loader, criterion)
                
                # 4. Log Validation Metrics
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                if metrics:
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"val_{metric_name}", value, step=epoch)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_path = self.checkpoint_dir / "best_model.pt"
                    torch.save(model.state_dict(), best_model_path)
                    
                    # 5. Log Model Artifact (Optional: log the actual file)
                    mlflow.log_artifact(str(best_model_path))
                    logger.info(f"Saved Best Model: {metrics}")

        # 6. Log the model using MLflow's native PyTorch flavor
        mlflow.pytorch.log_model(
            pytorch_model=model, 
            artifact_path="model",
            registered_model_name=f"{self.task.__class__.__name__}_Model"
        )
        
        return str(self.checkpoint_dir / "best_model.pt")

    def _validate(self, model, loader, criterion):
        model.eval()
        total_loss, all_preds, all_labels = 0, [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = model(x)
                total_loss += criterion(out, y).item()
                all_preds.append(out.cpu().numpy())
                all_labels.append(y.cpu().numpy())
        
        metrics = self.task.compute_metrics(np.concatenate(all_labels), np.concatenate(all_preds))
        return total_loss / len(loader), metrics
    
    def test(self, model, test_loader):
        """
        Public method to evaluate a model on a test dataset.
        Reuses the internal validation logic for consistency.
        """
        logger.info("Starting Final Model Evaluation (Test Set)...")
        model = model.to(self.device)
        criterion = self.task.get_loss_function()
        
        test_loss, metrics = self._validate(model, test_loader, criterion)
        
        # Log to MLflow (Assuming this is called within an mlflow.start_run() block)
        mlflow.log_metric("test_loss", test_loss)
        if metrics:
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)
                
        logger.info(f"Test Evaluation Complete. Loss: {test_loss:.4f} | Metrics: {metrics}")
        return test_loss, metrics