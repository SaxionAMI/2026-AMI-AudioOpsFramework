from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn

class AudioTask(ABC):
    """
    Abstract base class for audio processing tasks.
    Tasks must implement: build_model, compute_metrics, and postprocess.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...]) -> nn.Module:
        """Return task-specific PyTorch nn.Module."""
        pass

    @abstractmethod
    def compute_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Return dict of metrics (e.g., accuracy, f1)."""
        pass

    @abstractmethod
    def postprocess(self, raw_output: torch.Tensor) -> Dict[str, Any]:
        """Format raw model output for end-users."""
        pass

    def get_loss_function(self):
        """Default: Cross-Entropy for classification. Override for Regression/STT."""
        if self.config.criterion == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif self.config.criterion == "mse":
            return nn.MSELoss()
        raise ValueError(f"Unknown criterion: {self.config.criterion}")

    def get_optimizer(self, model: nn.Module):
        """Standard Adam/SGD optimizer based on config."""
        if self.config.optimizer == "adam":
            return torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(model.parameters(), lr=self.config.learning_rate, momentum=0.9)
        raise ValueError(f"Unknown optimizer: {self.config.optimizer}")