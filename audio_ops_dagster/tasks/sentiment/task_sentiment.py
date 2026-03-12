from typing import Dict, Tuple, Any
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from audio_ops_dagster.core.base_task import AudioTask
from audio_ops_dagster.registry import register_task


def get_device():
    if torch.cuda.is_available():
        # You could add logic here to check memory etc.
        return torch.device("cuda")
    elif torch.backends.mps.is_available(): # For Mac M1/M2 users
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# When building your model in the Dagster asset:
# model = MyModel().to(device)


@register_task("sentiment")
class SentimentTask(AudioTask):
    """Sentiment classification from audio spectrograms."""
    
    def build_model(self, input_shape: Tuple[int, ...]) -> nn.Module:
        model = models.resnet18(pretrained=True)
        original_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, self.config.num_classes)
        )
        return model
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        if y_pred.ndim == 2:
            y_pred_labels = y_pred.argmax(axis=1)
        else:
            y_pred_labels = y_pred
            
        accuracy = accuracy_score(y_true, y_pred_labels)
        f1 = f1_score(y_true, y_pred_labels, average='weighted', zero_division=0)
        
        return {
            'accuracy': float(accuracy),
            'f1_weighted': float(f1)
        }
    
    def postprocess(self, raw_output: torch.Tensor) -> Dict[str, Any]:
        if raw_output.ndim == 1:
            raw_output = raw_output.unsqueeze(0)
            
        probabilities = torch.softmax(raw_output, dim=1)
        predicted_indices = probabilities.argmax(dim=1)
        
        results = []
        for idx in predicted_indices:
            label = self.config.class_labels[idx.item()] if idx < len(self.config.class_labels) else "unknown"
            results.append({'label': label})
            
        return results[0] if len(results) == 1 else results
    
    def get_loss_function(self) -> nn.Module:
        return nn.CrossEntropyLoss()