import torch.nn as nn
from audio_ops_dagster.core.base_task import AudioTask

class SpeechToTextTask(AudioTask):
    def build_model(self, input_shape):
        # Your model here (Transformer, RNN, etc.)
        return nn.Sequential(...)
    
    def compute_metrics(self, y_true, y_pred):
        # Your metrics (WER, CER, etc.)
        return {"wer": 0.1, "cer": 0.05}
    
    def postprocess(self, raw_output):
        # Your postprocessing
        return {"transcription": "text"}