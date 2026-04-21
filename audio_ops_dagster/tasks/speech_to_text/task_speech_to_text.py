"""
Speech-to-Text Task — updated with InfluxDB word-timeline logging,
Grafana panel hints, and LLM-based transcription correction.
"""

from typing import Dict, Tuple, Any, List, Optional
import torch
import torch.nn as nn
import numpy as np

from audio_ops_dagster.core.base_task import AudioTask
from audio_ops_dagster.registry import register_task


@register_task("speech_to_text")
class SpeechToTextTask(AudioTask):
    """
    Automatic Speech Recognition using Wav2Vec2 from HuggingFace.

    Supports pretrained inference and CTC-based fine-tuning.
    """

    TASK_TYPE = "transcription"

    def __init__(self, config):
        super().__init__(config)
        self.model_name = getattr(config, "model_name", "facebook/wav2vec2-base-960h")
        self._processor = None
        self.target_sample_rate = 16000

    @property
    def processor(self):
        """Lazy-load the processor to avoid import failures in non-STT contexts."""
        if self._processor is None:
            from transformers import Wav2Vec2Processor
            self._processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        return self._processor

    def build_model(self, input_shape: Tuple[int, ...]) -> nn.Module:
        from transformers import Wav2Vec2ForCTC
        model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        return model.to(self.device)

    def compute_metrics(self, y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        import jiwer
        y_true = [str(ref) for ref in y_true]
        y_pred = [str(hyp) for hyp in y_pred]
        return {
            "wer": float(jiwer.wer(y_true, y_pred)),
            "cer": float(jiwer.cer(y_true, y_pred)),
            "mer": float(jiwer.mer(y_true, y_pred)),
            "wil": float(jiwer.wil(y_true, y_pred)),
        }

    def postprocess(self, raw_output: torch.Tensor) -> Any:
        logits = raw_output.logits if hasattr(raw_output, "logits") else raw_output
        predicted_ids = torch.argmax(logits, dim=-1)
        transcriptions = self.processor.batch_decode(predicted_ids)

        # Compute per-token confidence
        log_probs = torch.log_softmax(logits, dim=-1)
        confidences = []
        for i, pred_ids in enumerate(predicted_ids):
            token_log_probs = log_probs[i, range(len(pred_ids)), pred_ids]
            avg_conf = torch.exp(token_log_probs.mean()).item()
            confidences.append(avg_conf)

        if len(transcriptions) == 1:
            return {
                "transcription": transcriptions[0],
                "confidence": confidences[0],
                "word_count": len(transcriptions[0].split()),
            }
        return {
            "transcriptions": transcriptions,
            "confidences": confidences,
            "word_counts": [len(t.split()) for t in transcriptions],
        }

    def preprocess_for_inference(self, audio_array, sample_rate: int = 16000):
        """Resample and process audio for Wav2Vec2."""
        if sample_rate != self.target_sample_rate:
            import librosa
            audio_array = librosa.resample(
                audio_array, orig_sr=sample_rate, target_sr=self.target_sample_rate
            )
        inputs = self.processor(
            audio_array,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        return inputs.input_values.to(self.device)

    def get_loss_function(self) -> nn.Module:
        return nn.CTCLoss(
            blank=self.processor.tokenizer.pad_token_id,
            zero_infinity=True,
        )

    def compute_loss(self, model, batch, device):
        inputs, targets = batch
        labels = self.processor(text=targets, return_tensors="pt", padding=True).input_ids
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        logits = outputs.logits

        input_lengths = torch.full(
            (logits.shape[0],), logits.shape[1], dtype=torch.long
        )
        target_lengths = torch.tensor([len(t) for t in labels], dtype=torch.long)

        log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
        return self.get_loss_function()(log_probs, labels, input_lengths, target_lengths)

    # ── InfluxDB integration — word-level timeline ───────────────────────────

    def format_influx_record(
        self,
        segment_id: str,
        prediction: Any,
        audio_offset_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        STT → InfluxDB fields:
          transcription  (string)  — full transcription text
          confidence     (float)   — average model confidence
          word_count     (int)     — number of words in transcription

        The transcription text is stored as a field so it can be queried
        alongside the audio timeline in Grafana.
        """
        if isinstance(prediction, dict):
            text = prediction.get("transcription", prediction.get("transcriptions", [""])[0])
            conf = prediction.get("confidence", prediction.get("confidences", [0.0])[0])
            wc = prediction.get("word_count", prediction.get("word_counts", [0])[0])
        else:
            text, conf, wc = str(prediction), 0.0, 0

        fields: Dict[str, Any] = {
            "transcription": text,
            "confidence": float(conf),
            "word_count": int(wc),
        }
        if audio_offset_sec is not None:
            fields["audio_offset_sec"] = audio_offset_sec

        return {"fields": fields, "tags": {}}

    def grafana_panel_hints(self) -> List[Dict[str, Any]]:
        return [
            {
                "panel_type": "table",
                "title": "Transcription Timeline",
                "field": "transcription",
                "agg": "last",
                "description": "Words spoken over the audio timeline, aligned with timestamps",
            },
            {
                "panel_type": "timeseries",
                "title": "STT Confidence Over Time",
                "field": "confidence",
                "agg": "mean",
                "description": "Model confidence per audio segment",
            },
            {
                "panel_type": "timeseries",
                "title": "Words Per Segment",
                "field": "word_count",
                "agg": "sum",
                "description": "Number of transcribed words per time window",
            },
            {
                "panel_type": "stat",
                "title": "Total Words Transcribed",
                "field": "word_count",
                "agg": "sum",
                "description": "Total word count across all segments",
            },
            {
                "panel_type": "gauge",
                "title": "Average Confidence",
                "field": "confidence",
                "agg": "mean",
                "description": "Mean transcription confidence",
            },
        ]

    # ── LLM post-processing ──────────────────────────────────────────────────

    def postprocess_with_llm(self, raw_result, context=None, llm_client=None):
        """Use LLM to fix transcription errors and add punctuation."""
        if llm_client is None:
            return raw_result

        if isinstance(raw_result, dict) and "transcription" in raw_result:
            ctx_str = None
            if context and context.get("nonverbal_events"):
                ctx_str = f"Non-verbal sounds detected: {', '.join(context['nonverbal_events'])}"
            fixed = llm_client.fix_transcription(raw_result["transcription"], context=ctx_str)
            if fixed:
                raw_result["transcription_raw"] = raw_result["transcription"]
                raw_result["transcription"] = fixed
        return raw_result
