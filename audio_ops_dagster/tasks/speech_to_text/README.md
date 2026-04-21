# Speech-to-Text Task

Automatic Speech Recognition (ASR) using **Wav2Vec2** from HuggingFace Transformers.

## Overview

This task performs speech-to-text conversion using state-of-the-art pretrained models. It supports:

- ✅ **Pretrained inference** - Use models like `facebook/wav2vec2-base-960h` out of the box
- ✅ **Fine-tuning** - Adapt pretrained models to your specific domain
- ✅ **Standard metrics** - WER (Word Error Rate), CER (Character Error Rate)
- ✅ **Multiple languages** - Supports various Wav2Vec2 models for different languages

## Quick Start

### 1. Prepare Your Audio Data

Place your `.wav` files in the input directory:

```
audio_input/speech_to_text/
├── speaker1/
│   ├── audio1.wav
│   └── audio2.wav
└── speaker2/
    ├── audio3.wav
    └── audio4.wav
```

For **fine-tuning**, you'll also need transcriptions (handled in the dataset preparation step).

### 2. Configure the Task

Edit `config.yaml` to customize settings:

```yaml
# Model selection - choose from HuggingFace
model_name: facebook/wav2vec2-base-960h  # English (LibriSpeech)
# model_name: facebook/wav2vec2-large-960h-lv60-self  # Better English
# model_name: facebook/wav2vec2-large-xlsr-53  # Multilingual

# For fine-tuning
batch_size: 4
learning_rate: 0.0001
epochs: 10
```

### 3. Run the Pipeline

From the Dagster UI (http://localhost:3000):

1. Navigate to **Assets**
2. Select the **speech_to_text** group
3. Click **Materialize All** to run the full pipeline

Or use the CLI:

```bash
dagster asset materialize -m audio_ops_dagster.definitions \
  --select "speech_to_text*"
```

## Available Models

### English Models

| Model | Size | WER (LibriSpeech) | Best For |
|-------|------|-------------------|----------|
| `facebook/wav2vec2-base-960h` | 95M | ~3.4% | General English, fast inference |
| `facebook/wav2vec2-large-960h` | 317M | ~1.8% | High accuracy English |
| `facebook/wav2vec2-large-960h-lv60-self` | 317M | ~1.6% | Best accuracy English |

### Multilingual Models

| Model | Languages | Best For |
|-------|-----------|----------|
| `facebook/wav2vec2-large-xlsr-53` | 53 languages | Cross-lingual transfer |
| `facebook/wav2vec2-xls-r-300m` | 128 languages | Multilingual ASR |

### Domain-Specific

- Medical: `jonatasgrosman/wav2vec2-large-xlsr-53-english`
- Conversational: `patrickvonplaten/wav2vec2-large-960h-lv60-self-conversational`

## Pipeline Stages

### 1. Raw Audio Files
- Scans `audio_input/speech_to_text/` for `.wav` files
- Supports nested directory structures

### 2. Segmented Features
- Processes audio into features (raw waveforms or spectrograms)
- For Wav2Vec2, raw waveforms are used directly
- Resamples to 16kHz if needed

### 3. Dataset Splits
- Creates train/val/test splits
- Supports speaker-independent splitting
- Generates CSV manifests with audio paths and transcriptions

### 4. Trained Model
- For inference: loads pretrained weights
- For fine-tuning: adapts model to your data
- Checkpoints saved to `storage/speech_to_text/checkpoints/`
- Tracked in MLflow

### 5. Model Evaluation
- Computes WER, CER, MER, WIL metrics
- Logs results to MLflow
- Saves evaluation report to `storage/speech_to_text/outputs/`

## Metrics Explained

### WER (Word Error Rate)
- **Lower is better** (0% = perfect)
- Formula: `(Substitutions + Deletions + Insertions) / Total Words`
- Industry standard for ASR evaluation

### CER (Character Error Rate)
- **Lower is better** (0% = perfect)
- More fine-grained than WER
- Useful for languages without clear word boundaries

### MER (Match Error Rate)
- Focuses on matches vs. total tokens
- Alternative to WER

### WIL (Word Information Lost)
- Measures information preservation
- Accounts for word importance

## Inference Usage

### Python API

```python
from audio_ops_dagster.registry import get_task, TaskConfig
import librosa

# Load the trained model
config = TaskConfig(
    task_name="speech_to_text",
    input_dir="audio_input/speech_to_text",
    output_dir="storage/speech_to_text",
    feature_type="raw",
    model_name="facebook/wav2vec2-base-960h"
)

task = get_task("speech_to_text", config)
model = task.build_model(input_shape=(1, 16000))
model.eval()

# Load and preprocess audio
audio, sr = librosa.load("test_audio.wav", sr=16000)
audio_tensor = task.preprocess_audio(audio, sr)

# Run inference
with torch.no_grad():
    logits = model(audio_tensor).logits
    result = task.postprocess(logits)

print(f"Transcription: {result['transcription']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### REST API (via Inference Service)

```bash
curl -X POST http://localhost:8000/predict/speech_to_text \
  -F "audio=@my_audio.wav"
```

Response:
```json
{
  "transcription": "hello world this is a test",
  "confidence": 0.94,
  "metadata": {
    "model": "facebook/wav2vec2-base-960h",
    "sample_rate": 16000,
    "duration_seconds": 3.2
  }
}
```

## Fine-Tuning on Custom Data

### Data Format

Create a dataset with audio files and transcriptions:

```
audio_input/speech_to_text/
├── metadata.csv  # or transcriptions.txt
└── audio/
    ├── audio1.wav  → "hello world"
    ├── audio2.wav  → "how are you"
    └── audio3.wav  → "goodbye"
```

`metadata.csv` format:
```csv
file_path,transcription
audio/audio1.wav,"hello world"
audio/audio2.wav,"how are you"
audio/audio3.wav,"goodbye"
```

### Training Configuration

```yaml
# config.yaml
task_name: speech_to_text
input_dir: audio_input/speech_to_text
output_dir: storage/speech_to_text

model_name: facebook/wav2vec2-base-960h
batch_size: 4  # Reduce if OOM
learning_rate: 0.0001  # Fine-tuning rate
epochs: 10

optimizer: adam
criterion: ctc  # CTC loss for ASR

use_remote_storage: true  # Save to MinIO
```

### Run Training

```bash
# Materialize the full pipeline
dagster job execute -m audio_ops_dagster.definitions \
  -j speech_to_text_full_training_job
```

## Advanced Configuration

### Using Different Processors

```python
# In config or code
model_name: "facebook/wav2vec2-large-xlsr-53"  # Multilingual
# The processor is automatically loaded for the model
```

### Custom Loss Functions

```python
# Override in task class
def get_loss_function(self):
    return nn.CTCLoss(
        blank=self.processor.tokenizer.pad_token_id,
        reduction='mean',
        zero_infinity=True
    )
```

### Batch Inference

```python
# Process multiple files
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = []

for audio_file in audio_files:
    audio, sr = librosa.load(audio_file, sr=16000)
    audio_tensor = task.preprocess_audio(audio, sr)
    
    with torch.no_grad():
        logits = model(audio_tensor).logits
        result = task.postprocess(logits)
        results.append(result)
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` in `config.yaml`
- Use a smaller model variant
- Process shorter audio segments

### Poor Accuracy
- Check audio quality (16kHz, mono, clear speech)
- Try a larger model
- Fine-tune on domain-specific data
- Verify transcriptions are correct

### Model Not Found
- Check internet connection (downloads from HuggingFace)
- Verify model name is correct
- Try: `huggingface-cli login` for private models

### Slow Inference
- Use GPU if available
- Try smaller model variants
- Batch multiple files together
- Consider quantization for deployment

## Integration with Other Services

### MinIO Storage
When `use_remote_storage: true`:
- Features → `speech_to_text/features/`
- Datasets → `speech_to_text/dataset/`
- Models → `speech_to_text/models/`
- Results → `speech_to_text/outputs/`

### MLflow Tracking
- Experiment: `Audio_Ops_Speech_to_text`
- Metrics: WER, CER, MER, WIL
- Parameters: model_name, learning_rate, batch_size
- Artifacts: model checkpoints, evaluation reports

### Grafana Dashboards
- Monitor inference latency
- Track WER trends over time
- Audio processing throughput

## Examples

See the `examples/` directory for:
- `basic_inference.py` - Simple transcription
- `batch_processing.py` - Process multiple files
- `fine_tuning_example.py` - Custom domain adaptation
- `multilingual_demo.py` - Multiple language support

## References

- [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477)
- [HuggingFace Wav2Vec2 Docs](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
- [Fine-tuning Guide](https://huggingface.co/blog/fine-tune-wav2vec2-english)
