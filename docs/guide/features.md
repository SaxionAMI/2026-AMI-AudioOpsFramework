# Feature extractors

A feature extractor is any callable `(waveform: np.ndarray, sample_rate: int) -> np.ndarray`
registered under a name. A task selects one with `feature_type:`; training,
batch inference and the API all call the same function on the same window.

## Built-ins

| `feature_type` | Output (1 s @ 16 kHz) | Notes |
|---|---|---|
| `spectrogram` *(default)* | `(128, 32)` | log-mel normalised to [0, 1] |
| `logmel`, `mel` | `(128, 32)` | log-mel in dB |
| `mfcc` | `(40, 32)` | 40 MFCCs |
| `raw` | `(16000,)` | the waveform, for models with their own front end |
| `ast` | `(1024, 128)` | exact AST input via transformers (`ast` extra) |
| `ast_librosa` | `(1024, 128)` | AST-compatible approximation, librosa only |

The time axis grows with `segment_duration` (librosa's default hop is 512
samples). The pipeline logs the real shape; set `input_shape: [1, F, T]` to
match if your model needs a fixed size.

Two-dimensional features get a channel dimension, so a CNN sees
`(batch, 1, F, T)`; one-dimensional features stay `(batch, T)`.

!!! warning "Use the same feature for training and serving"
    `ast` and `ast_librosa` are close but not bit-identical. Changing
    `feature_type` invalidates the feature cache automatically, but you must
    retrain. The API warns when a checkpoint's recorded feature type differs
    from the config.

## Your own

Register it in your task module (it's imported during discovery, so the
registration happens before features are computed):

```python
import numpy as np
from audio_ops import register_feature


@register_feature("chroma")
def chroma(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    return librosa.feature.chroma_stft(y=y, sr=sr).astype(np.float32)
```

```yaml
feature_type: chroma
```

`audio-ops validate` warns about unregistered feature types (it can't know
about ones registered at import time). The warning goes away once the task
module registers it.
