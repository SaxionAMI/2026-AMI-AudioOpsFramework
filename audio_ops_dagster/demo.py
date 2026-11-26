"""
demo.py — synthetic audio so the quickstart works without any real data.

``audio-ops demo-data`` writes a small labelled dataset of tones in noise for
the bundled ``example_classifier`` (``class_a`` ≈ 300 Hz, ``class_b`` ≈ 1.2 kHz,
``class_c`` ≈ 3 kHz) plus a longer ``demo.wav`` for the streaming demo
(MediaMTX ``demo`` path / ``GET /demo``). Generated, so there are no licensing
questions about the audio itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

DEFAULT_CLASSES: Dict[str, float] = {"class_a": 300.0, "class_b": 1200.0, "class_c": 3000.0}


def _tone(freq: float, seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    f = freq * (1 + rng.uniform(-0.04, 0.04))
    vibrato = 1 + 0.01 * np.sin(2 * np.pi * rng.uniform(3, 6) * t)
    y = 0.3 * np.sin(2 * np.pi * f * vibrato * t)
    y += 0.08 * np.sin(2 * np.pi * 2 * f * t)                        # a harmonic
    y += rng.uniform(0.005, 0.03) * rng.standard_normal(t.size)      # noise floor
    fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.05) if t.size else t
    return (y * fade).astype(np.float32)


def write_dataset(out_dir: str | Path, classes: Optional[Dict[str, float]] = None,
                  files_per_class: int = 12, seconds: float = 3.0, sample_rate: int = 16000,
                  seed: int = 0) -> int:
    """Write ``<out_dir>/<label>/<label>_NN.wav``. Returns the number of files."""
    import soundfile as sf

    rng = np.random.default_rng(seed)
    classes = classes or DEFAULT_CLASSES
    out = Path(out_dir)
    n = 0
    for label, freq in classes.items():
        d = out / label
        d.mkdir(parents=True, exist_ok=True)
        for i in range(files_per_class):
            dur = seconds * rng.uniform(0.7, 1.3)
            sf.write(d / f"{label}_{i:02d}.wav", _tone(freq, dur, sample_rate, rng), sample_rate)
            n += 1
    return n


def write_stream_demo(path: str | Path, classes: Optional[Dict[str, float]] = None,
                      seconds_per_class: float = 4.0, sample_rate: int = 16000,
                      seed: int = 1) -> Path:
    """One WAV that cycles through the classes — handy for streaming demos."""
    import soundfile as sf

    rng = np.random.default_rng(seed)
    classes = classes or DEFAULT_CLASSES
    y = np.concatenate([_tone(f, seconds_per_class, sample_rate, rng) for f in classes.values()])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, sample_rate)
    return path
