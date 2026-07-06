"""
features.py — pluggable audio feature extractors.

The framework no longer hardcodes "everything is a mel spectrogram". A feature
extractor is any callable ``(waveform: np.ndarray, sample_rate: int) -> np.ndarray``
registered under a name. A task selects one with ``feature_type:`` in its config,
and you can add your own from a task module:

    from audio_ops_dagster.core.features import register_feature

    @register_feature("my_features")
    def my_features(y, sr):
        ...
        return feats  # np.ndarray, shape (C, T) or (T,)

Built-ins: spectrogram (normalized log-mel), logmel/mel (log-mel dB), mfcc, raw.
librosa is imported lazily inside each extractor so importing this registry is
cheap and dependency-light.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

FeatureExtractor = Callable[[np.ndarray, int], np.ndarray]

_FEATURE_EXTRACTORS: Dict[str, FeatureExtractor] = {}


class FeatureError(ValueError):
    """Unknown or invalid feature type."""


def register_feature(name: str):
    """Register a feature extractor under ``name`` (decorator or direct call)."""
    def deco(fn: FeatureExtractor) -> FeatureExtractor:
        _FEATURE_EXTRACTORS[name] = fn
        return fn
    return deco


def get_feature_extractor(name: str) -> FeatureExtractor:
    if name not in _FEATURE_EXTRACTORS:
        raise FeatureError(
            f"Unknown feature_type '{name}'. Registered: {sorted(_FEATURE_EXTRACTORS)}. "
            f"Register your own with @register_feature('{name}')."
        )
    return _FEATURE_EXTRACTORS[name]


def list_features() -> List[str]:
    return sorted(_FEATURE_EXTRACTORS)


def has_feature(name: str) -> bool:
    return name in _FEATURE_EXTRACTORS


# ── Built-in extractors ──────────────────────────────────────────────────────

@register_feature("spectrogram")
def _spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    """Normalized log-mel spectrogram in [0, 1], shape (128, T). (Default.)"""
    import librosa
    s = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    return np.clip((librosa.power_to_db(s, ref=np.max) + 40) / 40, 0, 1).astype(np.float32)


@register_feature("logmel")
@register_feature("mel")
def _logmel(y: np.ndarray, sr: int) -> np.ndarray:
    """Log-mel spectrogram in dB, shape (128, T)."""
    import librosa
    s = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    return librosa.power_to_db(s, ref=np.max).astype(np.float32)


@register_feature("mfcc")
def _mfcc(y: np.ndarray, sr: int) -> np.ndarray:
    """MFCCs, shape (40, T)."""
    import librosa
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).astype(np.float32)


@register_feature("raw")
def _raw(y: np.ndarray, sr: int) -> np.ndarray:
    """Raw waveform, shape (T,). For models that do their own feature extraction."""
    return y.astype(np.float32)


# ── AST (Audio Spectrogram Transformer) features ─────────────────────────────
# Produces the exact input MIT/ast-finetuned-audioset-10-10-0.4593 expects:
# a (1024, 128) normalized mel patch (AudioSet mean/std). Registered here so any
# task/project (sentiment, triage, ...) can select feature_type: ast and get
# identical features in training AND serving — no per-project reimplementation.
# The HF feature extractor is loaded lazily and cached, so importing this module
# stays cheap (task discovery doesn't pay the transformers import cost).
_AST_FE = None
_AST_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"


def _get_ast_fe():
    global _AST_FE
    if _AST_FE is None:
        from transformers import ASTFeatureExtractor
        _AST_FE = ASTFeatureExtractor.from_pretrained(_AST_CHECKPOINT)
    return _AST_FE


@register_feature("ast")
def _ast(y: np.ndarray, sr: int) -> np.ndarray:
    """AST mel features, shape (1024, 128) float32 (AudioSet-normalized)."""
    fe = _get_ast_fe()
    out = fe(y, sampling_rate=sr, return_tensors="np")
    return out["input_values"][0].astype(np.float32)