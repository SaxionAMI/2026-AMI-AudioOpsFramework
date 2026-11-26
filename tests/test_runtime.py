import numpy as np
import pytest

from audio_ops_dagster.core.audio_processor import segment_waveform
from audio_ops_dagster.core.runtime import (
    add_channel_dim,
    parse_segment_name,
    segment_name,
)


@pytest.mark.parametrize("stem,expected", [
    ("rec01_seg_003", ("rec01", 3)),
    ("my_segment_a_seg_000", ("my_segment_a", 0)),   # not truncated at the first "_seg"
    ("clip_chunk_12", ("clip", 12)),
    ("plain_name", ("plain_name", 0)),
])
def test_parse_segment_name(stem, expected):
    assert parse_segment_name(stem) == expected


def test_segment_name_roundtrip():
    assert parse_segment_name(segment_name("abc_def", 42)) == ("abc_def", 42)


def test_segment_waveform_counts():
    sr = 16000
    seg, hop = sr, int(sr * 0.75)
    assert len(segment_waveform(np.zeros(sr), seg, hop)) == 1           # exactly one window
    assert len(segment_waveform(np.zeros(int(2.5 * sr)), seg, hop)) == 3
    assert len(segment_waveform(np.zeros(100), seg, hop)) == 1          # short clip → padded
    out = segment_waveform(np.ones(int(1.2 * sr)), seg, hop)
    assert all(len(s) == seg for s in out)


def test_add_channel_dim_matches_training_rule():
    assert add_channel_dim(np.zeros((128, 32))).shape == (1, 128, 32)
    assert add_channel_dim(np.zeros(16000)).shape == (16000,)
    assert add_channel_dim(np.zeros((1, 128, 32))).shape == (1, 128, 32)


def test_to_model_input_ranks():
    torch = pytest.importorskip("torch")
    from audio_ops_dagster.core.runtime import to_model_input

    assert tuple(to_model_input(np.zeros((128, 32))).shape) == (1, 1, 128, 32)
    assert tuple(to_model_input(np.zeros(16000)).shape) == (1, 16000)
    x = to_model_input(np.array([np.nan, np.inf, 1.0]))
    assert torch.isfinite(x).all()
