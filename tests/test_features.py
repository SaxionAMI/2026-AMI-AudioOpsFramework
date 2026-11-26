import numpy as np
import pytest

from audio_ops_dagster.core import features as F


def test_builtins_registered():
    for name in ("spectrogram", "mel", "logmel", "mfcc", "raw", "ast", "ast_librosa"):
        assert F.has_feature(name), name


def test_unknown_feature_raises():
    with pytest.raises(F.FeatureError):
        F.get_feature_extractor("does_not_exist")


def test_register_custom():
    @F.register_feature("unit_test_feat")
    def _f(y, sr):
        return y

    assert F.has_feature("unit_test_feat")
    assert F.get_feature_extractor("unit_test_feat") is _f


@pytest.mark.parametrize("name,rows", [("spectrogram", 128), ("logmel", 128), ("mfcc", 40)])
def test_builtin_shapes(name, rows):
    pytest.importorskip("librosa")
    y = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    out = F.get_feature_extractor(name)(y, 16000)
    assert out.shape[0] == rows and out.dtype == np.float32
    assert np.isfinite(out).all()


def test_ast_librosa_shape():
    pytest.importorskip("librosa")
    out = F.get_feature_extractor("ast_librosa")(np.zeros(16000, dtype=np.float32), 16000)
    assert out.shape == (1024, 128)
