from audio_ops_dagster.core import features as F


def test_builtins_registered():
    for name in ("spectrogram", "mel", "logmel", "mfcc", "raw"):
        assert F.has_feature(name), name


def test_unknown_feature_raises():
    import pytest
    with pytest.raises(F.FeatureError):
        F.get_feature_extractor("does_not_exist")


def test_register_custom():
    @F.register_feature("unit_test_feat")
    def _f(y, sr):
        return y
    assert F.has_feature("unit_test_feat")
    assert F.get_feature_extractor("unit_test_feat") is _f
