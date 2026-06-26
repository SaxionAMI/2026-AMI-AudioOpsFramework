from audio_ops_dagster import capabilities as cap


def test_archetypes_expand():
    assert cap.resolve_capabilities({"archetype": "trainable_model"}) == \
        frozenset({cap.TRAIN, cap.EVALUATE, cap.BATCH_INFER, cap.SERVE})
    assert cap.resolve_capabilities({"archetype": "pretrained_servable"}) == frozenset({cap.SERVE})


def test_explicit_list_wins():
    c = cap.resolve_capabilities({"capabilities": ["train", "evaluate"]})
    assert c == frozenset({cap.TRAIN, cap.EVALUATE})
    assert not cap.produces_dagster_assets(frozenset({cap.SERVE}))
    assert cap.produces_dagster_assets(c)


def test_legacy_flag_fallback():
    assert cap.resolve_capabilities({"dagster": {"training_enabled": False}}) == frozenset({cap.SERVE})
    assert cap.resolve_capabilities({}) == cap.ARCHETYPES["trainable_model"]


def test_unknown_raises():
    import pytest
    with pytest.raises(cap.CapabilityError):
        cap.resolve_capabilities({"capabilities": ["trian"]})
