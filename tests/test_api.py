"""Inference API tests against a freshly trained example checkpoint."""

import io
import json

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("dagster")
sf = pytest.importorskip("soundfile")

from _helpers import train_example  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture
def client(demo_project, monkeypatch):
    from fastapi.testclient import TestClient

    train_example(demo_project)
    import audio_ops_inference.api as api

    monkeypatch.setattr(api.settings, "api_key", None)
    with TestClient(api.app) as c:
        yield c, api


def _wav(seconds=3.0, freq=300.0):
    from audio_ops_dagster.demo import _tone

    y = _tone(freq, seconds, 16000, np.random.default_rng(3))
    buf = io.BytesIO()
    sf.write(buf, y, 16000, format="WAV")
    buf.seek(0)
    return buf, y


def test_health_and_models(client):
    c, _ = client
    h = c.get("/health").json()
    assert h["status"] == "ok" and "example_classifier" in h["models"]
    models = c.get("/models").json()["models"]
    assert models[0]["feature_type"] == "spectrogram"


def test_infer_file_segments_like_training(client):
    c, _ = client
    buf, _ = _wav(3.0, 300.0)            # class_a tone
    r = c.post("/infer/file", files={"file": ("a.wav", buf, "audio/wav")})
    assert r.status_code == 200
    res = r.json()["tasks"]["example_classifier"]
    assert res["label"] == "class_a"
    assert len(res["segments"]) == 4 and res["segments"][1]["start_sec"] == 0.75


def test_infer_file_rejects_garbage(client):
    c, _ = client
    r = c.post("/infer/file", files={"file": ("a.wav", io.BytesIO(b"nope"), "audio/wav")})
    assert r.status_code == 422


def test_api_key_enforced(client, monkeypatch):
    c, api = client
    monkeypatch.setattr(api.settings, "api_key", "s3cret")
    assert c.get("/health").status_code == 200                    # always open
    assert c.get("/models").status_code == 401
    assert c.get("/models", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert c.get("/models?api_key=s3cret").status_code == 200


def test_websocket_ingest_streams_results(client):
    c, _ = client
    _, y = _wav(3.0, 1200.0)             # class_b tone
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
    with c.websocket_connect("/ws/ingest/mic") as ws:
        hello = ws.receive_json()
        assert hello["event"] == "connected" and "example_classifier" in hello["models"]
        for i in range(0, len(pcm), 3200):
            ws.send_bytes(pcm[i:i + 3200])
        events = [ws.receive_json() for _ in range(2)]
        ws.send_text(json.dumps({"cmd": "close"}))
    assert all(e["task"] == "example_classifier" for e in events)
    assert events[0]["media_ts_start"] == 0.0 and events[1]["media_ts_end"] == 2.0
    assert events[-1]["label"] == "class_b"


def test_per_stream_wrappers_are_isolated(client):
    _, api = client
    a, b = api.store.for_stream(), api.store.for_stream()
    wa, wb = a["example_classifier"], b["example_classifier"]
    assert wa is not wb and wa.model is wb.model                  # shared weights
    wa.predict(np.ones(8000, dtype=np.float32) * 0.1, 0.0)
    assert len(wa._window) == 8000 and len(wb._window) == 0       # private buffers
