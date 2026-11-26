import json

import numpy as np
import pytest

pytest.importorskip("librosa")
sf = pytest.importorskip("soundfile")

from audio_ops_dagster.core.audio_processor import MANIFEST_NAME, AudioProcessor  # noqa: E402


@pytest.fixture
def audio_dir(tmp_path):
    d = tmp_path / "in" / "lbl"
    d.mkdir(parents=True)
    sf.write(d / "a.wav", np.zeros(int(2.5 * 16000), dtype=np.float32), 16000)
    (tmp_path / "in" / "notes.txt").write_text("ignored")
    return tmp_path / "in"


def test_process_and_cache(audio_dir, tmp_path):
    out = tmp_path / "out"
    p = AudioProcessor(segment_duration=1.0, overlap=0.25)
    s1 = p.process_directory(str(audio_dir), str(out))
    assert s1["total_segments"] == 3 and s1["feature_shape"] == [128, 32]
    assert json.loads((out / MANIFEST_NAME).read_text())["feature_type"] == "spectrogram"
    s2 = p.process_directory(str(audio_dir), str(out))
    assert s2["skipped_files"] == 1


def test_changed_settings_invalidate_cache_and_remove_stale(audio_dir, tmp_path):
    out = tmp_path / "out"
    AudioProcessor(segment_duration=0.5, overlap=0.0).process_directory(str(audio_dir), str(out))
    assert len(list((out / "lbl").glob("*.npy"))) == 5
    s = AudioProcessor(segment_duration=1.0, overlap=0.0).process_directory(str(audio_dir),
                                                                           str(out))
    assert s["skipped_files"] == 0
    assert len(list((out / "lbl").glob("*.npy"))) == 3     # old 0.5 s segments are gone


def test_invalid_overlap():
    with pytest.raises(ValueError):
        AudioProcessor(overlap=1.0)
