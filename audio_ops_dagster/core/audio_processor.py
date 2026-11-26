"""
audio_processor.py — segment audio files and extract features to ``.npy``.

Feature extraction is pluggable (see ``core/features.py``). The processor
handles loading, segmenting and saving; *what* gets computed is decided by the
registered extractor named in the task's ``feature_type``.

Layout: ``<input_dir>/<label>/<file>.wav`` → ``<output_dir>/<label>/<stem>_seg_NNN.npy``.
Files directly in ``input_dir`` (no label folder) go under ``unknown/``.

Caching: a ``_features_manifest.json`` in the output directory records the
feature settings. Re-runs skip files whose segments already exist, *unless*
the settings changed — then everything is recomputed automatically, so stale
features from an old ``feature_type``/``segment_duration`` are never reused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from audio_ops_dagster._log import get_logger
from audio_ops_dagster.core.features import FeatureError, get_feature_extractor, list_features
from audio_ops_dagster.core.runtime import list_audio_files, segment_name

logger = get_logger(__name__)

MANIFEST_NAME = "_features_manifest.json"


def segment_waveform(y: np.ndarray, segment_samples: int, hop_samples: int) -> List[np.ndarray]:
    """Split ``y`` into fixed windows (last one zero-padded).

    Windows start every ``hop_samples``. A trailing window is only emitted if it
    carries at least half a hop of real audio (or if it is the only window),
    so short tails don't produce near-silent segments.
    """
    n = len(y)
    if n == 0:
        return [np.zeros(segment_samples, dtype=np.float32)]
    hop = max(1, hop_samples)
    out: List[np.ndarray] = []
    start = 0
    while start < n:
        remaining = n - start
        if out and remaining < segment_samples and remaining < hop // 2:
            break
        seg = y[start:start + segment_samples]
        if len(seg) < segment_samples:
            seg = np.pad(seg, (0, segment_samples - len(seg)))
        out.append(seg.astype(np.float32, copy=False))
        if start + segment_samples >= n:
            break
        start += hop
    return out


class AudioProcessor:
    def __init__(self, sample_rate=16000, segment_duration=1.0, overlap=0.25,
                 feature_type="spectrogram"):
        if not 0 <= overlap < 1:
            raise ValueError(f"segment_overlap must be in [0, 1), got {overlap}")
        self.sample_rate = int(sample_rate)
        self.segment_duration = float(segment_duration)
        self.overlap = float(overlap)
        self.feature_type = feature_type
        self.segment_samples = int(self.segment_duration * self.sample_rate)
        self.hop_samples = max(1, int(self.segment_samples * (1 - self.overlap)))
        try:
            self._extractor = get_feature_extractor(feature_type)
        except FeatureError as e:
            raise FeatureError(f"{e}  (available: {list_features()})") from None

    @property
    def hop_seconds(self) -> float:
        return self.hop_samples / self.sample_rate

    def settings(self) -> Dict[str, object]:
        return {
            "feature_type": self.feature_type,
            "sample_rate": self.sample_rate,
            "segment_duration": self.segment_duration,
            "segment_overlap": self.overlap,
        }

    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """Features for one already-segmented window."""
        return self._extractor(np.asarray(waveform, dtype=np.float32), self.sample_rate)

    def segments(self, waveform: np.ndarray) -> List[np.ndarray]:
        return segment_waveform(np.asarray(waveform, dtype=np.float32),
                                self.segment_samples, self.hop_samples)

    def _manifest_matches(self, output_path: Path) -> bool:
        f = output_path / MANIFEST_NAME
        if not f.exists():
            # Legacy dirs (no manifest) are trusted only if empty.
            return not any(output_path.rglob("*.npy"))
        try:
            return json.loads(f.read_text()) == self.settings()
        except Exception:  # noqa: BLE001
            return False

    def process_directory(self, input_dir: str, output_dir: str,
                          force_recompute: bool = False,
                          progress: Optional[Callable[[str], None]] = None,
                          log_every: int = 100) -> dict:
        """Walk a directory, segment audio, and save ``.npy`` features per segment.

        Parameters
        ----------
        force_recompute : recompute even if segments already exist.
        progress        : optional ``callable(msg)`` for live progress (e.g.
                          Dagster's ``context.log.info``).
        log_every       : emit an "i/total" heartbeat every N files.
        """
        import librosa

        emit = progress or logger.info

        input_path, output_path = Path(input_dir), Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stats = {"total_files": 0, "total_segments": 0, "skipped_files": 0,
                 "errored_files": 0, "feature_type": self.feature_type,
                 "feature_shape": None}

        if not input_path.exists():
            logger.error("Input path does not exist: %s", input_path)
            return {**stats, "error": "path_not_found"}

        if not force_recompute and not self._manifest_matches(output_path):
            emit(f"Feature settings changed since the last run (or no manifest) — "
                 f"recomputing all features in {output_path}.")
            force_recompute = True

        audio_files = list_audio_files(input_path)
        total = len(audio_files)
        emit(f"Feature extraction: 0/{total} files (feature_type={self.feature_type}, "
             f"segment={self.segment_duration}s, overlap={self.overlap}, "
             f"force_recompute={force_recompute})")

        first_error = None
        for idx, audio_file in enumerate(audio_files, start=1):
            try:
                label = audio_file.parent.name if audio_file.parent != input_path else "unknown"
                label_dir = output_path / label
                label_dir.mkdir(exist_ok=True)
                existing = sorted(label_dir.glob(f"{audio_file.stem}_seg_*.npy"))

                if existing and not force_recompute:
                    stats["total_segments"] += len(existing)
                    stats["skipped_files"] += 1
                    stats["total_files"] += 1
                    if stats["feature_shape"] is None:
                        stats["feature_shape"] = list(np.load(existing[0], mmap_mode="r").shape)
                else:
                    for old in existing:          # never leave stale extra segments
                        old.unlink()
                    y, _ = librosa.load(str(audio_file), sr=self.sample_rate, mono=True)
                    for i, segment in enumerate(self.segments(y)):
                        feature = self.extract(segment)
                        if stats["feature_shape"] is None:
                            stats["feature_shape"] = list(np.shape(feature))
                        np.save(label_dir / f"{segment_name(audio_file.stem, i)}.npy", feature)
                        stats["total_segments"] += 1
                    stats["total_files"] += 1
            except Exception as e:  # noqa: BLE001
                stats["errored_files"] += 1
                if first_error is None:
                    import traceback
                    first_error = traceback.format_exc()
                logger.error("Error processing %s: %s", audio_file, e)

            if idx % log_every == 0 or idx == total:
                emit(f"  features: {idx}/{total} files "
                     f"({stats['skipped_files']} cached, {stats['errored_files']} errored)")

        # Written last: an interrupted run leaves the old manifest, so the next
        # run still knows the directory may hold features from other settings.
        (output_path / MANIFEST_NAME).write_text(json.dumps(self.settings(), indent=2))

        emit(f"Processed {stats['total_files']} files into {stats['total_segments']} "
             f"segments ({stats['skipped_files']} cached, {stats['errored_files']} errored). "
             f"Feature shape: {stats['feature_shape']}")

        # Fail loudly instead of reporting a hollow success.
        if audio_files and stats["total_segments"] == 0:
            raise RuntimeError(
                f"Feature extraction produced 0 segments from {len(audio_files)} "
                f"audio files — every file failed. First error was:\n{first_error}"
            )
        return stats
