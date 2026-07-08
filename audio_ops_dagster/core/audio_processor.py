"""
audio_processor.py — segment audio files and extract features to .npy.

Feature extraction is pluggable (see core/features.py). The processor handles
loading, segmenting, and saving; *what* features get computed is decided by the
registered extractor named in the task's `feature_type`.
"""

import logging
from pathlib import Path

import numpy as np

from audio_ops_dagster.core.features import get_feature_extractor, list_features, FeatureError

logger = logging.getLogger(__name__)


class AudioProcessor:
    def __init__(self, sample_rate=16000, segment_duration=1.0, overlap=0.25,
                 feature_type="spectrogram"):
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.overlap = overlap
        self.feature_type = feature_type
        self.segment_samples = int(segment_duration * sample_rate)
        self.hop_samples = int(self.segment_samples * (1 - overlap))
        # Fail fast with a helpful message if the feature type isn't registered.
        try:
            self._extractor = get_feature_extractor(feature_type)
        except FeatureError as e:
            raise FeatureError(
                f"{e}  (available: {list_features()})"
            ) from None

    def process_directory(self, input_dir: str, output_dir: str,
                          force_recompute: bool = False,
                          progress=None, log_every: int = 100) -> dict:
        """Walk a directory, segment audio, and save .npy features per segment.

        Convention: <input_dir>/<label>/<file>.wav  → the parent folder is the
        label, mirrored into <output_dir>/<label>/<stem>_seg_NNN.npy

        Idempotency
        -----------
        A file whose segments already exist on disk is SKIPPED unless
        ``force_recompute`` is set. This makes re-runs cheap: a second run over
        the same 30k files with unchanged config does almost no work instead of
        re-extracting everything.

        Parameters
        ----------
        force_recompute : recompute even if segments already exist.
        progress        : optional callable(msg: str) for live progress (e.g.
                          Dagster's ``context.log.info``). Falls back to the
                          module logger.
        log_every       : emit an "i/total" heartbeat every N files.
        """
        import librosa

        emit = progress or logger.info

        input_path, output_path = Path(input_dir), Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            logger.error("Input path does not exist: %s", input_path)
            return {"error": "path_not_found", "total_files": 0, "total_segments": 0}

        stats = {"total_files": 0, "total_segments": 0, "skipped_files": 0,
                 "feature_type": self.feature_type}

        audio_files = [f for f in input_path.rglob("*")
                       if f.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a")]
        total = len(audio_files)
        # Heartbeat so a hung step is distinguishable from a slow one.
        emit(f"AST feature extraction: 0/{total} (feature_type={self.feature_type}, "
             f"segment={self.segment_duration}s overlap={self.overlap}, "
             f"force_recompute={force_recompute})")

        errors = 0
        first_error = None
        for idx, audio_file in enumerate(audio_files, start=1):
            try:
                label = audio_file.parent.name if audio_file.parent != input_path else "unknown"
                label_dir = output_path / label
                label_dir.mkdir(exist_ok=True)

                # Skip-if-exists: if this file already produced segments, don't
                # redo it. Count the existing segments so downstream sees a
                # populated feature dir and the "0 segments" guard doesn't misfire.
                if not force_recompute:
                    existing = list(label_dir.glob(f"{audio_file.stem}_seg_*.npy"))
                    if existing:
                        stats["total_segments"] += len(existing)
                        stats["skipped_files"] += 1
                        stats["total_files"] += 1
                        if idx % log_every == 0 or idx == total:
                            emit(f"  features: {idx}/{total} "
                                 f"({stats['skipped_files']} skipped, {errors} errored)")
                        continue

                y, _ = librosa.load(str(audio_file), sr=self.sample_rate)

                for i, start in enumerate(range(0, max(1, len(y)), self.hop_samples)):
                    segment = y[start:start + self.segment_samples]
                    if len(segment) < self.segment_samples:
                        segment = np.pad(segment, (0, self.segment_samples - len(segment)))
                    feature = self._extractor(segment, self.sample_rate)
                    np.save(label_dir / f"{audio_file.stem}_seg_{i:03d}.npy", feature)
                    stats["total_segments"] += 1

                stats["total_files"] += 1
            except Exception as e:
                errors += 1
                if first_error is None:
                    import traceback
                    first_error = traceback.format_exc()
                logger.error("Error processing %s: %s", audio_file, e)

            if idx % log_every == 0 or idx == total:
                emit(f"  features: {idx}/{total} "
                     f"({stats['skipped_files']} skipped, {errors} errored)")

        emit(f"Processed {stats['total_files']} files into {stats['total_segments']} "
             f"segments ({stats['skipped_files']} skipped, {errors} errored).")

        # Fail loudly instead of reporting a hollow success: if we found audio but
        # produced (and had cached) zero segments, the extractor is broken —
        # surface the real traceback rather than letting downstream steps hit "no
        # .npy files". Note: a run where everything was validly SKIPPED still has
        # total_segments > 0 (existing segments are counted), so this won't misfire.
        if audio_files and stats["total_segments"] == 0:
            raise RuntimeError(
                f"Feature extraction produced 0 segments from {len(audio_files)} "
                f"audio files — every file failed. First error was:\n{first_error}"
            )
        return stats