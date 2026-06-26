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

    def process_directory(self, input_dir: str, output_dir: str) -> dict:
        """Walk a directory, segment audio, and save .npy features per segment.

        Convention: <input_dir>/<label>/<file>.wav  → the parent folder is the
        label, mirrored into <output_dir>/<label>/<stem>_seg_NNN.npy
        """
        import librosa

        input_path, output_path = Path(input_dir), Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            logger.error("Input path does not exist: %s", input_path)
            return {"error": "path_not_found", "total_files": 0, "total_segments": 0}

        stats = {"total_files": 0, "total_segments": 0, "feature_type": self.feature_type}

        audio_files = [f for f in input_path.rglob("*")
                       if f.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a")]
        logger.info("Found %d audio files under %s (feature_type=%s)",
                    len(audio_files), input_path, self.feature_type)

        for audio_file in audio_files:
            try:
                label = audio_file.parent.name if audio_file.parent != input_path else "unknown"
                label_dir = output_path / label
                label_dir.mkdir(exist_ok=True)

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
                logger.error("Error processing %s: %s", audio_file, e)

        logger.info("Processed %d files into %d segments.",
                    stats["total_files"], stats["total_segments"])
        return stats
