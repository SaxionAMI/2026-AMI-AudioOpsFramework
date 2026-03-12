import numpy as np
import librosa
from pathlib import Path
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, sample_rate=16000, segment_duration=1.0, overlap=0.25, feature_type="spectrogram"):
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.overlap = overlap
        self.feature_type = feature_type
        self.segment_samples = int(segment_duration * sample_rate)
        self.hop_samples = int(self.segment_samples * (1 - overlap))

    def process_directory(self, input_dir: str, output_dir: str) -> dict:
        """Walks through directory, segments audio, and saves .npy features."""
        input_path, output_path = Path(input_dir), Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
    
        print(f"DEBUG: Processing directory: {input_path.absolute()}")
        if not input_path.exists():
            print(f"DEBUG ERROR: The path {input_path} does not exist inside the container!")
            return {"error": "path_not_found"}
        
        all_files = list(input_path.rglob("*"))
        print(f"DEBUG: Found {len(all_files)} total items in the directory (including folders).")
   
        
        stats = {"total_files": 0, "total_segments": 0}
        
        for audio_file in [f for f in input_path.rglob("*") if f.suffix.lower() in [".wav", ".mp3"]]:
            try:
                # Use folder name as label (standard for image/audio classification)
                label = audio_file.parent.name if audio_file.parent != input_path else "unknown"
                label_dir = output_path / label
                label_dir.mkdir(exist_ok=True)

                y, _ = librosa.load(str(audio_file), sr=self.sample_rate)
                
                # Segmenting
                for i, start in enumerate(range(0, len(y), self.hop_samples)):
                    end = start + self.segment_samples
                    segment = y[start:end]
                    if len(segment) < self.segment_samples:
                        segment = np.pad(segment, (0, self.segment_samples - len(segment)))
                    
                    # Feature Extraction
                    feature = self._extract(segment)
                    np.save(label_dir / f"{audio_file.stem}_seg_{i:03d}.npy", feature)
                    stats["total_segments"] += 1
                
                stats["total_files"] += 1
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")
        print(f"DEBUG: Finished. Processed {stats['total_files']} audio files.")
        return stats

    def _extract(self, y: np.ndarray) -> np.ndarray:
        if self.feature_type == "spectrogram":
            s = librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_mels=128)
            return np.clip((librosa.power_to_db(s, ref=np.max) + 40) / 40, 0, 1).astype(np.float32)
        return y.astype(np.float32) # Default raw