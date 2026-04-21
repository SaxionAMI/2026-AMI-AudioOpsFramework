"""
Speech-to-Text task using Wav2Vec2 for automatic speech recognition.
"""

from audio_ops_dagster.tasks.speech_to_text.task_speech_to_text import SpeechToTextTask

__all__ = ["SpeechToTextTask"]
