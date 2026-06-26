"""
LLMClient — Abstraction for local LLM inference in the Audio Ops pipeline.

Supports:
  • Ollama  (default, easiest to set up locally)
  • vLLM    (high-throughput production serving)
  • OpenAI-compatible APIs  (LocalAI, LiteLLM, text-generation-inference)

Generic OpenAI-compatible chat/completion client. The framework does not
require it; tasks may use it from their optional postprocess_with_llm() hook
(e.g. to refine, explain, or summarize a prediction). It is enabled only when
ENABLE_LLM=true.

Usage
-----
    from audio_ops_dagster.core.llm_client import LLMClient

    llm = LLMClient.from_env()

    # Simple completion
    fixed = llm.complete(
        "Fix the transcription errors and add punctuation: "
        "ther wer too many peeple at the meting today"
    )

    # Structured prompt with system context
    summary = llm.chat([
        {"role": "system", "content": "You are a medical transcription assistant."},
        {"role": "user",   "content": f"Patient audio analysis:\\n"
                                       f"Speech: {transcription}\\n"
                                       f"Non-verbal: {nonverbal_events}\\n"
                                       f"Sentiment: {sentiment}\\n"
                                       f"Provide a brief clinical summary."},
    ])
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for the LLM backend."""

    provider: str = "ollama"  # "ollama", "vllm", "openai_compatible"
    base_url: str = "http://ollama:11434"
    model: str = "llama3.2"
    api_key: Optional[str] = None  # only needed for authenticated APIs
    temperature: float = 0.3
    max_tokens: int = 512
    timeout_sec: int = 30
    enabled: bool = False  # opt-in — don't break pipelines that don't need LLM

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            provider=os.getenv("LLM_PROVIDER", "ollama"),
            base_url=os.getenv("LLM_BASE_URL", "http://ollama:11434"),
            model=os.getenv("LLM_MODEL", "llama3.2"),
            api_key=os.getenv("LLM_API_KEY"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512")),
            timeout_sec=int(os.getenv("LLM_TIMEOUT_SEC", "30")),
            enabled=os.getenv("ENABLE_LLM", "false").lower() == "true",
        )


class LLMClient:
    """
    Unified interface to local LLM APIs.

    All providers are accessed through an OpenAI-compatible chat/completions
    endpoint — Ollama, vLLM, and LocalAI all expose this format.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self._session = None

    @classmethod
    def from_env(cls) -> "LLMClient":
        return cls(LLMConfig.from_env())

    def _api_url(self) -> str:
        """Build the chat completions URL based on provider."""
        base = self.config.base_url.rstrip("/")
        if self.config.provider == "ollama":
            return f"{base}/api/chat"
        # vLLM, LocalAI, LiteLLM all use the OpenAI format
        return f"{base}/v1/chat/completions"

    def _build_payload(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Build the JSON payload for the API call."""
        if self.config.provider == "ollama":
            return {
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            }
        return {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }

    def _parse_response(self, data: Dict[str, Any]) -> str:
        """Extract the assistant message from the API response."""
        if self.config.provider == "ollama":
            return data.get("message", {}).get("content", "")
        # OpenAI format
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Send a chat completion request and return the assistant's reply.

        Returns empty string on failure (never raises — inference must not
        block the pipeline).
        """
        if not self.config.enabled:
            return ""

        import urllib.request
        import urllib.error

        url = self._api_url()
        payload = self._build_payload(messages)

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    **({"Authorization": f"Bearer {self.config.api_key}"}
                       if self.config.api_key else {}),
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return self._parse_response(data)
        except urllib.error.URLError as exc:
            logger.warning("LLM request failed (connection): %s", exc)
            return ""
        except Exception as exc:
            logger.warning("LLM request failed: %s", exc)
            return ""

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        """Convenience wrapper: single user prompt → assistant reply."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages)

    # ── Audio-specific prompt templates ──────────────────────────────────────

    def fix_transcription(self, raw_text: str, context: Optional[str] = None) -> str:
        """Fix STT errors, add punctuation, clean up transcription."""
        system = (
            "You are a precise transcription editor. Fix spelling errors, "
            "add proper punctuation and capitalization. Do NOT add content "
            "that wasn't in the original. Return ONLY the corrected text."
        )
        prompt = f"Fix this transcription:\n\n{raw_text}"
        if context:
            prompt += f"\n\nAdditional context:\n{context}"
        return self.complete(prompt, system=system) or raw_text

    def interpret_audio_analysis(
        self,
        transcription: Optional[str] = None,
        sentiment: Optional[str] = None,
        nonverbal_events: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> str:
        """
        Synthesize multi-task outputs into a natural-language interpretation.

        Combines STT, sentiment, and non-verbal detections into a coherent
        summary — the "so what" of the audio analysis.
        """
        system = (
            "You are an audio analysis interpreter. Given multiple analysis "
            "outputs from an audio processing pipeline, provide a brief, "
            "coherent summary of what happened in the audio. Be concise."
        )
        parts = []
        if transcription:
            parts.append(f"Speech content: {transcription}")
        if sentiment:
            parts.append(f"Emotional tone: {sentiment}")
        if nonverbal_events:
            parts.append(f"Non-verbal sounds: {', '.join(nonverbal_events)}")
        if context:
            parts.append(f"Context: {context}")

        if not parts:
            return ""

        prompt = "Interpret this audio analysis:\n\n" + "\n".join(parts)
        return self.complete(prompt, system=system)
