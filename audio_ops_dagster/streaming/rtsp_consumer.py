"""
RTSP Audio Consumer
===================
Pulls a continuous audio stream from a MediaMTX RTSP endpoint via an
FFmpeg subprocess and yields fixed-size PCM chunks (numpy arrays) for
downstream inference.

Usage
-----
    async for chunk, timestamp in RTSPConsumer("rtsp://localhost:8554/microphone"):
        results = await model.predict(chunk)

Architecture
------------
  MediaMTX RTSP ──► FFmpeg (decode → 16 kHz mono PCM) ──► asyncio queue
                                                              │
                                                        consumer loop
                                                              │
                                                     inference engine
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for an RTSP audio stream consumer."""
    url: str
    sample_rate: int = 16_000          # Hz  — match your model's expected SR
    chunk_duration_ms: int = 1_000     # ms  — size of each yielded chunk
    overlap_duration_ms: int = 0       # ms  — sliding-window overlap
    channels: int = 1                  # mono
    reconnect_delay: float = 3.0       # seconds between reconnect attempts
    max_reconnects: int = -1           # -1 = infinite
    stream_id: str = ""                # logical name (e.g. "room_1")
    ffmpeg_extra_args: list[str] = field(default_factory=list)

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_duration_ms / 1000)

    @property
    def overlap_samples(self) -> int:
        return int(self.sample_rate * self.overlap_duration_ms / 1000)


class RTSPConsumer:
    """
    Async generator that continuously reads PCM audio from an RTSP stream.

    Each iteration yields ``(audio_chunk: np.ndarray[float32], utc_timestamp: float)``.
    Audio is normalised to the range [-1.0, 1.0].

    The consumer automatically reconnects on stream failure.
    """

    def __init__(self, config: StreamConfig | str, **kwargs):
        if isinstance(config, str):
            config = StreamConfig(url=config, **kwargs)
        self.config = config
        self._stop_event = asyncio.Event()
        self._reconnect_count = 0
        self._is_running = False

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def stop(self) -> None:
        """Signal the consumer to stop after the current chunk."""
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        return self._is_running

    async def __aiter__(self) -> AsyncIterator[tuple[np.ndarray, float]]:
        self._is_running = True
        try:
            async for chunk, ts in self._stream_loop():
                yield chunk, ts
        finally:
            self._is_running = False

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    async def _stream_loop(self) -> AsyncIterator[tuple[np.ndarray, float]]:
        cfg = self.config
        overlap_buf: Optional[np.ndarray] = None

        while not self._stop_event.is_set():
            logger.info(
                "[%s] Connecting to %s (attempt %d)",
                cfg.stream_id or "stream",
                cfg.url,
                self._reconnect_count + 1,
            )
            proc = None
            try:
                proc = await self._spawn_ffmpeg()
                chunk_bytes = cfg.chunk_samples * 2  # int16 = 2 bytes / sample

                while not self._stop_event.is_set():
                    # Read exactly one chunk of raw PCM from FFmpeg stdout
                    raw = await asyncio.wait_for(
                        self._read_exact(proc.stdout, chunk_bytes),
                        timeout=10.0,
                    )
                    if not raw:
                        logger.warning("[%s] FFmpeg stdout closed", cfg.stream_id)
                        break

                    timestamp = time.time()
                    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

                    # Apply overlap if configured
                    if overlap_buf is not None and len(overlap_buf):
                        audio = np.concatenate([overlap_buf, audio])

                    if cfg.overlap_samples > 0:
                        overlap_buf = audio[-cfg.overlap_samples:]
                    else:
                        overlap_buf = None

                    yield audio[: cfg.chunk_samples], timestamp

                self._reconnect_count += 1

            except asyncio.TimeoutError:
                logger.warning("[%s] Timeout reading from stream", cfg.stream_id)
                self._reconnect_count += 1

            except Exception as exc:
                logger.exception("[%s] Stream error: %s", cfg.stream_id, exc)
                self._reconnect_count += 1

            finally:
                if proc:
                    try:
                        proc.kill()
                        await proc.wait()
                    except Exception:
                        pass

            if cfg.max_reconnects != -1 and self._reconnect_count >= cfg.max_reconnects:
                logger.error("[%s] Max reconnects reached — stopping", cfg.stream_id)
                return

            if not self._stop_event.is_set():
                logger.info(
                    "[%s] Reconnecting in %.1f s …",
                    cfg.stream_id,
                    cfg.reconnect_delay,
                )
                await asyncio.sleep(cfg.reconnect_delay)

    async def _spawn_ffmpeg(self) -> asyncio.subprocess.Process:
        """Launch FFmpeg to decode RTSP to raw PCM on stdout."""
        cfg = self.config
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-rtsp_transport", "tcp",       # prefer TCP for reliability
            "-i", cfg.url,
            "-vn",                           # drop video track if any
            "-acodec", "pcm_s16le",          # 16-bit signed PCM
            "-ar", str(cfg.sample_rate),     # resample to model SR
            "-ac", str(cfg.channels),        # mono
            *cfg.ffmpeg_extra_args,
            "-f", "s16le",                   # raw PCM format
            "pipe:1",                        # stdout
        ]
        logger.debug("[%s] FFmpeg cmd: %s", cfg.stream_id, " ".join(cmd))
        return await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

    @staticmethod
    async def _read_exact(stream: asyncio.StreamReader, n: int) -> bytes:
        """Read exactly n bytes from an asyncio stream, or return b'' on EOF."""
        buf = bytearray()
        while len(buf) < n:
            chunk = await stream.read(n - len(buf))
            if not chunk:
                return bytes(buf) if buf else b""
            buf.extend(chunk)
        return bytes(buf)


# ------------------------------------------------------------------ #
# Convenience: push a WAV / audio file to MediaMTX as RTSP           #
# ------------------------------------------------------------------ #

async def push_file_to_rtsp(
    file_path: str,
    rtsp_url: str,
    loop: bool = False,
    sample_rate: int = 16_000,
) -> None:
    """
    Push a local audio file to MediaMTX so it becomes an RTSP stream.
    Useful for testing the inference pipeline without a live microphone.

    Args:
        file_path: Path to any audio file FFmpeg can decode.
        rtsp_url:  Destination RTSP URL, e.g. ``rtsp://localhost:8554/file_stream``.
        loop:      If True, loop the file indefinitely.
        sample_rate: Output sample rate (Hz).
    """
    loop_arg = ["-stream_loop", "-1"] if loop else []
    cmd = [
        "ffmpeg",
        "-loglevel", "warning",
        "-re",                          # read at native pace (real-time)
        *loop_arg,
        "-i", file_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        rtsp_url,
    ]
    logger.info("Pushing %s → %s", file_path, rtsp_url)
    proc = await asyncio.create_subprocess_exec(*cmd)
    await proc.wait()
    if proc.returncode and proc.returncode != 255:
        logger.error("FFmpeg exited with code %d", proc.returncode)


# ------------------------------------------------------------------ #
# CLI helper: python -m audio_ops_dagster.streaming.rtsp_consumer     #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys

    async def _demo():
        url = sys.argv[1] if len(sys.argv) > 1 else "rtsp://localhost:8554/microphone"
        cfg = StreamConfig(url=url, chunk_duration_ms=1000, stream_id="demo")
        consumer = RTSPConsumer(cfg)
        print(f"Listening to {url} — Ctrl-C to stop")
        async for chunk, ts in consumer:
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            print(f"  t={ts:.3f}  samples={len(chunk)}  rms={rms:.4f}")

    asyncio.run(_demo())