# Streaming inference

The inference service scores live audio from three kinds of sources and pushes
results as they happen.

```bash
audio-ops serve                  # or the inference compose file
open http://localhost:8000/docs  # interactive OpenAPI docs
```

## Sources

### File upload

```bash
curl -F file=@clip.wav "http://localhost:8000/infer/file?tasks=example_classifier"
```

Model tasks return per-segment results (segmented exactly like training) plus
an aggregate (`aggregation: mean_scores` when segments carry `scores`).
Transcription and event tasks process the whole clip.

### RTSP / HTTP pull

```bash
curl -X POST localhost:8000/streams -H 'content-type: application/json' \
     -d '{"url": "rtsp://mediamtx:8554/demo", "stream_id": "lobby", "chunk_duration_ms": 1000}'
```

The server runs `ffmpeg` to decode the stream to 16 kHz mono PCM, reconnects
on failure, and feeds `chunk_duration_ms` chunks to every loaded model.
Register several at once with `POST /streams/batch`; stop one with
`DELETE /streams/{id}`.

Set `AUTO_RTSP_STREAMS=rtsp://host/path|stream_id,...` to start streams at boot,
or `STREAM_AUTODISCOVER=true` to follow whatever is publishing on MediaMTX
(`STREAM_AUTODISCOVER_FILTER` takes a regex).

### WebSocket push (browser microphone or a file the browser plays)

Connect to `ws://host:8000/ws/ingest/{stream_id}` (add `?api_key=` if the
server requires a key) and send **binary** frames of little-endian Int16 PCM,
mono, 16 kHz, any frame size. Text frames `{"cmd":"reset"}` clear the timeline,
`{"cmd":"close"}` ends the session. Results come back on the same socket.

```js
const ws = new WebSocket("ws://localhost:8000/ws/ingest/mic");
ws.binaryType = "arraybuffer";
ws.onmessage = (e) => console.log(JSON.parse(e.data));
// from an AudioWorklet: Float32 → Int16 at 16 kHz
ws.send(int16Samples.buffer);
```

Because the browser captures exactly what it plays, the server's media clock
equals the browser's playhead. A WebSocket session on a stream id takes over from an RTSP pull with the same id.

## Results

Subscribe with Server-Sent Events, per stream or for all streams:

```bash
curl -N localhost:8000/streams/lobby/events
curl -N localhost:8000/events
```

```json
{"stream_id": "lobby", "task": "example_classifier",
 "media_ts_start": 3.0, "media_ts_end": 4.0, "media_now": 4.0,
 "label": "class_b", "confidence": 0.98, "scores": {"class_a": 0.01, "...": 0.0},
 "rms": 0.21, "buffering": false, "timestamp": 1764000000.0}
```

Transcription events add `transcription`, `words`, `language`; event taggers
add `primary_event`, `event_media_ts`, `top_classes`. A subscription may be
opened before the stream exists.

## How chunks are scored

- **Model tasks** keep a rolling window of the last `segment_duration` seconds
  and score it on every chunk, so a model trained on 3 s windows still sees
  3 s of context with 1 s chunks.
- **Transcription** buffers `stream_buffer_seconds` of audio per call, skips
  near-silence (`silence_rms`) and drops known hallucinations ("thank you for
  watching", subtitle credits, degenerate repetition; extend with
  `STT_HALLUCINATION_PHRASES`). It runs in its own lane so it can't stall the
  fast models.
- **Event taggers** buffer `stream_buffer_seconds` (default 2 s).

Under load each lane drops the **oldest** queued chunk (`PROC_QUEUE_MAX`,
`STT_QUEUE_MAX`). The stream stays live and the timestamps stay correct.

## The demo media server

`docker-compose.inference.yml` ships MediaMTX with two paths:

| Path | What |
|---|---|
| `rtsp://localhost:8554/demo` | loops `mediamtx/test_audio/demo.wav` on demand |
| `rtsp://localhost:8554/microphone` | publish target (RTSP, or WHIP at `http://localhost:8889/microphone/whip`) |

Generate the demo file with `audio-ops demo-data --stream-demo mediamtx/test_audio/demo.wav`
and add your own paths in `mediamtx/mediamtx.yml`.

## Fusion timeline (optional)

With `ENABLE_FUSION=true`, each stream keeps a `TranscriptFuser` that merges
transcript words, non-verbal events and sentiment labels onto one timeline,
emits a rolling `{"task": "fusion", ...}` event, and exposes:

- `GET /fusion/{stream_id}/timeline?window_s=120`: structured segments
- `POST /fusion/{stream_id}/llm`: an LLM summary (`ENABLE_LLM=true`), with
  optional `system`/`instruction` overrides

Tags and default prompts come in English (default) and Dutch
(`FUSION_LANGUAGE=nl`); pass your own prompts for other domains. Fusion lives in
`extras/`. It is an example of project-level logic built on the core, not part
of the core.
