# HTTP API

The service publishes an OpenAPI schema at `/openapi.json` and interactive docs
at `/docs`. With `API_KEY` set, every endpoint except `/health` requires the
header `X-API-Key: <key>` (or the query parameter `?api_key=<key>` for
EventSource/WebSocket clients); otherwise `401`.

## Service

| Method & path | Description |
|---|---|
| `GET /health` | `status`, `version`, loaded `models`, stream count, `influxdb`, `fusion`, `llm`, `auth` flags |
| `GET /models` | `models`: each wrapper's description; `not_loaded`: task → reason |
| `POST /models/reload` | re-discover tasks and reload checkpoints (new streams use the new models) |
| `GET /demo` | the demo WAV (`DEMO_AUDIO_PATH`) |

## One-shot inference

`POST /infer/file`: multipart `file` (any format libsndfile reads), optional
query `tasks=a,b`. `413` above `MAX_UPLOAD_MB`, `422` if the audio can't be decoded.

```json
{
  "tasks": {
    "example_classifier": {
      "label": "class_a", "confidence": 0.93,
      "scores": {"class_a": 0.93, "class_b": 0.05, "class_c": 0.02},
      "aggregation": "mean_scores",
      "segments": [
        {"label": "class_a", "confidence": 0.95, "scores": {"...": 0}, "start_sec": 0.0, "end_sec": 1.0}
      ],
      "latency_ms": 41.2
    }
  },
  "duration_s": 3.0, "rms": 0.21, "source": "file"
}
```

## Streams

| Method & path | Body / params | Description |
|---|---|---|
| `POST /streams` | `{"url", "stream_id"?, "chunk_duration_ms"?=1000}` | start pulling a stream (`409` if active) |
| `POST /streams/batch` | `{"streams": [{"stream_id", "url", "chunk_duration_ms"?}]}` | start several |
| `GET /streams` | | id, url, active, mode (`rtsp`/`ws`/`pending`), frames, errors, dropped, rms, media_t, models |
| `DELETE /streams/{id}` | | stop and remove |
| `GET /discovery` | | MediaMTX auto-discovery state |
| `WS /ws/ingest/{id}` | binary Int16 PCM 16 kHz mono; text `{"cmd": "reset"\|"close"}` | push audio, receive events on the same socket; close code `4401` = bad key |

## Events (Server-Sent Events)

| Path | |
|---|---|
| `GET /streams/{id}/events` | one stream (may be opened before the stream exists) |
| `GET /events` | all streams |

Each `data:` line is one JSON event. Common fields: `timestamp` (wall clock),
`stream_id`, `task`, `rms`, `media_ts_start`, `media_ts_end`, `media_now`,
`buffering`. Then, depending on the wrapper: `label`, `confidence`, `scores`,
`value`, `labels` (model tasks); `transcription`, `words`, `word_count`,
`language` (STT); `primary_event`, `primary_event_conf`, `event_media_ts`,
`top_classes` (event taggers); `error` if a model failed. A `: keep-alive`
comment is sent every 30 s.

## Fusion (`ENABLE_FUSION=true`)

| Method & path | Description |
|---|---|
| `GET /fusion/{id}/timeline?window_s=` | fused transcript, segments, sentiment and non-verbal tracks |
| `POST /fusion/{id}/llm` | `{"window_s"?, "system"?, "instruction"?}` → `{"summary"}` (`503` if the LLM is disabled/unavailable) |

Streams also emit `{"task": "fusion", "fused_transcript", "segments", "sentiment_track", "nonverbal_track"}` events.
