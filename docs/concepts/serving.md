# Serving

The inference service (`audio_ops_inference`) is a standalone FastAPI daemon.
On startup its `ModelStore`:

1. discovers tasks (same discovery as Dagster) and keeps those whose
   capabilities include `serve`;
2. resolves a **serve wrapper** type for each: `serve_wrapper` in the config,
   else `pretrained_event` for `task_type: nonverbal_pretrained`, else
   `transcription` for transcription tasks, else `classifier`;
3. calls the wrapper factory. The `classifier` factory builds the network with
   the task's `build_model()` and loads `best_model.pt` (local first; from
   object storage when `STORAGE_BACKEND=minio`). A task without a checkpoint is
   listed under `not_loaded` in `GET /models` and served after training +
   `POST /models/reload`.

## Built-in wrappers

| Type | For | Behaviour |
|---|---|---|
| `classifier` | any `build_model` task (classification, regression, detection) | same extractor, sample rate and `segment_duration` window as training; output through the task's `postprocess` |
| `transcription` | tasks with `predict(audio, t_offset=...)` | buffers `stream_buffer_seconds`, silence gate, hallucination filter, word timestamps on the media clock |
| `pretrained_event` | taggers with `predict(audio, sample_rate=, source=)` | buffers `stream_buffer_seconds` (default 2 s), places the event at its best frame |

Add your own with [`@register_wrapper`](../guide/serve-wrappers.md).

## Streams and isolation

Every stream (RTSP pull or WebSocket push) gets **its own wrapper instances**
from `store.for_stream()`: model weights are shared, buffers are private, so
audio from one stream can never leak into another's transcription or event
window. STT runs in a separate lane so a slow transcription pass can't stall
the fast models; both lanes drop the oldest queued chunk under load to stay live.

## The media clock

Each chunk is stamped with `media_t0` (seconds since stream start, advanced at
arrival) and every result carries `media_ts_start`/`media_ts_end` on that
clock. A client playing the same audio can line events up with its playhead
regardless of inference latency.

## Graceful degradation

InfluxDB, the LLM client and fusion are optional. If they are disabled or
unavailable the service logs a note and serves everything else.
