"""
TranscriptFuser — single-timeline fusion of STT + nonverbal + sentiment.
=========================================================================
This is the piece that makes the three task streams "speak the same language"
so they can be handed to an LLM as one coherent object.

Every task in the pipeline emits results on the SAME media clock
(seconds since the audio stream started — see api.py StreamHandle.media_samples).
The fuser collects them onto one timeline and produces:

  1. render_transcript()      -> human-readable text with nonverbal events
                                 interleaved inline at their real time position
                                 and sentiment shifts annotated, e.g.

        [00:12] (kalm) Dokter, ik heb sinds gisteren pijn in mijn borst [kreunt]
        [00:19] (angstig) en het wordt erger als ik adem [hoest] [piepende ademhaling]

  2. build_llm_payload()      -> structured dict (segments with text +
                                 concurrent nonverbal tags + sentiment context)
                                 ready to JSON-dump into an LLM prompt.

  3. build_llm_messages()     -> ready-to-send OpenAI/Ollama chat messages.

It is pure-Python (no torch / no tf) so it can live in core/ and be imported
by both the inference service and any Dagster asset.

Design notes
------------
* Thread-unsafe by itself; the API wraps one fuser per stream and only
  touches it from the single asyncio worker, so no lock is needed there.
* It keeps a rolling horizon (default 300 s) and prunes anything older so a
  long-running stream does not grow unbounded.
* Word-level timestamps are optional. If the STT backend does not emit word
  timestamps the fuser falls back to placing the whole utterance evenly
  across its window — alignment is then coarse but never wrong.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Timeline primitives                                                          #
# --------------------------------------------------------------------------- #

@dataclass
class WordEvent:
    t_start: float          # media seconds
    t_end: float
    word: str
    confidence: float = 0.0
    language: str = ""


@dataclass
class NonverbalEvent:
    t_start: float
    t_end: float
    label: str              # curated group: cough / cry / pain / ...
    confidence: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class SentimentEvent:
    t_start: float
    t_end: float
    label: str
    confidence: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)


def _fmt_clock(t: float) -> str:
    if t < 0 or math.isnan(t):
        t = 0.0
    m = int(t) // 60
    s = int(t) % 60
    return f"{m:02d}:{s:02d}"


# --------------------------------------------------------------------------- #
# The fuser                                                                    #
# --------------------------------------------------------------------------- #

class TranscriptFuser:
    """
    Rolling, media-clock-aligned fusion of the three audio tasks.

    Parameters
    ----------
    horizon_seconds:
        Drop events whose t_end is older than (latest_seen - horizon).
    segment_gap_seconds:
        A silence gap larger than this starts a new transcript segment/line.
    nonverbal_min_conf:
        Curated nonverbal events below this confidence are ignored in the
        rendered transcript (kept in the raw payload).
    nonverbal_merge_seconds:
        Repeated identical nonverbal labels within this window collapse into
        one inline tag (avoids "[hoest] [hoest] [hoest]").
    """

    # Dutch-friendly display names for inline tags. Unknown labels fall back
    # to the raw label, so adding a curated group needs no change here.
    NONVERBAL_DISPLAY = {
        "cough":     "hoest",
        "laugh":     "lacht",
        "cry":       "huilt",
        "pain":      "kreunt",
        "breathing": "ademhaling",
        "silence":   "stilte",
        "music":     "muziek",
        "alarm":     "alarm",
        "applause":  "applaus",
        # 'speech' is intentionally omitted — speech is already the transcript.
    }
    NONVERBAL_IGNORE_INLINE = {"speech", "silence", "unknown"}

    def __init__(
        self,
        horizon_seconds: float = 300.0,
        segment_gap_seconds: float = 1.6,
        nonverbal_min_conf: float = 0.12,
        nonverbal_merge_seconds: float = 1.2,
    ):
        self.horizon = horizon_seconds
        self.segment_gap = segment_gap_seconds
        self.nv_min_conf = nonverbal_min_conf
        self.nv_merge = nonverbal_merge_seconds

        self.words: List[WordEvent] = []
        self.nonverbals: List[NonverbalEvent] = []
        self.sentiments: List[SentimentEvent] = []
        self._latest_t: float = 0.0

    # ----- ingestion ------------------------------------------------------- #

    def add_words(self, words: List[Dict[str, Any]], language: str = "") -> None:
        """words: list of {word, t_start, t_end, confidence} on the media clock."""
        for w in words:
            txt = (w.get("word") or "").strip()
            if not txt:
                continue
            ev = WordEvent(
                t_start=float(w.get("t_start", 0.0)),
                t_end=float(w.get("t_end", w.get("t_start", 0.0))),
                word=txt,
                confidence=float(w.get("confidence") or 0.0),
                language=language or w.get("language", ""),
            )
            self.words.append(ev)
            self._latest_t = max(self._latest_t, ev.t_end)
        self._prune()

    def add_utterance(
        self,
        text: str,
        t_start: float,
        t_end: float,
        confidence: float = 0.0,
        language: str = "",
    ) -> None:
        """
        Fallback when no word timestamps exist: spread the utterance evenly
        across [t_start, t_end] so inline placement is still approximately right.
        """
        toks = [t for t in text.split() if t]
        if not toks:
            return
        span = max(t_end - t_start, 1e-3)
        step = span / len(toks)
        for i, tok in enumerate(toks):
            ws = t_start + i * step
            self.words.append(WordEvent(ws, ws + step, tok, confidence, language))
        self._latest_t = max(self._latest_t, t_end)
        self._prune()

    def add_nonverbal(
        self,
        label: str,
        t_start: float,
        t_end: float,
        confidence: float = 0.0,
        scores: Optional[Dict[str, float]] = None,
    ) -> None:
        ev = NonverbalEvent(
            t_start=float(t_start),
            t_end=float(t_end),
            label=label or "unknown",
            confidence=float(confidence or 0.0),
            scores=scores or {},
        )
        self.nonverbals.append(ev)
        self._latest_t = max(self._latest_t, ev.t_end)
        self._prune()

    def add_sentiment(
        self,
        label: str,
        t_start: float,
        t_end: float,
        confidence: float = 0.0,
        scores: Optional[Dict[str, float]] = None,
    ) -> None:
        ev = SentimentEvent(
            t_start=float(t_start),
            t_end=float(t_end),
            label=label or "neutral",
            confidence=float(confidence or 0.0),
            scores=scores or {},
        )
        self.sentiments.append(ev)
        self._latest_t = max(self._latest_t, ev.t_end)
        self._prune()

    # ----- maintenance ----------------------------------------------------- #

    def _prune(self) -> None:
        cutoff = self._latest_t - self.horizon
        if cutoff <= 0:
            return
        self.words = [w for w in self.words if w.t_end >= cutoff]
        self.nonverbals = [n for n in self.nonverbals if n.t_end >= cutoff]
        self.sentiments = [s for s in self.sentiments if s.t_end >= cutoff]

    def reset(self) -> None:
        self.words.clear()
        self.nonverbals.clear()
        self.sentiments.clear()
        self._latest_t = 0.0

    # ----- queries --------------------------------------------------------- #

    def _sentiment_at(self, t: float) -> Optional[SentimentEvent]:
        best = None
        for s in self.sentiments:
            if s.t_start <= t <= s.t_end:
                if best is None or s.confidence > best.confidence:
                    best = s
        if best is None:
            prior = [s for s in self.sentiments if s.t_start <= t]
            if prior:
                best = max(prior, key=lambda s: s.t_start)
        return best

    def _dedup_nonverbals(self) -> List[NonverbalEvent]:
        """Collapse repeated identical labels that are close in time."""
        usable = sorted(
            (n for n in self.nonverbals
             if n.confidence >= self.nv_min_conf
             and n.label not in self.NONVERBAL_IGNORE_INLINE),
            key=lambda n: n.t_start,
        )
        out: List[NonverbalEvent] = []
        for n in usable:
            if out and out[-1].label == n.label and (n.t_start - out[-1].t_end) <= self.nv_merge:
                out[-1].t_end = max(out[-1].t_end, n.t_end)
                out[-1].confidence = max(out[-1].confidence, n.confidence)
                continue
            out.append(NonverbalEvent(n.t_start, n.t_end, n.label, n.confidence, dict(n.scores)))
        return out

    def _nv_display(self, label: str) -> str:
        return self.NONVERBAL_DISPLAY.get(label, label)

    # ----- structured segments (the heart of fusion) ---------------------- #

    def segments(self) -> List[Dict[str, Any]]:
        """
        Build ordered transcript segments. Each segment is a run of words with
        no gap larger than segment_gap. Concurrent nonverbal events are attached
        inline (with the token index they follow) and the dominant sentiment for
        the segment is attached as context.
        """
        words = sorted(self.words, key=lambda w: w.t_start)
        nvs = self._dedup_nonverbals()

        segments: List[Dict[str, Any]] = []
        if not words:
            for n in nvs:
                segments.append({
                    "t_start": round(n.t_start, 3), "t_end": round(n.t_end, 3),
                    "text": "", "tokens": [],
                    "nonverbal": [{"label": n.label, "display": self._nv_display(n.label),
                                   "t": round(n.t_start, 3), "after_token": 0,
                                   "confidence": round(n.confidence, 3)}],
                    "sentiment": self._sentiment_dict(self._sentiment_at(n.t_start)),
                    "language": "",
                    "avg_confidence": 0.0,
                })
            return segments

        groups: List[List[WordEvent]] = [[words[0]]]
        for w in words[1:]:
            if (w.t_start - groups[-1][-1].t_end) > self.segment_gap:
                groups.append([w])
            else:
                groups[-1].append(w)

        attached = set()
        for grp in groups:
            seg_start, seg_end = grp[0].t_start, grp[-1].t_end
            tokens = [w.word for w in grp]
            seg_nvs = [n for n in nvs
                       if seg_start - self.segment_gap <= n.t_start <= seg_end + self.segment_gap]
            inline: List[Dict[str, Any]] = []
            for n in seg_nvs:
                attached.add(id(n))
                follow = sum(1 for w in grp if w.t_start <= n.t_start)
                follow = max(0, min(follow, len(tokens)))
                inline.append({
                    "label": n.label,
                    "display": self._nv_display(n.label),
                    "t": round(n.t_start, 3),
                    "after_token": follow,
                    "confidence": round(n.confidence, 3),
                })

            sent = self._dominant_sentiment(seg_start, seg_end)
            segments.append({
                "t_start": round(seg_start, 3),
                "t_end": round(seg_end, 3),
                "text": " ".join(tokens),
                "tokens": tokens,
                "nonverbal": inline,
                "sentiment": self._sentiment_dict(sent),
                "language": grp[0].language or "",
                "avg_confidence": round(sum(w.confidence for w in grp) / len(grp), 3),
            })

        # Non-verbals that landed in a speech gap (no nearby words) still matter —
        # surface them as their own standalone lines so crying/pain/applause during
        # silence never disappears from the transcript.
        for n in nvs:
            if id(n) in attached:
                continue
            segments.append({
                "t_start": round(n.t_start, 3), "t_end": round(n.t_end, 3),
                "text": "", "tokens": [],
                "nonverbal": [{"label": n.label, "display": self._nv_display(n.label),
                               "t": round(n.t_start, 3), "after_token": 0,
                               "confidence": round(n.confidence, 3)}],
                "sentiment": self._sentiment_dict(self._sentiment_at(n.t_start)),
                "language": "", "avg_confidence": 0.0,
            })

        segments.sort(key=lambda s: s["t_start"])
        return segments

    def _dominant_sentiment(self, t0: float, t1: float) -> Optional[SentimentEvent]:
        overlap = [s for s in self.sentiments if s.t_end >= t0 and s.t_start <= t1]
        if not overlap:
            return self._sentiment_at(t0)
        agg: Dict[str, float] = {}
        repr_ev: Dict[str, SentimentEvent] = {}
        for s in overlap:
            agg[s.label] = agg.get(s.label, 0.0) + s.confidence
            if s.label not in repr_ev or s.confidence > repr_ev[s.label].confidence:
                repr_ev[s.label] = s
        best_label = max(agg, key=agg.get)
        return repr_ev[best_label]

    @staticmethod
    def _sentiment_dict(s: Optional[SentimentEvent]) -> Optional[Dict[str, Any]]:
        if s is None:
            return None
        return {"label": s.label, "confidence": round(s.confidence, 3), "scores": s.scores}

    # ----- renderers ------------------------------------------------------- #

    def render_transcript(self, with_timestamps: bool = True,
                          with_sentiment: bool = True) -> str:
        """Human-readable fused transcript with inline nonverbal tags."""
        lines: List[str] = []
        prev_sentiment = None
        for seg in self.segments():
            inline_by_idx: Dict[int, List[str]] = {}
            for nv in seg["nonverbal"]:
                inline_by_idx.setdefault(nv["after_token"], []).append(f"[{nv['display']}]")

            parts: List[str] = []
            if 0 in inline_by_idx:
                parts.extend(inline_by_idx[0])
            for i, tok in enumerate(seg["tokens"], start=1):
                parts.append(tok)
                if i in inline_by_idx:
                    parts.extend(inline_by_idx[i])
            body = " ".join(parts).strip()
            if not body:
                body = " ".join(inline_by_idx.get(0, [])) or "..."

            prefix = ""
            if with_timestamps:
                prefix += f"[{_fmt_clock(seg['t_start'])}] "
            if with_sentiment and seg.get("sentiment"):
                slabel = seg["sentiment"]["label"]
                if slabel != prev_sentiment:
                    prefix += f"({slabel}) "
                    prev_sentiment = slabel
            lines.append(prefix + body)
        return "\n".join(lines)

    def build_llm_payload(self, window_seconds: Optional[float] = None) -> Dict[str, Any]:
        """
        Structured object for an LLM. If window_seconds is given, only the most
        recent slice of the timeline is included.
        """
        segs = self.segments()
        if window_seconds is not None and segs:
            cut = self._latest_t - window_seconds
            segs = [s for s in segs if s["t_end"] >= cut]

        in_window = lambda end: window_seconds is None or end >= (self._latest_t - window_seconds)
        sentiment_track = [
            {"t_start": round(s.t_start, 2), "t_end": round(s.t_end, 2),
             "label": s.label, "confidence": round(s.confidence, 3)}
            for s in sorted(self.sentiments, key=lambda x: x.t_start) if in_window(s.t_end)
        ]
        nonverbal_track = [
            {"t": round(n.t_start, 2), "label": n.label, "confidence": round(n.confidence, 3)}
            for n in self._dedup_nonverbals() if in_window(n.t_end)
        ]

        # rebuild a transcript scoped to the window
        return {
            "media_time": round(self._latest_t, 2),
            "fused_transcript": self._render_segments(segs),
            "segments": segs,
            "sentiment_track": sentiment_track,
            "nonverbal_track": nonverbal_track,
        }

    def _render_segments(self, segs: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        prev = None
        for seg in segs:
            inline_by_idx: Dict[int, List[str]] = {}
            for nv in seg["nonverbal"]:
                inline_by_idx.setdefault(nv["after_token"], []).append(f"[{nv['display']}]")
            parts: List[str] = []
            if 0 in inline_by_idx:
                parts.extend(inline_by_idx[0])
            for i, tok in enumerate(seg["tokens"], start=1):
                parts.append(tok)
                if i in inline_by_idx:
                    parts.extend(inline_by_idx[i])
            body = " ".join(parts).strip() or " ".join(inline_by_idx.get(0, [])) or "..."
            prefix = f"[{_fmt_clock(seg['t_start'])}] "
            if seg.get("sentiment"):
                sl = seg["sentiment"]["label"]
                if sl != prev:
                    prefix += f"({sl}) "
                    prev = sl
            lines.append(prefix + body)
        return "\n".join(lines)

    def build_llm_messages(
        self,
        window_seconds: Optional[float] = None,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Ready-to-send chat messages combining the fused transcript + tracks.
        Pair with audio_ops_dagster.core.llm_client.LLMClient.chat().
        """
        payload = self.build_llm_payload(window_seconds=window_seconds)
        system = system_prompt or (
            "Je bent een medische triage-assistent. Je krijgt een gefuseerde "
            "transcriptie van een telefonisch consult: de gesproken tekst, "
            "non-verbale geluiden tussen [haken] op hun tijdspositie, en de "
            "emotionele toon (sentiment) per segment. Vat bondig samen wat er "
            "speelt, noem opvallende non-verbale signalen (hoesten, kreunen, "
            "huilen, ademhaling) en de emotionele toestand. Verzin niets."
        )
        instr = instruction or (
            "Geef een korte klinische samenvatting (max 5 zinnen) en een "
            "urgentie-inschatting (laag / midden / hoog) met onderbouwing."
        )
        # Compact summary instead of dumping all segment JSON. The fused
        # transcript already carries the inline [non-verbal] tags and sentiment,
        # so a full JSON dump just bloats the prompt (thousands of tokens, far too
        # slow on CPU). We add only short aggregate counts.
        from collections import Counter
        nv_counts = Counter(n["label"] for n in payload["nonverbal_track"])
        nv_summary = ", ".join(f"{self._nv_display(k)} x{v}"
                               for k, v in nv_counts.most_common()) or "geen"
        sent_counts = Counter(s["label"] for s in payload["sentiment_track"])
        sent_summary = ", ".join(f"{k} ({v})" for k, v in sent_counts.most_common()) or "onbekend"
        user = (
            f"{instr}\n\n"
            f"=== Transcript (met non-verbale events en sentiment) ===\n"
            f"{payload['fused_transcript']}\n\n"
            f"=== Samenvatting ===\n"
            f"Non-verbale events: {nv_summary}.\n"
            f"Sentiment-verdeling: {sent_summary}.\n"
            f"Duur tot nu: {payload['media_time']:.0f}s."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]


__all__ = ["TranscriptFuser", "WordEvent", "NonverbalEvent", "SentimentEvent"]