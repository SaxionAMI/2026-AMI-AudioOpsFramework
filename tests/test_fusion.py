from audio_ops_dagster.extras.fusion import TranscriptFuser


def _fill(f):
    f.add_words([{"word": "hello", "t_start": 0.1, "t_end": 0.4},
                 {"word": "there", "t_start": 0.5, "t_end": 0.8}])
    f.add_nonverbal(label="cough", t_start=0.6, t_end=0.9, confidence=0.9, scores={})
    f.add_sentiment(label="calm", t_start=0, t_end=1, confidence=0.8, scores={})
    return f


def test_english_default(monkeypatch):
    monkeypatch.delenv("FUSION_LANGUAGE", raising=False)
    f = _fill(TranscriptFuser())
    assert "[coughs]" in f.render_transcript()
    assert "Summary" in f.build_llm_messages()[1]["content"]


def test_dutch_pack():
    f = _fill(TranscriptFuser(language="nl"))
    assert "[hoest]" in f.render_transcript()
    assert "Samenvatting" in f.build_llm_messages()[1]["content"]


def test_custom_prompts_win():
    msgs = _fill(TranscriptFuser()).build_llm_messages(system_prompt="SYS", instruction="DO")
    assert msgs[0]["content"] == "SYS" and msgs[1]["content"].startswith("DO")
