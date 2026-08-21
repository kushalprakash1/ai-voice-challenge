import asyncio
import json
import struct
import threading
import wave

import pytest

from voiceprobe.v3.accent import (
    ACCENT_MELO_INDIA,
    AccentCache,
    PersistentMeloIndiaRenderer,
    accent_cache_key,
    accent_cache_preflight,
    accent_mode_from_environment,
    medication_refill_phrase_inventory,
    metadata_for_wav,
    write_metadata,
)
from voiceprobe.v3.audiosocket_kokoro import (
    AudioSocketKokoroConfig,
    AudioSocketKokoroSpeechTask,
    KokoroTelephonyRenderer,
)


class Recorder:
    def __init__(self):
        self.events = []

    def record_event(self, name, **fields):
        self.events.append((name, fields))


class Frame:
    def __init__(self, text):
        self.text = text


def add_entry(cache: AccentCache, text: str, *, sample=1200) -> bytes:
    _key, metadata_path, wav_path = cache.paths(text)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    pcm = struct.pack("<160h", *([sample] * 160))
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(pcm)
    write_metadata(metadata_for_wav(text=text, wav_path=wav_path), metadata_path)
    return pcm


def fallback_renderer(cache=None):
    calls = []
    renderer = KokoroTelephonyRenderer(
        pipeline="normal",
        accent_cache=cache,
        synthesize_fn=lambda **kwargs: calls.append(kwargs["text"]) or [0.2],
        normalize_fn=lambda text: text,
        resample_fn=lambda audio: audio,
        encode_fn=lambda audio: b"normal-pcm",
    )
    return renderer, calls


def sticky_melo_renderer(cache):
    kokoro_calls = []
    melo_calls = []
    renderer = KokoroTelephonyRenderer(
        pipeline="normal",
        accent_cache=cache,
        synthesize_fn=lambda **kwargs: kokoro_calls.append(kwargs["text"]) or [0.2],
        normalize_fn=lambda text: text,
        resample_fn=lambda audio: audio,
        encode_fn=lambda audio: b"kokoro-pcm",
        melo_synthesize_fn=lambda text: melo_calls.append(text) or b"melo-en-india-pcm",
    )
    return renderer, kokoro_calls, melo_calls


def test_default_accent_is_off_and_only_india_is_supported(monkeypatch):
    monkeypatch.delenv("VOICEPROBE_ACCENT_MODE", raising=False)
    assert accent_mode_from_environment() == "none"
    monkeypatch.setenv("VOICEPROBE_ACCENT_MODE", "melo_india")
    assert accent_mode_from_environment() == ACCENT_MELO_INDIA
    monkeypatch.setenv("VOICEPROBE_ACCENT_MODE", "british")
    with pytest.raises(ValueError):
        accent_mode_from_environment()


def test_cache_key_uses_exact_text_identity():
    assert accent_cache_key("The medication is lisinopril.") != accent_cache_key(
        "The medication is lisinopril"
    )


def test_cache_metadata_audio_and_renderer_hit(tmp_path):
    text = "The medication is lisinopril."
    cache = AccentCache(tmp_path, scenario="medication-refill-correction")
    pcm = add_entry(cache, text)
    renderer, calls = fallback_renderer(cache)
    rendered, fields = renderer.render_with_metadata(text)
    assert rendered == pcm
    assert calls == []
    assert fields["accent_cache_hit"] is True
    assert fields["accent_fallback_used"] is False
    assert fields["accent_speaker"] == "EN_INDIA"


def test_metadata_text_mismatch_is_rejected(tmp_path):
    cache = AccentCache(tmp_path)
    text = "Ten milligrams."
    add_entry(cache, text)
    _, metadata_path, _ = cache.paths(text)
    metadata = json.loads(metadata_path.read_text())
    metadata["exact_source_text"] = "Ten milligrams"
    metadata_path.write_text(json.dumps(metadata))
    assert not cache.lookup(text).hit


def test_melo_india_cache_miss_uses_same_voice_and_records_diagnostic(tmp_path, monkeypatch):
    monkeypatch.delenv("VOICEPROBE_V3_LIVE", raising=False)

    async def run():
        cache = AccentCache(tmp_path, scenario="medication-refill-correction")
        renderer, kokoro_calls, melo_calls = sticky_melo_renderer(cache)
        recorder = Recorder()
        sent = []
        speech = AudioSocketKokoroSpeechTask(
            connection=object(), renderer=renderer, send_lock=threading.Lock(), recorder=recorder,
            config=AudioSocketKokoroConfig(echo_guard_seconds=0),
            send_audio_fn=lambda connection, pcm, **kwargs: sent.append(pcm),
        )
        await speech.queue_frame(Frame("Unexpected exact response."))
        await speech.wait_for_idle(timeout_seconds=2)
        assert sent == [b"melo-en-india-pcm"]
        assert kokoro_calls == []
        assert melo_calls == ["Unexpected exact response."]
        prepared = next(fields for name, fields in recorder.events if name == "v3_playback_prepared")
        assert prepared["accent_speaker"] == "EN_INDIA"
        assert prepared["tts_backend"] == "MeloTTS"
        assert prepared["accent_fallback_used"] is False
        assert prepared["accent_same_voice_miss_rendered"] is True
        miss = next(fields for name, fields in recorder.events if name == "v3_accent_cache_miss")
        assert miss["scenario"] == "medication-refill-correction"
        assert miss["exact_response_text"] == "Unexpected exact response."
        assert miss["fallback_used"] is False
        assert miss["same_voice_miss_rendered"] is True

    asyncio.run(run())


@pytest.mark.parametrize(
    "text",
    (
        "Could you rephrase that scheduling question?",
        "This sentence is intentionally not in the accent cache.",
    ),
)
def test_arbitrary_melo_india_cache_misses_never_call_kokoro(tmp_path, text):
    cache = AccentCache(tmp_path, scenario="farthest-date-scheduling")
    renderer, kokoro_calls, melo_calls = sticky_melo_renderer(cache)

    rendered, fields = renderer.render_with_metadata(text)

    assert rendered == b"melo-en-india-pcm"
    assert melo_calls == [text]
    assert kokoro_calls == []
    assert fields["accent_cache_hit"] is False
    assert fields["accent_speaker"] == "EN_INDIA"
    assert fields["tts_backend"] == "MeloTTS"
    assert fields["accent_fallback_used"] is False


def test_accent_preflight_requires_complete_medication_inventory(tmp_path, capsys):
    cache = AccentCache(tmp_path)
    phrases = medication_refill_phrase_inventory()
    for text in phrases:
        add_entry(cache, text)
    result = accent_cache_preflight(
        scenario="medication-refill-correction", mode="melo_india", cache=cache
    )
    assert result["cached"] == result["required"] == len(phrases)
    assert result["coverage"] == 100.0
    assert "missing=0" in capsys.readouterr().out


def test_persistent_melo_start_is_nonblocking_and_miss_waits_for_same_worker(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    rendered = []

    def fake_start(self):
        self._process = type("Process", (), {"pid": 4321, "poll": lambda self: None})()

        def become_ready():
            started.set()
            release.wait(2)
            self._ready.set()

        threading.Thread(target=become_ready, daemon=True).start()

    monkeypatch.setattr(PersistentMeloIndiaRenderer, "_start", fake_start)
    renderer = PersistentMeloIndiaRenderer(startup_timeout=1)
    assert started.wait(0.2)
    assert renderer.ready is False
    assert renderer.pid == 4321

    def fake_render(text):
        renderer.wait_until_ready()
        rendered.append(text)
        return b"melo-en-india-pcm"

    waiter = threading.Thread(target=lambda: fake_render("uncached"), daemon=True)
    waiter.start()
    assert rendered == []
    release.set()
    waiter.join(1)
    assert rendered == ["uncached"]
    assert renderer.ready is True


def test_cached_accent_uses_cached_audio_without_changing_callbacks(tmp_path):
    async def run():
        text = "Can you add lisinopril to my demo profile so I can continue with the refill request?"
        cache = AccentCache(tmp_path, scenario="medication-refill-correction")
        add_entry(cache, text)
        renderer, _ = fallback_renderer(cache)
        recorder = Recorder()
        sent, callbacks = [], []
        speech = AudioSocketKokoroSpeechTask(
            connection=object(), renderer=renderer, send_lock=threading.Lock(), recorder=recorder,
            config=AudioSocketKokoroConfig(echo_guard_seconds=0),
            send_audio_fn=lambda connection, pcm, **kwargs: sent.append(pcm),
            on_playback_finished=lambda: callbacks.append("spoken"),
        )
        await speech.queue_frame(Frame(text))
        await speech.wait_for_idle(timeout_seconds=2)
        assert sent
        assert callbacks == ["spoken"]
        prepared = next(fields for name, fields in recorder.events if name == "v3_playback_prepared")
        assert prepared["accent_cache_hit"] is True

    asyncio.run(run())
