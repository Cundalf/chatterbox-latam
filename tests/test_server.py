"""Unit tests for the Remote TTS server.

The heavy Chatterbox model is replaced by a fake, so the suite runs with
plain pytest + fastapi TestClient - no torch, no CUDA, no system libs.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server


class FakeModel:
    sr = 24000

    def __init__(self):
        self.conds = None
        self.prepared = []

    def prepare_conditionals(self, path, exaggeration=0.5):
        self.prepared.append((path, exaggeration))
        self.conds = object()

    def generate(self, text, language_id=None, exaggeration=0.5, cfg_weight=0.5, temperature=0.8):
        t = np.linspace(0, 0.5, int(self.sr * 0.5), endpoint=False)
        return np.sin(2 * np.pi * 220 * t).astype(np.float32)[None, :]


class StubSoundFile:
    def write(self, buf, data, sr, format=None, subtype=None):
        buf.write(b"RIFF" + np.asarray(data).tobytes())


@pytest.fixture()
def client(tmp_path, monkeypatch):
    server._ready = False
    server._model = None
    server._cond_cache.clear()
    server.MODELS_DIR = tmp_path / "latam"
    server.CONDS_DIR = tmp_path / "conds"
    server.VOICES_DIR = tmp_path / "voices"
    server.CONDS_DIR.mkdir()
    server.VOICES_DIR.mkdir()
    (server.CONDS_DIR / "default.wav").write_bytes(b"fake-default-clip")

    monkeypatch.setattr(server, "load_model", lambda: FakeModel())
    monkeypatch.setattr(server, "_encode_wav", lambda wav, sr: b"RIFF" + wav.tobytes())

    server._ensure_loaded()
    with TestClient(server.app) as c:
        yield c


@pytest.fixture()
def not_ready_client(monkeypatch):
    server._ready = False
    server._model = None
    server._cond_cache.clear()

    def boom():
        raise RuntimeError("no model here")

    monkeypatch.setattr(server, "load_model", boom)
    with TestClient(server.app) as c:
        yield c


def test_health_ready(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["model"] == "t3_es_mx_latam.safetensors"


def test_health_not_ready(not_ready_client):
    r = not_ready_client.get("/health")
    assert r.status_code == 503
    assert r.json()["ok"] is False


def test_speak_returns_wav_bytes(client):
    r = client.post("/speak", json={"text": "Buenas noches, SUB/WAVE.", "voice": "default"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content.startswith(b"RIFF")
    assert "X-TTS-Fell-Back" not in r.headers


def test_speak_unknown_voice_falls_back(client):
    r = client.post("/speak", json={"text": "Hola", "voice": "no-such-voice"})
    assert r.status_code == 200
    assert r.headers["X-TTS-Fell-Back"] == "true"
    assert r.headers["X-TTS-Voice-Used"] == "default"
    assert "not found" in r.headers["X-TTS-Fell-Back-Reason"]


def test_speak_custom_voice_no_fallback(client):
    (server.VOICES_DIR / "daniela.wav").write_bytes(b"fake-daniela-clip")
    r = client.post("/speak", json={"text": "Hola", "voice": "daniela"})
    assert r.status_code == 200
    assert "X-TTS-Fell-Back" not in r.headers
    assert "daniela" in server._model.prepared[0][0]


def test_speak_traversal_voice_rejected(client):
    r = client.post("/speak", json={"text": "Hola", "voice": "../../etc/passwd"})
    assert r.status_code == 200
    assert r.headers["X-TTS-Fell-Back"] == "true"
    assert r.headers["X-TTS-Voice-Used"] == "default"


def test_speak_rejects_blank_text(client):
    r = client.post("/speak", json={"text": "   ", "voice": "default"})
    assert r.status_code == 400


def test_speak_rejects_missing_text(client):
    r = client.post("/speak", json={"voice": "default"})
    assert r.status_code == 422


def test_speak_before_ready(not_ready_client):
    r = not_ready_client.post("/speak", json={"text": "Hola", "voice": "default"})
    assert r.status_code == 503


def test_conditionals_cached_per_voice(client):
    (server.VOICES_DIR / "juan.wav").write_bytes(b"fake-juan-clip")
    for _ in range(3):
        r = client.post("/speak", json={"text": "Hola", "voice": "juan"})
        assert r.status_code == 200
    assert len(server._model.prepared) == 1


def test_voices_endpoint(client):
    (server.VOICES_DIR / "juan.wav").write_bytes(b"fake-juan-clip")
    r = client.get("/voices")
    assert r.status_code == 200
    assert set(r.json()["voices"]) == {"default", "juan"}


def test_info_endpoint(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "chatterbox-latam"


def test_resolve_device_from_env(monkeypatch):
    monkeypatch.setenv("TTS_DEVICE", "cpu")
    assert server._resolve_device() == "cpu"


def test_resolve_device_empty_env_auto(monkeypatch):
    monkeypatch.setenv("TTS_DEVICE", "")
    assert server._resolve_device(has_cuda=True) == "cuda"
    assert server._resolve_device(has_cuda=False) == "cpu"


def test_resolve_device_unset_env_auto(monkeypatch):
    monkeypatch.delenv("TTS_DEVICE", raising=False)
    assert server._resolve_device(has_cuda=True) == "cuda"
    assert server._resolve_device(has_cuda=False) == "cpu"