"""Chatterbox LatAm — SUB/WAVE Remote TTS engine.

Implements the Subwave-native Remote contract (SUB/WAVE >= 1.13.0):

    GET  /health  ->  200 {"ok": true}                      (503 until the model is warm)
    POST /speak   ->  200, request JSON {"text", "voice"}, WAV bytes in the response body

Optional fallback headers make silent voice substitutions visible:
X-TTS-Fell-Back, X-TTS-Voice-Used, X-TTS-Fell-Back-Reason.

The heavy model is loaded lazily on a background thread at startup, so the
process binds the port immediately and /health reports not-ready (503) until
the weights are in VRAM. SUB/WAVE polls /health every 30s and only marks the
engine as available once it returns 200.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("chatterbox-latam")

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models/latam"))
VOICES_DIR = Path(os.getenv("VOICES_DIR", "/voices"))
CONDS_DIR = Path(os.getenv("CONDS_DIR", "/models/conds"))
T3_MODEL = os.getenv("T3_MODEL", "t3_es_mx_latam.safetensors")
LANGUAGE = os.getenv("TTS_LANGUAGE", "es")
EXAGGERATION = float(os.getenv("TTS_EXAGGERATION", "0.5"))
CFG_WEIGHT = float(os.getenv("TTS_CFG_WEIGHT", "0.5"))
TEMPERATURE = float(os.getenv("TTS_TEMPERATURE", "0.8"))
DEFAULT_VOICE = "default"


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    voice: str = ""


_model = None
_ready = False
_model_lock = threading.Lock()
_gen_lock = threading.Lock()
_cond_cache: dict[tuple[str, float], object] = {}


def _resolve_device(has_cuda: bool | None = None) -> str:
    device = os.getenv("TTS_DEVICE", "").strip().lower()
    if device:
        return device
    if has_cuda is None:
        import torch  # lazy: keeps this module importable without torch (tests)

        has_cuda = torch.cuda.is_available()
    return "cuda" if has_cuda else "cpu"


def load_model():
    """Build the Chatterbox Multilingual LatAm model on CUDA (or CPU)."""
    device = _resolve_device()

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # lazy: heavy import

    logger.info("Loading %s on %s ...", T3_MODEL, device)
    model = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device=device, t3_model=T3_MODEL)
    logger.info("Model ready on %s", device)
    return model


def _ensure_loaded() -> None:
    global _model, _ready
    if _ready:
        return
    with _model_lock:
        if _ready:
            return
        try:
            _model = load_model()
            _ready = True
        except Exception:
            logger.error("Model load failed:\n%s", traceback.format_exc())
            raise


def _bootstrap() -> None:
    try:
        _ensure_loaded()
    except Exception:
        logger.error("Model unavailable; /health will report not ready")


def _safe_voice_name(voice: str) -> bool:
    parts = Path(voice).parts
    return bool(voice) and not Path(voice).is_absolute() and ".." not in parts


def resolve_clip(voice: str) -> tuple[Path, str, bool, str]:
    """Map a Remote voice string to a reference clip on disk.

    Returns (clip_path, voice_used, fell_back, reason). Unknown voices fall
    back to the bundled default clip and report the substitution so SUB/WAVE
    can log it.
    """
    default = CONDS_DIR / f"{DEFAULT_VOICE}.wav"

    if voice and voice != DEFAULT_VOICE:
        if _safe_voice_name(voice):
            for base in (VOICES_DIR, CONDS_DIR):
                for candidate in (base / f"{voice}.wav", base / voice):
                    if candidate.is_file():
                        return candidate, voice, False, ""
        if default.is_file():
            return default, DEFAULT_VOICE, True, f"voice '{voice}' not found"

    if default.is_file():
        return default, DEFAULT_VOICE, False, ""
    raise HTTPException(status_code=500, detail="no voice clip available (default clip missing)")


def _conditionals(model, clip: Path, exaggeration: float):
    """Reference-clip conditioning, cached per (clip, exaggeration)."""
    key = (str(clip), exaggeration)
    conds = _cond_cache.get(key)
    if conds is None:
        model.prepare_conditionals(str(clip), exaggeration=exaggeration)
        conds = model.conds
        _cond_cache[key] = conds
        logger.info("Conditioned on %s (exaggeration=%.2f)", clip.name, exaggeration)
    return conds


def _to_numpy(wav) -> np.ndarray:
    if hasattr(wav, "cpu"):
        wav = wav.squeeze(0).cpu()
    return np.asarray(wav, dtype=np.float32)


def _encode_wav(wav: np.ndarray, sr: int) -> bytes:
    import soundfile as sf  # lazy: keeps tests free of system libs

    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_bootstrap, daemon=True).start()
    yield


app = FastAPI(
    title="Chatterbox LatAm - SUB/WAVE Remote TTS",
    description="Subwave-native Remote TTS engine backed by ResembleAI Chatterbox Multilingual LatAm.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def info() -> dict:
    return {
        "service": "chatterbox-latam",
        "engine": "Remote (Subwave contract)",
        "model": T3_MODEL,
        "language": LANGUAGE,
    }


@app.get("/health")
def health() -> JSONResponse:
    body = {"ok": _ready}
    if _ready:
        body["model"] = T3_MODEL
        body["language"] = LANGUAGE
    return JSONResponse(body, status_code=200 if _ready else 503)


@app.get("/voices")
def list_voices() -> dict:
    names = set()
    for base in (VOICES_DIR, CONDS_DIR):
        if base.is_dir():
            names.update(p.stem for p in base.glob("*.wav"))
    return {"voices": sorted(names)}


@app.post("/speak")
def speak(req: SpeakRequest) -> Response:
    if not _ready:
        raise HTTPException(status_code=503, detail="model not ready")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    clip, voice_used, fell_back, reason = resolve_clip(req.voice)
    model = _model

    with _gen_lock:
        conds = _conditionals(model, clip, EXAGGERATION)
        model.conds = conds
        wav = model.generate(
            req.text.strip(),
            language_id=LANGUAGE,
            exaggeration=EXAGGERATION,
            cfg_weight=CFG_WEIGHT,
            temperature=TEMPERATURE,
        )

    headers = {}
    if fell_back:
        headers["X-TTS-Fell-Back"] = "true"
        headers["X-TTS-Voice-Used"] = voice_used
        headers["X-TTS-Fell-Back-Reason"] = reason

    return Response(content=_encode_wav(_to_numpy(wav), model.sr), media_type="audio/wav", headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5005)