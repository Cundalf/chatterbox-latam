"""Rebuild the bundled reference voices shipped in voices/.

Seven LatAm Spanish reference clips, each 24 kHz mono PCM-16 WAV, sourced as:

  es_mx_f1.wav  F  México      Official Chatterbox demo clip (MIT project)
  es_f1.wav     F  LatAm/neutral  Official Chatterbox V2 demo clip (FLAC)
  es_m1.wav     M  LatAm/neutral  Official Chatterbox V2 demo clip (FLAC)
  paisa_m1.wav  M  Colombia    David Bedoya's own voice, a-lo-paisa space (MIT)
  mx_f1.wav     F  México      Piper es_MX-claude, rendered reference (MIT)
  mx_m1.wav     M  México      Piper es_MX-ald, rendered reference (MIT)
  ar_f1.wav     F  Argentina   Piper es_AR-daniela, rendered reference (MIT)

Piper voices are synthesized with piper-tts (MIT) so no recording rights are
involved; the demo clips are Resemble AI's official Chatterbox samples.

Usage:
    pip install soundfile numpy piper-tts
    python scripts/build_voices.py [--voices-dir voices]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 24_000
MIN_SECONDS = 5.0
MAX_SECONDS = 12.0
SILENCE = 0.02

CURATED = [
    {
        "name": "es_mx_f1.wav",
        "url": "https://storage.googleapis.com/chatterbox-demo-samples/mtl-v3-single-language-prompts/es-latam/es_mx_f1.wav",
    },
    {
        "name": "es_f1.wav",
        "url": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
    },
    {
        "name": "es_m1.wav",
        "url": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_m1.flac",
    },
    {
        "name": "paisa_m1.wav",
        "url": "https://huggingface.co/spaces/jdavibedoya/a-lo-paisa/resolve/main/data/voice_reference.wav",
    },
]

PIPER_VOICES = [
    {
        "name": "mx_f1.wav",
        "model": "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx",
        "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx.json",
        "text": (
            "Hola, soy la voz del estudio. Esta noche vamos a viajar entre canciones, "
            "recuerdos y buenas historias. Quédate con nosotros."
        ),
    },
    {
        "name": "mx_m1.wav",
        "model": "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/ald/medium/es_MX-ald-medium.onnx",
        "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/ald/medium/es_MX-ald-medium.onnx.json",
        "text": (
            "Buenas noches, bienvenidos a la transmisión. Aquí arranca la música, "
            "el ritmo y la compañía de siempre."
        ),
    },
    {
        "name": "ar_f1.wav",
        "model": "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_AR/daniela/high/es_AR-daniela-high.onnx",
        "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_AR/daniela/high/es_AR-daniela-high.onnx.json",
        "text": (
            "Buenas noches, gracias por sintonizar. Esta noche la música nos acompaña, "
            "el dial se enciende y la ciudad se queda escuchando."
        ),
    },
]


def fetch(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 1_000:
        print(f"[skip] {dest.name} already present")
        return
    print(f"[get ] {url}")
    urllib.request.urlretrieve(url, dest)
    if dest.stat().st_size < 1_000:
        raise RuntimeError(f"download too small: {url}")


def trim_to_window(wav: np.ndarray, sr: int) -> np.ndarray:
    if len(wav) < int(MIN_SECONDS * sr):
        return wav
    edges = np.flatnonzero(np.abs(wav) > SILENCE)
    if edges.size:
        wav = wav[edges[0]:edges[-1] + 1]
    limit = int(MAX_SECONDS * sr)
    if len(wav) > limit:
        start = (len(wav) - limit) // 2
        wav = wav[start:start + limit]
    pad = int(0.2 * sr)
    return np.pad(wav, (pad, pad))


def normalize(wav: np.ndarray) -> np.ndarray:
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav / peak * 0.9
    return wav.astype(np.float32)


def write_target(src: Path, dst: Path) -> None:
    wav, sr = sf.read(src, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        import scipy.signal

        wav = scipy.signal.resample_poly(wav, TARGET_SR, sr)
    wav = trim_to_window(normalize(wav), TARGET_SR)
    sf.write(dst, wav, TARGET_SR, format="WAV", subtype="PCM_16")
    print(f"[wav ] {dst.name}: {len(wav) / TARGET_SR:.1f}s @ {TARGET_SR} Hz mono PCM16")


def render_piper(entry: dict, cache: Path, dst: Path) -> None:
    model = cache / Path(entry["model"]).name
    config = cache / Path(entry["config"]).name
    fetch(entry["model"], model)
    fetch(entry["config"], config)

    try:
        from piper import PiperVoice
    except ImportError:
        print("[fail] piper-tts not installed; run: pip install piper-tts", file=sys.stderr)
        sys.exit(1)

    voice = PiperVoice.load(str(model), config_path=str(config))
    tmp = dst.with_suffix(".raw.wav")
    with wave.open(str(tmp), "wb") as wav_file:
        voice.synthesize_wav(entry["text"], wav_file)
    write_target(tmp, dst)
    tmp.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voices-dir", type=Path, default=Path("voices"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/piper-voices"))
    args = parser.parse_args()

    args.voices_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        for entry in CURATED:
            tmp = args.cache_dir / entry["name"]
            fetch(entry["url"], tmp)
            write_target(tmp, args.voices_dir / entry["name"])

        for entry in PIPER_VOICES:
            render_piper(entry, args.cache_dir, args.voices_dir / entry["name"])
    except Exception as exc:
        print(f"[fail] {exc}", file=sys.stderr)
        return 1

    print(f"OK: voices ready in {args.voices_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())