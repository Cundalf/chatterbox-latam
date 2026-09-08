"""Idempotent downloader for the Chatterbox LatAm model assets.

Runs once as an init-container (docker-compose "downloader" service) before
the server starts. Downloads:

  - ResembleAI/Chatterbox-Multilingual-es-mx-latam:
      t3_es_mx_latam.safetensors      (2.14 GB, LatAm Spanish T3)
      s3gen_v3.pt                     (1.06 GB, S3Gen v3 decoder)
      grapheme_mtl_merged_expanded_v1.json  (tokenizer, vocab 2454)
  - ResembleAI/chatterbox:
      ve.pt                           (voice encoder, shared asset)

Then assembles the exact directory layout `from_local` expects:

    /models/latam/s3gen.pt            (copy of s3gen_v3.pt, torch.load)
    /models/latam/t3_es_mx_latam.safetensors
    /models/latam/grapheme_mtl_merged_expanded_v1.json
    /models/latam/ve.pt

and drops the bundled default voice clip into /models/conds/default.wav.

Re-running is safe: every asset is skipped when already present, so the
volume survives container rebuilds without re-downloading anything.
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models/latam"))
CONDS_DIR = Path(os.getenv("CONDS_DIR", "/models/conds"))

LATAM_REPO = "ResembleAI/Chatterbox-Multilingual-es-mx-latam"
BASE_REPO = "ResembleAI/chatterbox"

LATAM_FILES = [
    "t3_es_mx_latam.safetensors",
    "s3gen_v3.pt",
    "grapheme_mtl_merged_expanded_v1.json",
]
BASE_FILES = ["ve.pt"]

DEFAULT_VOICE_URL = (
    "https://storage.googleapis.com/chatterbox-demo-samples/"
    "mtl-v3-single-language-prompts/es-latam/es_mx_f1.wav"
)
DEFAULT_VOICE_MIN_BYTES = 100_000
CHECKPOINT_MIN_BYTES = 1_000_000


def fetch(repo: str, name: str, dest: Path, min_bytes: int) -> None:
    target = dest / name
    if target.is_file() and target.stat().st_size > min_bytes:
        print(f"[skip] {repo}/{name} already present")
        return
    from huggingface_hub import hf_hub_download  # lazy: only needed on first run

    print(f"[get ] {repo}/{name}")
    path = Path(hf_hub_download(repo, name, local_dir=dest))
    size = path.stat().st_size
    if size < min_bytes:
        raise RuntimeError(f"download too small ({size} bytes): {repo}/{name}")
    print(f"[ok  ] {path} ({size / 1e6:.1f} MB)")


def ensure_s3gen_pt() -> None:
    target = MODELS_DIR / "s3gen.pt"
    if target.is_file():
        print("[skip] s3gen.pt already present")
        return
    src_pt = MODELS_DIR / "s3gen_v3.pt"
    if src_pt.is_file():
        print("[copy] s3gen_v3.pt -> s3gen.pt")
        shutil.copy2(src_pt, target)
        return
    src_safe = MODELS_DIR / "s3gen_v3.safetensors"
    if src_safe.is_file():
        print("[conv] s3gen_v3.safetensors -> s3gen.pt")
        import torch  # lazy: only needed for the conversion fallback
        from safetensors.torch import load_file

        torch.save(load_file(str(src_safe)), target)
        return
    raise RuntimeError("s3gen_v3.pt missing - download did not complete")


def ensure_default_voice() -> None:
    CONDS_DIR.mkdir(parents=True, exist_ok=True)
    target = CONDS_DIR / "default.wav"
    if target.is_file() and target.stat().st_size > DEFAULT_VOICE_MIN_BYTES:
        print("[skip] default.wav already present")
        return
    print(f"[get ] {DEFAULT_VOICE_URL}")
    urllib.request.urlretrieve(DEFAULT_VOICE_URL, target)
    print(f"[ok  ] {target} ({target.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for name in LATAM_FILES:
            fetch(LATAM_REPO, name, MODELS_DIR, CHECKPOINT_MIN_BYTES)
        for name in BASE_FILES:
            fetch(BASE_REPO, name, MODELS_DIR, CHECKPOINT_MIN_BYTES)
        ensure_s3gen_pt()
        ensure_default_voice()
    except Exception as exc:
        print(f"[fail] {exc}", file=sys.stderr)
        return 1

    total = sum(p.stat().st_size for p in MODELS_DIR.iterdir() if p.is_file())
    print(f"OK: checkpoints ready in {MODELS_DIR} ({total / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())