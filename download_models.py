"""Idempotent downloader for the Chatterbox LatAm model assets.

Runs once as an init-container (docker-compose "downloader" service) before
the server starts. Downloads:

  - ResembleAI/Chatterbox-Multilingual-es-mx-latam:
      t3_es_mx_latam.safetensors      (2.14 GB, LatAm Spanish T3)
      grapheme_mtl_merged_expanded_v1.json  (tokenizer, vocab 2454)
  - ResembleAI/chatterbox (base repo):
      ve.pt                           (voice encoder, shared asset)
      s3gen.pt                        (1.06 GB, S3Gen v3 decoder + tokenizer)

The base repo's s3gen.pt is required: it is the checkpoint the current
Chatterbox code loads (torch.load + strict load_state_dict) and it is the
only one that carries the S3 tokenizer buffers (tokenizer._mel_filters,
tokenizer.window). The per-language repos only ship s3gen_v3.pt, which
lacks those keys and fails to load.

Then assembles the exact directory layout `from_local` expects:

    /models/latam/s3gen.pt
    /models/latam/t3_es_mx_latam.safetensors
    /models/latam/grapheme_mtl_merged_expanded_v1.json
    /models/latam/ve.pt

and drops the bundled default voice clip into /models/conds/default.wav.

Re-running is safe: every asset is skipped when already present, so the
volume survives container rebuilds without re-downloading anything.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models/latam"))
CONDS_DIR = Path(os.getenv("CONDS_DIR", "/models/conds"))

LATAM_REPO = "ResembleAI/Chatterbox-Multilingual-es-mx-latam"
BASE_REPO = "ResembleAI/chatterbox"

# (filename, minimum expected size in bytes) - sanity guard against
# truncated downloads; sizes are the actual file sizes on HuggingFace.
LATAM_FILES = [
    ("t3_es_mx_latam.safetensors", 2_000_000_000),
    ("grapheme_mtl_merged_expanded_v1.json", 50_000),
]
BASE_FILES = [("ve.pt", 1_000_000)]
S3GEN_MIN_BYTES = 1_000_000_000

DEFAULT_VOICE_URL = (
    "https://storage.googleapis.com/chatterbox-demo-samples/"
    "mtl-v3-single-language-prompts/es-latam/es_mx_f1.wav"
)
DEFAULT_VOICE_MIN_BYTES = 100_000


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


def _s3gen_keys_ok(path: Path) -> bool:
    import torch  # lazy: only needed for validation

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    return "tokenizer._mel_filters" in state and "tokenizer.window" in state


def ensure_s3gen_pt() -> None:
    target = MODELS_DIR / "s3gen.pt"
    if target.is_file() and _s3gen_keys_ok(target):
        print("[skip] s3gen.pt already present (tokenizer keys verified)")
        return
    if target.is_file():
        print("[stale] s3gen.pt lacks tokenizer keys - re-downloading")
        target.unlink()
    fetch(BASE_REPO, "s3gen.pt", MODELS_DIR, S3GEN_MIN_BYTES)
    if not _s3gen_keys_ok(target):
        raise RuntimeError("s3gen.pt downloaded but missing tokenizer keys")
    print("[ok  ] s3gen.pt verified (tokenizer keys present)")


def fix_models_permissions(uid: int = 1000, gid: int = 1000) -> None:
    """Give the server user (appuser, uid 1000) ownership of /models.

    The downloader runs as root; without this the read-only server still
    works, but the HuggingFace cache inside the volume stays unwritable
    and the server logs cache warnings at import time.
    """
    if os.geteuid() != 0:
        return
    base = MODELS_DIR.parent
    if not base.exists():
        return
    os.chown(base, uid, gid)
    for root, dirs, files in os.walk(base):
        for name in dirs + files:
            os.chown(os.path.join(root, name), uid, gid)


def ensure_default_voice() -> None:
    CONDS_DIR.mkdir(parents=True, exist_ok=True)
    target = CONDS_DIR / "default.wav"
    if target.is_file() and target.stat().st_size > DEFAULT_VOICE_MIN_BYTES:
        print("[skip] default.wav already present")
        return
    print(f"[get ] {DEFAULT_VOICE_URL}")
    urllib.request.urlretrieve(DEFAULT_VOICE_URL, target)
    print(f"[ok  ] {target} ({target.stat().st_size / 1e6:.1f} MB)")


def remove_legacy_s3gen_v3() -> None:
    leftover = MODELS_DIR / "s3gen_v3.pt"
    if leftover.is_file():
        leftover.unlink()
        print(f"[clean] removed unused {leftover.name}")


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for name, min_bytes in LATAM_FILES:
            fetch(LATAM_REPO, name, MODELS_DIR, min_bytes)
        for name, min_bytes in BASE_FILES:
            fetch(BASE_REPO, name, MODELS_DIR, min_bytes)
        ensure_s3gen_pt()
        ensure_default_voice()
        remove_legacy_s3gen_v3()
        fix_models_permissions()
    except Exception as exc:
        print(f"[fail] {exc}", file=sys.stderr)
        return 1

    total = sum(p.stat().st_size for p in MODELS_DIR.iterdir() if p.is_file())
    print(f"OK: checkpoints ready in {MODELS_DIR} ({total / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())