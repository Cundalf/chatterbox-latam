<div align="center">

# chatterbox-latam · SUB/WAVE Remote TTS

A self-hosted **Latin American Spanish** voice for [SUB/WAVE Radio](https://www.getsubwave.com), built on
[Resemble AI's Chatterbox Multilingual LatAm](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-es-mx-latam).
Speaks the native **Remote engine** contract, so no custom code runs inside your radio stack.

[![SUB/WAVE](https://img.shields.io/badge/SUB/WAVE-v1.13.0-7c3aed)](https://www.getsubwave.com/manual/voices)
[![Chatterbox](https://img.shields.io/badge/Chatterbox-V3%20LatAm-2563eb)](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-es-mx-latam)
[![GPU](https://img.shields.io/badge/GPU-RTX%202080%20Super%20(8GB)-16a34a)](https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-20-series/rtx-2080-super/)
[![License](https://img.shields.io/badge/license-MIT-facc15)](LICENSE)

[English](README.md) · [Español](README.es.md)

</div>

---

## Why this project

SUB/WAVE ships six TTS engines out of the box, but none matched what we wanted for a
Spanish-language station:

- **Piper** is fast but robotic; **Kokoro** sounds better but its default voices aren't
  dialect-specific.
- **Chatterbox / PocketTTS** live in the optional `tts-heavy` sidecar, whose bundled image is
  **CPU-only** — driving a GPU requires rebuilding it with CUDA wheels.
- **Cloud** costs per use and depends on the network.

This project takes the route SUB/WAVE documents for exactly this case: a **Remote** TTS server you
run yourself, on your own GPU. One small FastAPI service speaks the native Remote contract
(`GET /health`, `POST /speak`) and renders every DJ line with a dedicated **Mexican / Latin
American Spanish** fine-tune of Chatterbox Multilingual V3, with zero-shot voice cloning from short
reference clips.

Everything runs in Docker on a machine with an **RTX 2080 Super (8 GB VRAM)**. The full model fits
in fp32 (~3.4 GB of weights), leaving comfortable headroom for activations.

## Features

- **Native SUB/WAVE Remote engine** — no wrapper, no `tts-heavy` rebuild, no OpenAI impersonation.
- **Init-container downloader** — the same image fetches the ~3.3 GB of model assets once into a
  Docker volume; the server starts only after the download finishes. Idempotent: rebuilds never
  re-download.
- **Zero-shot voice cloning** — drop a 6–10 s WAV in `voices/` and reference it by name from the
  SUB/WAVE persona editor. The bundled `default` voice works out of the box.
- **Per-voice conditioning cache** — reference clips are embedded once per voice, so every segment
  after the first renders without re-encoding the speaker embedding.
- **Fallback transparency** — unknown voices fall back to the default and report it via the
  `X-TTS-Fell-Back` headers SUB/WAVE logs (issue #238).
- **Health-aware readiness** — `/health` returns 503 until the model is warm, so SUB/WAVE only
  marks the engine ready when it can actually speak.
- **Non-root server, pinned CUDA 12.4 wheels (sm_75)** — Turing GPUs like the 2080 Super are fully
  supported by torch 2.6.0 cu124.

## Architecture

```
┌─────────────────────── SUB/WAVE stack (unchanged) ───────────────────────┐
│  Admin console ──► TTS engine: Remote ──► Server URL http://<host>:5005  │
│  Controller polls GET /health (30 s) and POSTs /speak (180 s timeout)    │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │  LAN / Tailscale (never 127.0.0.1)
┌───────────────────────────────▼──────────────────────────────────────────┐
│  This repo — chatterbox-latam (docker compose up -d)                     │
│                                                                          │
│  ┌──────────────┐  depends_on:            ┌───────────────────────────┐  │
│  │  downloader  │  service_completed_     │  server (FastAPI :5005)   │  │
│  │  (one-shot)  │  successfully ────────► │  /health  /speak  /voices │  │
│  └──────┬───────┘                         └─────────────┬─────────────┘  │
│         │ writes                                          │ CUDA (cu124) │
│  ┌──────▼──────────────────────────────────────────────────▼───────────┐ │
│  │  volume `models`      /models/latam  (checkpoints + tokenizer)      │ │
│  │                       /models/conds  (bundled default voice clip)   │ │
│  │                       /models/hf     (HuggingFace cache)            │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│  bind mount `./voices`  ──► your reference clips (daniela.wav, ...)      │
└──────────────────────────────────────────────────────────────────────────┘
```

## The model

| Asset | Source | Size | Role |
| --- | --- | --- | --- |
| `t3_es_mx_latam.safetensors` | es-mx-latam repo | 2.14 GB | LatAm Spanish T3 (text → speech tokens) |
| `s3gen_v3.pt` → `s3gen.pt` | es-mx-latam repo | 1.06 GB | S3Gen v3 decoder (speech tokens → audio) |
| `grapheme_mtl_merged_expanded_v1.json` | es-mx-latam repo | 70 KB | Tokenizer (vocab 2454) |
| `ve.pt` | `ResembleAI/chatterbox` | small | Voice encoder (speaker embedding) |
| `es_mx_f1.wav` | demo samples bucket | 1.6 MB | Bundled `default` voice clip |

The downloader assembles exactly the layout [`ChatterboxMultilingualTTS.from_local`](https://github.com/resemble-ai/chatterbox)
expects, including the `s3gen_v3.pt → s3gen.pt` copy (the loader does `torch.load("s3gen.pt")`).

## The contract

| Endpoint | Request | Response |
| --- | --- | --- |
| `GET /health` | — | `200 {"ok": true, "model": ...}` · `503 {"ok": false}` while warming up |
| `POST /speak` | `{"text": "...", "voice": "daniela"}` | `200`, WAV bytes, `Content-Type: audio/wav` |
| `GET /voices` | — | `{"voices": ["daniela", "default", ...]}` |
| `GET /` | — | service info (model, language) |

`voice` is free text forwarded by SUB/WAVE. Resolution order: `voices/<voice>.wav`, `voices/<voice>`,
`models/conds/<voice>.wav`, then the `default` clip. If the requested voice does not exist, the
server renders with `default` and sets `X-TTS-Fell-Back`, `X-TTS-Voice-Used` and
`X-TTS-Fell-Back-Reason` so SUB/WAVE logs the substitution instead of guessing.

## Requirements

| Component | Requirement |
| --- | --- |
| Docker | Compose v2 (`docker compose`), ≥ 24.x |
| GPU driver | NVIDIA driver ≥ 550 (CUDA 12.4) + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| GPU | 8 GB VRAM or more (tested: RTX 2080 Super) |
| Disk | ~14 GB free (image ~6–8 GB, models volume ~7 GB) |
| Network | Hugging Face + Google Cloud Storage reachable on first run |

## Quick start

```bash
git clone <this-repo> && cd chatterbox-latam-subwave
mkdir -p voices
cp .env.example .env          # optional — sensible defaults already set

docker compose up -d --build
docker compose logs -f chatterbox-latam-download   # ~3.3 GB on first run
```

The `server` service starts automatically once the downloader exits 0. Give it a minute to load
the weights, then:

```bash
curl http://<server-ip>:5005/health
curl -X POST http://<server-ip>:5005/speak -H "Content-Type: application/json" \
  -d '{"text":"Buenas noches, SUB/WAVE. Acá Daniela, acompañándote en la madrugada.","voice":"default"}' \
  --output test.wav
```

Use a LAN or Tailscale IP in place of `<server-ip>` — the URL you set in SUB/WAVE must be reachable
from the controller container, so `127.0.0.1` will not work.

## Voices

| Location | Purpose |
| --- | --- |
| `./voices/*.wav` | Your reference clips (bind-mounted, no rebuild needed) |
| `models` volume `/models/conds/default.wav` | Bundled default voice (es-MX female) |

To add a voice:

1. Record or cut a clip: **6–10 s, WAV, 16–24 kHz, mono, clean** (no music, no reverb).
2. Drop it in `voices/` as `daniela.wav`.
3. In SUB/WAVE, set the persona's **Remote voice** to `daniela`.

Changing a clip requires no rebuild and no volume reset — the conditioning cache re-embeds any clip
whose path was never seen before. Clips are cached in RAM per `(clip, exaggeration)`, so repeat
segments skip the embedding step.

## Configuring SUB/WAVE Radio (v1.13.x)

The latest SUB/WAVE manual documents the Remote engine under *Voices & TTS*
([manual/voices](https://www.getsubwave.com/manual/voices)). Setup:

1. **Deploy this server** on your GPU box and confirm `curl http://<server-ip>:5005/health`
   returns `{"ok": true}`.
2. **Open the admin console** → **TTS voice**.
3. **Set the engine to `Remote`** and fill **Server URL** with
   `http://<server-ip>:5005` — a LAN or Tailscale IP the controller container can reach,
   *not* `127.0.0.1`.
4. The console shows **ready** once the health probe passes (it polls every 30 s; you can also
   save and reopen the settings to force an immediate probe).
5. **Assign voices per persona**: on the *Personas* page, set the persona's **Remote voice** to the
   name of a clip in `voices/` (e.g. `daniela`) — it is forwarded verbatim to this server.
   `default` uses the bundled es-MX voice.
6. Optionally mix engines **per segment kind** (e.g. Remote for station IDs, Piper for routine
   time checks) — everything else falls through to the default engine.

Notes on SUB/WAVE behaviour:

- The controller uses a **180 s request timeout** — plenty for Chatterbox segments on an 8 GB GPU.
- If this server is down, unreachable, or `/health` fails, SUB/WAVE **falls back to Piper
  automatically**; your DJ never goes silent, it just changes voice.
- If a voice you requested does not exist here, you'll see the substitution in the controller log
  via the `X-TTS-Fell-Back` headers.

### Alternative: the OpenAI-compatible route

If you prefer, SUB/WAVE's **Cloud** engine with provider `OpenAI-compatible` can point at any
`/v1/audio/speech` server. This repo deliberately implements the **Remote** contract instead: it is
the native, no-dressing-up path, keeps per-request voice forwarding, and needs no `model` id
management.

## Tuning

Set any of these in `.env` before `docker compose up -d` (see `.env.example`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `CHATTERBOX_PORT` | `5005` | Host port for the server |
| `TTS_LANGUAGE` | `es` | Language id passed to the model |
| `TTS_EXAGGERATION` | `0.5` | Emotional prosody intensity (0.25–2.0) |
| `TTS_CFG_WEIGHT` | `0.5` | Classifier-free guidance — lower ≈ slower, more deliberate pacing |
| `TTS_TEMPERATURE` | `0.8` | Sampling temperature |
| `TTS_DEVICE` | *auto* | `cuda` or `cpu` (CPU is slow but usable for testing) |

Radio-DJ starting points, from the official Chatterbox tips:

- **Natural daytime links**: `TTS_EXAGGERATION=0.5`, `TTS_CFG_WEIGHT=0.5`.
- **Warmer, slower evening delivery**: `TTS_EXAGGERATION=0.7`, `TTS_CFG_WEIGHT=0.3`.
- If your reference speaker talks fast, lower `TTS_CFG_WEIGHT` to ~0.3.

## Operations

```bash
docker compose logs -f chatterbox-latam          # server logs
docker compose ps                                # status + health
docker compose up -d                             # restart after reboot (restart: unless-stopped)

# Force a re-download of the models (idempotent: skips what exists)
docker compose run --rm downloader

# Full reset (removes the ~7 GB volume — re-downloads next up)
docker compose down -v
```

The image is built once (`chatterbox-latam:latest`); code changes only need
`docker compose up -d --build`. The models volume persists across rebuilds.

## Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `/health` keeps returning 503 | Model still loading — watch `docker compose logs -f chatterbox-latam`. On a cold start (first load from disk) expect 20–60 s on an 8 GB GPU. |
| Server never starts | The downloader exited non-zero (network, HuggingFace outage). Check `docker compose logs chatterbox-latam-download`. |
| SUB/WAVE falls back to Piper | Server URL unreachable from the controller container. Use a LAN/Tailscale IP, never `127.0.0.1`; check firewall / `curl` from another machine. |
| `speak` returns 500 "model not ready" | Same as the 503 case — retry after the model finishes loading. |
| Empty audio file / silent segment | Requested voice not found → the server fell back to `default`; check logs. If the response body is empty it never renders. |
| CUDA out of memory | Chatterbox LatAm fits in 8 GB fp32 (~3.4 GB weights). Nothing else should share the GPU; check `nvidia-smi`. |
| Very slow generation | Server fell back to CPU (`TTS_DEVICE` unset and CUDA unavailable). Check `docker compose logs` for "Model ready on". |
| Download fails mid-way | Re-run `docker compose run --rm downloader` — it resumes/skips existing files. |
| GPU not used at all | NVIDIA Container Toolkit not installed on the host, or driver < 550 (CUDA 12.4). |

## Security

The server binds `0.0.0.0:5005` with **no authentication** — it renders whatever text it receives
and reads only WAV files. Run it inside your LAN, behind a firewall, or on Tailscale. Voice names
are sanitized (absolute paths and `..` are rejected and fall back to `default`).

Note: Chatterbox embeds an imperceptible [Perth watermark](https://github.com/resemble-ai/perth)
in every generated clip — this is by design (Resemble Detect can identify the audio).

## Repository layout

```
├── server.py            # FastAPI Remote engine (contract above)
├── download_models.py   # idempotent init-container downloader
├── Dockerfile           # python:3.11-slim + torch 2.6.0 cu124 + chatterbox-tts
├── docker-compose.yml   # downloader + server, GPU reservation, healthcheck
├── .env.example         # optional tuning knobs
├── voices/              # drop your reference clips here (gitignored)
└── tests/               # pytest suite with a fake model (no torch needed)
```

## License

MIT — see [LICENSE](LICENSE). Model weights are MIT licensed by Resemble AI
([chatterbox](https://github.com/resemble-ai/chatterbox), [es-mx-latam](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-es-mx-latam)).
SUB/WAVE Radio is MIT licensed by [perminder-klair/subwave](https://github.com/perminder-klair/subwave).

*Built because default engines weren't good enough. On the air, in your own voice.*