<div align="center">

# chatterbox-latam · Voz TTS Remota para SUB/WAVE

Una voz de **español latinoamericano** auto-alojada para [SUB/WAVE Radio](https://www.getsubwave.com),
basada en [Chatterbox Multilingual LatAm de Resemble AI](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-es-mx-latam).
Habla el contrato nativo del **Remote engine**, así que no corre código propio dentro de tu stack de radio.

[![SUB/WAVE](https://img.shields.io/badge/SUB/WAVE-v1.13.0-7c3aed)](https://www.getsubwave.com/manual/voices)
[![Chatterbox](https://img.shields.io/badge/Chatterbox-V3%20LatAm-2563eb)](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-es-mx-latam)
[![GPU](https://img.shields.io/badge/GPU-RTX%202080%20Super%20(8GB)-16a34a)](https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-20-series/rtx-2080-super/)
[![Licencia](https://img.shields.io/badge/license-MIT-facc15)](LICENSE)

[English](README.md) · [Español](README.es.md)

</div>

---

## Por qué este proyecto

SUB/WAVE trae seis motores TTS de fábrica, pero ninguno cumplió lo que queríamos para una
estación en español:

- **Piper** es rápido pero robótico; **Kokoro** suena mejor pero sus voces por defecto no son
  de un dialecto específico.
- **Chatterbox / PocketTTS** viven en el sidecar opcional `tts-heavy`, cuya imagen incluida es
  **solo CPU** — usarla con GPU exige recompilarla con wheels de CUDA.
- **Cloud** cuesta por uso y depende de la red.

Este proyecto toma el camino que SUB/WAVE documenta justamente para este caso: un servidor TTS
**Remote** que operás vos, en tu propia GPU. Un servicio FastAPI pequeño habla el contrato nativo
Remoto (`GET /health`, `POST /speak`) y lee cada línea del DJ con el fine-tune dedicado de
**español mexicano / latinoamericano** de Chatterbox Multilingual V3, con clonación de voz
zero-shot a partir de clips de referencia cortos.

Todo corre en Docker en una máquina con una **RTX 2080 Super (8 GB de VRAM)**. El modelo completo
entra en fp32 (~3.4 GB de pesos), dejando margen cómodo para las activaciones.

## Características

- **Remote engine nativo de SUB/WAVE** — sin wrappers, sin recompilar `tts-heavy`, sin hacerse
  pasar por OpenAI.
- **Descargador init-container** — la misma imagen baja los ~3.3 GB de pesos una sola vez a un
  volumen de Docker; el servidor arranca solo cuando la descarga terminó. Idempotente: reconstruir
  no vuelve a descargar nada.
- **Clonación de voz zero-shot** — poné un WAV de 6–10 s en `voices/` y referencialo por nombre
  desde el editor de personas de SUB/WAVE. La voz `default` incluida funciona de entrada.
- **Caché de condicionamiento por voz** — los clips de referencia se embeden una vez por voz; los
  segmentos siguientes no re-calculan la embedding del hablante.
- **Fallback transparente** — las voces desconocidas caen a `default` y lo reportan por los
  headers `X-TTS-Fell-Back` que SUB/WAVE registra (issue #238).
- **Readiness consciente de salud** — `/health` responde 503 hasta que el modelo está caliente,
  así SUB/WAVE solo marca el motor listo cuando realmente puede hablar.
- **Servidor no-root, wheels CUDA 12.4 fijados (sm_75)** — GPUs Turing como la 2080 Super están
  totalmente soportadas por torch 2.6.0 cu124.

## Arquitectura

```
┌─────────────────────── Stack SUB/WAVE (sin cambios) ─────────────────────┐
│  Consola admin ──► Motor TTS: Remote ──► Server URL http://<host>:5005   │
│  Controller consulta GET /health (30 s) y hace POST /speak (timeout 180 s)│
└───────────────────────────────┬──────────────────────────────────────────┘
                                │  LAN / Tailscale (nunca 127.0.0.1)
┌───────────────────────────────▼──────────────────────────────────────────┐
│  Este repo — chatterbox-latam (docker compose up -d)                     │
│                                                                          │
│  ┌──────────────┐  depends_on:            ┌───────────────────────────┐  │
│  │  downloader  │  service_completed_     │  server (FastAPI :5005)   │  │
│  │  (one-shot)  │  successfully ────────► │  /health  /speak  /voices │  │
│  └──────┬───────┘                         └─────────────┬─────────────┘  │
│         │ escribe                                          │ CUDA (cu124) │
│  ┌──────▼──────────────────────────────────────────────────▼───────────┐ │
│  │  volumen `models`    /models/latam  (checkpoints + tokenizer)       │ │
│  │                      /models/conds  (clip de voz default incluido)  │ │
│  │                      /models/hf     (caché de HuggingFace)          │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│  bind mount `./voices`  ──► tus clips de referencia (daniela.wav, ...)   │
└──────────────────────────────────────────────────────────────────────────┘
```

## El modelo

| Asset | Fuente | Tamaño | Rol |
| --- | --- | --- | --- |
| `t3_es_mx_latam.safetensors` | repo es-mx-latam | 2.14 GB | T3 LatAm (texto → speech tokens) |
| `s3gen_v3.pt` → `s3gen.pt` | repo es-mx-latam | 1.06 GB | Decoder S3Gen v3 (speech tokens → audio) |
| `grapheme_mtl_merged_expanded_v1.json` | repo es-mx-latam | 70 KB | Tokenizer (vocab 2454) |
| `ve.pt` | `ResembleAI/chatterbox` | pequeño | Voice encoder (embedding del hablante) |
| `es_mx_f1.wav` | bucket de demo samples | 1.6 MB | Clip de voz `default` incluido |

El descargador arma exactamente el layout que espera
[`ChatterboxMultilingualTTS.from_local`](https://github.com/resemble-ai/chatterbox), incluida la
copia `s3gen_v3.pt → s3gen.pt` (el loader hace `torch.load("s3gen.pt")`).

El paquete chatterbox-tts se instala **desde GitHub master, fijado a un commit** — la release de
PyPI (0.1.7, 2025-06) es anterior al finetune es-mx-latam y su `from_local` no puede seleccionar
`t3_es_mx_latam.safetensors`.

## El contrato

| Endpoint | Request | Response |
| --- | --- | --- |
| `GET /health` | — | `200 {"ok": true, "model": ...}` · `503 {"ok": false}` mientras calienta |
| `POST /speak` | `{"text": "...", "voice": "daniela"}` | `200`, bytes WAV, `Content-Type: audio/wav` |
| `GET /voices` | — | `{"voices": ["daniela", "default", ...]}` |
| `GET /` | — | info del servicio (modelo, idioma) |

`voice` es texto libre que SUB/WAVE reenvía tal cual. Orden de resolución: `voices/<voice>.wav`,
`voices/<voice>`, `models/conds/<voice>.wav` y por último el clip `default`. Si la voz pedida no
existe, el servidor renderiza con `default` y setea `X-TTS-Fell-Back`, `X-TTS-Voice-Used` y
`X-TTS-Fell-Back-Reason` para que SUB/WAVE registre la sustitución en vez de adivinarla.

## Requisitos

| Componente | Requisito |
| --- | --- |
| Docker | Compose v2 (`docker compose`), ≥ 24.x |
| Driver GPU | NVIDIA ≥ 550 (CUDA 12.4) + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| GPU | 8 GB de VRAM o más (probado: RTX 2080 Super) |
| Disco | ~14 GB libres (imagen ~6–8 GB, volumen de modelos ~7 GB) |
| Red | Acceso a Hugging Face y Google Cloud Storage en el primer arranque |

## Puesta en marcha

```bash
git clone <este-repo> && cd chatterbox-latam-subwave
mkdir -p voices
cp .env.example .env          # opcional — los defaults ya están bien

docker compose up -d --build
docker compose logs -f chatterbox-latam-download   # ~3.3 GB la primera vez
```

El servicio `server` arranca solo cuando el downloader termina con éxito. Dale un minuto para
cargar los pesos y después:

```bash
curl http://<ip-del-servidor>:5005/health
curl -X POST http://<ip-del-servidor>:5005/speak -H "Content-Type: application/json" \
  -d '{"text":"Buenas noches, SUB/WAVE. Acá Daniela, acompañándote en la madrugada.","voice":"default"}' \
  --output test.wav
```

Usá una IP de LAN o de Tailscale en lugar de `<ip-del-servidor>` — la URL que configures en
SUB/WAVE debe ser alcanzable desde el contenedor del controller, así que `127.0.0.1` no funciona.

## Voces

| Ubicación | Propósito |
| --- | --- |
| `./voices/*.wav` | Tus clips de referencia (bind-mount, sin rebuild) |
| Volumen `models`, `/models/conds/default.wav` | Voz default incluida (femenina es-MX) |

Para agregar una voz:

1. Grabá o cortá un clip: **6–10 s, WAV, 16–24 kHz, mono, limpio** (sin música ni reverb).
2. Ponelo en `voices/` como `daniela.wav`.
3. En SUB/WAVE, seteá la **Remote voice** de la persona a `daniela`.

Cambiar un clip no requiere rebuild ni resetear el volumen — la caché de condicionamiento re-embede
cualquier clip cuya ruta no haya visto antes. Los clips se cachean en RAM por `(clip, exaggeration)`,
así los segmentos repetidos se saltan el paso de embedding.

## Configurar SUB/WAVE Radio (v1.13.x)

El manual actual de SUB/WAVE documenta el Remote engine en *Voices & TTS*
([manual/voices](https://www.getsubwave.com/manual/voices)). Setup:

1. **Desplegá este servidor** en tu máquina con GPU y confirmá que
   `curl http://<ip-del-servidor>:5005/health` responde `{"ok": true}`.
2. **Abrí la consola de admin** → **TTS voice**.
3. **Elegí el motor `Remote`** y completá **Server URL** con
   `http://<ip-del-servidor>:5005` — una IP de LAN o Tailscale alcanzable desde el contenedor del
   controller, *no* `127.0.0.1`.
4. La consola muestra **ready** cuando el health probe pasa (consulta cada 30 s; también podés
   guardar y reabrir los ajustes para forzar un probe inmediato).
5. **Asigná voces por persona**: en la página *Personas*, seteá la **Remote voice** de la persona
   al nombre de un clip de `voices/` (ej. `daniela`) — se reenvía tal cual a este servidor.
   `default` usa la voz es-MX incluida.
6. Opcional: mezclá motores **por tipo de segmento** (ej. Remote para los IDs de estación, Piper
   para los avisos de hora) — el resto cae al motor por defecto.

Notas sobre el comportamiento de SUB/WAVE:

- El controller usa un **timeout de petición de 180 s** — de sobra para segmentos de Chatterbox en
  una GPU de 8 GB.
- Si este servidor está caído, es inalcanzable o `/health` falla, SUB/WAVE **cae a Piper
  automáticamente**; tu DJ nunca queda en silencio, solo cambia de voz.
- Si pedís una voz que no existe acá, vas a ver la sustitución en el log del controller vía los
  headers `X-TTS-Fell-Back`.

### Alternativa: la ruta compatible con OpenAI

Si preferís, el motor **Cloud** de SUB/WAVE con provider `OpenAI-compatible` puede apuntar a
cualquier servidor `/v1/audio/speech`. Este repo implementa a propósito el contrato **Remote**:
es el camino nativo, sin disfrazar nada, mantiene el reenvío de voz por petición y no requiere
manejar ids de `model`.

## Ajuste fino

Cualquiera de estas variables se setea en `.env` antes de `docker compose up -d` (ver
`.env.example`):

| Variable | Default | Significado |
| --- | --- | --- |
| `CHATTERBOX_PORT` | `5005` | Puerto del servidor en el host |
| `TTS_LANGUAGE` | `es` | Id de idioma que recibe el modelo |
| `TTS_EXAGGERATION` | `0.5` | Intensidad de la prosodia emocional (0.25–2.0) |
| `TTS_CFG_WEIGHT` | `0.5` | Guía classifier-free — más bajo ≈ ritmo más lento y deliberado |
| `TTS_TEMPERATURE` | `0.8` | Temperatura de muestreo |
| `TTS_DEVICE` | *auto* | `cuda` o `cpu` (CPU es lento pero útil para probar) |

Puntos de partida para un DJ de radio, según los tips oficiales de Chatterbox:

- **Links diurnos naturales**: `TTS_EXAGGERATION=0.5`, `TTS_CFG_WEIGHT=0.5`.
- **Noche más cálida y pausada**: `TTS_EXAGGERATION=0.7`, `TTS_CFG_WEIGHT=0.3`.
- Si tu locutor de referencia habla rápido, bajá `TTS_CFG_WEIGHT` a ~0.3.

## Operación

```bash
docker compose logs -f chatterbox-latam          # logs del servidor
docker compose ps                                # estado + health
docker compose up -d                             # rearranque tras reboot (restart: unless-stopped)

# Forzar re-descarga de los modelos (idempotente: saltea lo que ya existe)
docker compose run --rm downloader

# Reset completo (borra el volumen de ~7 GB — re-descarga en el próximo up)
docker compose down -v
```

La imagen se construye una vez (`chatterbox-latam:latest`); los cambios de código solo requieren
`docker compose up -d --build`. El volumen de modelos persiste entre rebuilds.

## Solución de problemas

| Síntoma | Causa probable / solución |
| --- | --- |
| `/health` sigue devolviendo 503 | El modelo sigue cargando — mirá `docker compose logs -f chatterbox-latam`. En arranque en frío (primera carga desde disco) esperá 20–60 s en una GPU de 8 GB. |
| El servidor nunca arranca | El downloader salió con error (red, caída de HuggingFace). Revisá `docker compose logs chatterbox-latam-download`. |
| SUB/WAVE cae a Piper | La URL del servidor no es alcanzable desde el contenedor del controller. Usá una IP de LAN/Tailscale, nunca `127.0.0.1`; revisá firewall / `curl` desde otra máquina. |
| `speak` devuelve 500 "model not ready" | Igual que el caso 503 — reintentá cuando el modelo termine de cargar. |
| Archivo de audio vacío / segmento silencioso | La voz pedida no existe → el servidor cayó a `default`; revisá los logs. Si el body está vacío, nunca se renderizó. |
| CUDA out of memory | Chatterbox LatAm entra en 8 GB fp32 (~3.4 GB de pesos). Nada más debería compartir la GPU; revisá `nvidia-smi`. |
| Generación muy lenta | El servidor cayó a CPU (`TTS_DEVICE` sin setear y CUDA no disponible). Revisá en los logs "Model ready on". |
| La descarga falla a la mitad | Volvé a correr `docker compose run --rm downloader` — retoma/salte los archivos existentes. |
| La GPU no se usa | NVIDIA Container Toolkit no está instalado en el host, o el driver es < 550 (CUDA 12.4). |

## Seguridad

El servidor escucha en `0.0.0.0:5005` **sin autenticación** — renderiza cualquier texto que reciba
y solo lee archivos WAV. Correlo dentro de tu LAN, detrás de un firewall o en Tailscale. Los
nombres de voz se sanitizan (rutas absolutas y `..` se rechazan y caen a `default`).

Nota: Chatterbox incrusta una [marca de agua Perth](https://github.com/resemble-ai/perth)
imperceptible en cada clip generado — es por diseño (Resemble Detect puede identificar el audio).

## Estructura del repo

```
├── server.py            # FastAPI Remote engine (contrato de arriba)
├── download_models.py   # descargador idempotente (init-container)
├── Dockerfile           # python:3.11-slim + torch 2.6.0 cu124 + chatterbox-tts
├── docker-compose.yml   # downloader + server, reserva de GPU, healthcheck
├── .env.example         # knobs de ajuste opcionales
├── voices/              # poné tus clips de referencia acá (gitignored)
└── tests/               # suite pytest con modelo fake (no necesita torch)
```

## Licencia

MIT — ver [LICENSE](LICENSE). Los pesos del modelo son MIT de Resemble AI
([chatterbox](https://github.com/resemble-ai/chatterbox), [es-mx-latam](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-es-mx-latam)).
SUB/WAVE Radio es MIT de [perminder-klair/subwave](https://github.com/perminder-klair/subwave).

*Nacido porque los motores por defecto no alcanzaban. Al aire, con tu propia voz.*