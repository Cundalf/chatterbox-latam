FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf

RUN apt-get update \
 && apt-get install -y --no-install-recommends git libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Chatterbox pins torch==2.6.0 (pyproject.toml); CUDA 12.4 wheels support
# Turing (sm_75) GPUs such as the RTX 2080 Super. Install first so the
# chatterbox-tts dependency resolver reuses this exact build.
# Chatterbox is installed from GitHub master pinned to a commit: the PyPI
# release (0.1.7, 2025-06) predates the es-mx-latam finetune and its
# `from_local` cannot select t3_es_mx_latam.safetensors.
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 torchaudio==2.6.0 \
 && pip install \
        "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@5de7a54aa4e5e2baadb0182dde554908b48b85c2" \
        "fastapi>=0.110,<1" \
        "uvicorn[standard]>=0.29,<1" \
        "soundfile>=0.12"

RUN useradd --create-home --uid 1000 appuser

COPY server.py download_models.py /app/
WORKDIR /app

USER appuser
EXPOSE 5005

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5005"]