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
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 torchaudio==2.6.0 \
 && pip install \
        chatterbox-tts \
        fastapi>=0.110,<1 \
        "uvicorn[standard]>=0.29,<1" \
        soundfile>=0.12

RUN useradd --create-home --uid 1000 appuser

COPY server.py download_models.py /app/
WORKDIR /app

USER appuser
EXPOSE 5005

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5005"]