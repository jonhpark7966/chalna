FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1     CHALNA_RESULTS_DIR=/data/results     HF_HOME=/models/huggingface     TORCH_HOME=/models/torch

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     ca-certificates     curl     ffmpeg     git     libsndfile1     && rm -rf /var/lib/apt/lists/*

# Node.js + Codex CLI (LLM refinement / translation). No in-image login: auth is
# provided at runtime by mounting the host's ~/.codex/auth.json (see compose).
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @openai/codex@0.133.0 \
    && rm -rf /var/lib/apt/lists/*
ENV CODEX_HOME=/root/.codex

COPY pyproject.toml README.md /app/
COPY external /app/external

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -e /app/external/VibeVoice \
    && pip install -e /app/external/Qwen3-ASR   # Qwen forced aligner (qwen_asr); without it,
    #                                             alignment is skipped and refinement splits are dropped

COPY src /app/src

RUN pip install -e /app

RUN mkdir -p /data/results /models

EXPOSE 7861

CMD ["uvicorn", "chalna.server:app", "--host", "0.0.0.0", "--port", "7861"]
