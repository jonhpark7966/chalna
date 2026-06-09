FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHALNA_RESULTS_DIR=/data/results \
    CHALNA_SCRIBE_CACHE_DIR=/data/results/scribe_cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Node.js + Codex CLI (LLM refinement / translation). No in-image login: auth is
# provided at runtime by mounting the host's ~/.codex/auth.json (see compose).
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @openai/codex@0.133.0 \
    && rm -rf /var/lib/apt/lists/*
ENV CODEX_HOME=/root/.codex

COPY pyproject.toml README.md /app/
RUN python -m pip install --upgrade pip setuptools wheel

COPY src /app/src

RUN pip install -e /app

RUN mkdir -p /data/results/scribe_cache

EXPOSE 7861

CMD ["uvicorn", "chalna.server:app", "--host", "0.0.0.0", "--port", "7861"]
