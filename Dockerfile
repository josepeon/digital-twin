# Digital Twin — Jose Peon
# Multi-stage build: install deps, then copy app + data

# ── Stage 1: Dependencies ──────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for sentence-transformers (builds native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ src/
COPY pyproject.toml .

# Copy personal data (RAG index + voice IDs) — these are gitignored but needed at runtime.
# The Docker build context includes them via .dockerignore exceptions.
COPY data/personal/rag_index/ data/personal/rag_index/
COPY data/personal/voice/voice_id_en.txt data/personal/voice/voice_id_en.txt
COPY data/personal/voice/voice_id_es.txt data/personal/voice/voice_id_es.txt
COPY data/personal/identity/ data/personal/identity/

# Pre-download the sentence-transformers model so startup is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the API server
CMD ["python", "-m", "uvicorn", "src.twin_api:app", "--host", "0.0.0.0", "--port", "8000"]
