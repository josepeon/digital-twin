# Digital Twin — Jose Peon
# Multi-stage build: install deps, then copy app + data

# ── Stage 1: Dependencies ──────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps: build-essential for sentence-transformers' native extensions,
# git for the self-improving-agent git dependency in requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# A venv so the CPU-only torch installed first satisfies the torch that
# sentence-transformers asks for. The default PyPI wheel bundles several GB
# of CUDA libraries this CPU container never uses, and the image that size
# timed out Railway's image export.
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

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

# Create writable dirs: ChromaDB WAL + SQLite memory
RUN mkdir -p data/personal/rag_index data/personal/memory
RUN useradd --create-home appuser && \
    chown -R appuser:appuser data/personal/rag_index data/personal/memory
USER appuser

# Memory DB path — writable location for conversation history
ENV MEMORY_DB_PATH=/app/data/personal/memory/memory.db

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the API server (reads PORT from env, defaults to 8000)
CMD ["python", "-c", "from src.twin_api import main; main()"]
