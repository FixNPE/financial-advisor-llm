# syntax=docker/dockerfile:1.7
# ---------- Stage 1: builder ----------
# Installs deps into an isolated venv so the runtime image only carries the
# resolved wheels — no pip cache, no compilers, no build tooling.
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential is only needed to compile any wheels that lack a manylinux
# build (e.g. some transitive C-extension pins). Discarded with the stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt


# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# curl is used by the HEALTHCHECK; libgomp1 is a common runtime dep for
# sentence-transformers / numpy on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — the app never needs to write outside /app/data.
RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# Source last so code edits don't invalidate the deps layer.
COPY --chown=app:app src ./src
COPY --chown=app:app app ./app
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app corpus ./corpus
COPY --chown=app:app data/seed ./data/seed

# Writable dirs for the SQLite HITL DB, Chroma index, price cache, plus the
# HF cache dir (sentence-transformers downloads the embedder to ~/.cache).
RUN mkdir -p /app/data/chroma /app/data/prices /app/data/processed /app/data/raw /app/.cache \
    && chown -R app:app /app/data /app/.cache

# Baked at build time so the image is self-contained on platforms without
# persistent storage (e.g. Hugging Face Spaces free tier): the RAG index
# over corpus/, and the seeded hero customers. Runs as `app` so the output
# files are owned correctly (matches the chown above). Re-running against
# an existing --db is idempotent (see scripts/seed_customers.py).
USER app
RUN python scripts/build_rag_index.py \
    && python scripts/seed_customers.py

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py"]
