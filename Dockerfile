FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Baked at build time so the deployed image is self-contained (no persistent
# storage needed): RAG index over corpus/, and the 5 seeded hero customers.
RUN python scripts/build_rag_index.py \
    && python scripts/seed_customers.py

# Hugging Face Spaces (Docker SDK) routes traffic to this port by default.
ENV PORT=7860
EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

ENTRYPOINT ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT} --server.address=0.0.0.0"]
