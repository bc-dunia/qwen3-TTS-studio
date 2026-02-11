# Multi-stage build: builder stage
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Install build dependencies
RUN set -eux; \
    sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources; \
    sed -i '/^Suites:/ s/[[:space:]][^[:space:]]*-updates//g' /etc/apt/sources.list.d/debian.sources; \
    success=0; \
    max_attempts=8; \
    i=1; \
    while [ "${i}" -le "${max_attempts}" ]; do \
        if apt-get -o Acquire::Retries=3 update && \
            apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
                build-essential; then \
            success=1; \
            break; \
        fi; \
        echo "apt-get failed (attempt ${i}/${max_attempts}), retrying..." >&2; \
        rm -rf /var/lib/apt/lists/*; \
        sleep 5; \
        i=$((i + 1)); \
    done; \
    test "${success}" = 1; \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt /build/requirements.txt

# Install Python dependencies to a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Final stage: runtime
FROM python:3.12-slim

# OCI Image Labels for GHCR metadata
LABEL org.opencontainers.image.title="Qwen3-TTS Studio" \
      org.opencontainers.image.description="Professional-grade interface for Qwen3-TTS with fine-grained control and intuitive workflows" \
      org.opencontainers.image.version="0.1.5" \
      org.opencontainers.image.source="https://github.com/bc-dunia/qwen3-tts-studio" \
      org.opencontainers.image.authors="bc-dunia" \
      org.opencontainers.image.vendor="bc-dunia" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_HOME=/home/appuser/.cache/huggingface \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install runtime dependencies only (no build tools)
RUN set -eux; \
    sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources; \
    sed -i '/^Suites:/ s/[[:space:]][^[:space:]]*-updates//g' /etc/apt/sources.list.d/debian.sources; \
    success=0; \
    max_attempts=8; \
    i=1; \
    while [ "${i}" -le "${max_attempts}" ]; do \
        if apt-get -o Acquire::Retries=3 update && \
            apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
                libsndfile1 \
                libgomp1 \
                sox \
                libsox-fmt-all \
                ffmpeg \
                curl; then \
            success=1; \
            break; \
        fi; \
        echo "apt-get failed (attempt ${i}/${max_attempts}), retrying..." >&2; \
        rm -rf /var/lib/apt/lists/*; \
        sleep 5; \
        i=$((i + 1)); \
    done; \
    test "${success}" = 1; \
    rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -m -r -g appuser appuser && \
    mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser && \
    chown appuser:appuser /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=appuser:appuser . /app

USER appuser

EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "qwen_tts_ui.py"]
