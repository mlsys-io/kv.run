# MLOC Application Docker image
FROM mloc:base

USER root

# Install additional runtime dependencies
RUN apt-get update && apt-get install -y \
    redis-tools \
    postgresql-client \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Switch back to non-root user
USER mloc

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=mloc:mloc src/ ./src/
COPY --chown=mloc:mloc pyproject.toml ./
COPY --chown=mloc:mloc README.md ./

# Install the application in development mode
RUN uv pip install -e .

# Create directories for runtime
RUN mkdir -p /tmp/mloc_work && \
    mkdir -p /app/logs && \
    mkdir -p /app/artifacts

# Environment variables
ENV MLOC_LOG_LEVEL=INFO
ENV MLOC_REDIS_URL=redis://redis:6379
ENV MLOC_NODE_TYPE=WORKER
ENV MLOC_HOST=0.0.0.0
ENV MLOC_PORT=8000

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
  CMD python -c "from mloc.main import app; print('MLOC app healthy')" || exit 1

# Default command
CMD ["mloc", "start"]