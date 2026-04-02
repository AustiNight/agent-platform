FROM python:3.11-slim

# Minimal system deps before Playwright's --with-deps handles the rest
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source and install — separate layer so pip cache survives code-only changes
COPY pyproject.toml .
COPY src/ ./src/

RUN pip install --no-cache-dir .

# Install Playwright + all Chromium system deps in one pass
RUN playwright install --with-deps chromium

# Remaining files: alembic, config, examples, etc.
COPY . .

EXPOSE 8000

# Default: gateway. Worker service overrides this in Railway dashboard.
CMD ["uvicorn", "src.gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
