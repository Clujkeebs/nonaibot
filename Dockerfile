# Deterministic build — avoids nixpacks' nix-cache fetches that were failing on
# the network. python:3.11-slim is a widely-cached image with manylinux wheels
# available for numpy/pandas/alpaca-py, so nothing compiles from source.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so this layer caches across code changes.
# --prefer-binary: wheels only for the heavy packages (no source compile).
# --retries/--timeout: ride out transient PyPI blips.
COPY requirements.txt .
RUN pip install --prefer-binary --retries 8 --timeout 180 -r requirements.txt

# Application code
COPY . .

# Health/status server port
EXPOSE 8080

CMD ["python", "main.py"]
