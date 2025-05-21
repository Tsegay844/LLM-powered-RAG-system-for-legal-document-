# indexer_service/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies needed by some Python packages (like pypdf often needs libpoppler-cpp-dev)
# This helps avoid common installation issues
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpoppler-cpp-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# No CMD here, this service is designed to be run explicitly with `docker compose run`