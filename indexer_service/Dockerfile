# indexer_service/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies needed by some Python packages 
# and to ensure the environment is suitable for building Python packages.
# This includes poppler-utils for PDF processing.
## Note: The `libpoppler-cpp-dev` package is required for the `pdfminer.six` library,
## which is used for PDF text extraction in the indexer service.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpoppler-cpp-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
# Use --no-cache-dir to avoid caching the installed packages, reducing image size
# and ensuring that the image is built with the latest versions of the packages.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
