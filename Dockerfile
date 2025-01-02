FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.7.1

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml poetry.lock* ./
COPY README.md ./
COPY scripts/ ./scripts/

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-root

# Set the entrypoint
ENTRYPOINT ["poetry", "run", "arxiv-to-obsidian"]