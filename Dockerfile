FROM python:3.12-slim

WORKDIR /app

# System deps for Playwright and nvidia-ml-py
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application code
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "alchemy.server:app", "--host", "0.0.0.0", "--port", "8000"]
