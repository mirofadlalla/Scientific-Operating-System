# Use an official lightweight Python base image
FROM python:3.10-slim

# Set system environment variables to optimize Python inside Docker
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

# Set the working directory inside the container
WORKDIR /code

# Install system-level dependencies if required by httpx or asyncio packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to utilize Docker layer caching
COPY requirements.txt .

# Install Python packages defined in requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the entire application source code into the container workdir
COPY . .

# Hugging Face Spaces runs on port 7860 by default, expose it
EXPOSE 7860

# Run the FastAPI core app using Uvicorn on host 0.0.0.0 and port 7860
# Using main:app to execute the root-level main.py entry point
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]