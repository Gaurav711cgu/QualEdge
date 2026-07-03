# Use a slim, stable Python base image
FROM python:3.11-slim

# Set system-level environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/code

# Set working directory inside the container
WORKDIR /code

# Install system dependencies (build-essential for scikit-learn compilation if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the entire workspace into the container
COPY . /code

# Expose port (FastAPI default, matching Hugging Face Spaces default port 7860)
EXPOSE 7860

# Run FastAPI backend using Uvicorn
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
