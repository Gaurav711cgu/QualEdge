# Stage 1: Build the React + Vite frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY ./frontend/package*.json ./
RUN npm ci
COPY ./frontend/ ./
RUN VITE_API_BASE_URL="" npm run build

# Stage 2: Final Python application container
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

# Copy the compiled React assets from Stage 1 into the frontend/dist folder
COPY --from=frontend-builder /app/frontend/dist /code/frontend/dist

# Expose port (FastAPI default, matching Hugging Face Spaces default port 7860)
EXPOSE 7860

# Run FastAPI backend using Uvicorn
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
