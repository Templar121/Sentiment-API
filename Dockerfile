# Use official Python base image
FROM python:3.10-slim

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    # Download required NLTK data (so you don't depend on dynamic downloading at runtime)
    && python -m nltk.downloader punkt stopwords \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy app files
COPY . .

# Expose port
EXPOSE 5000

# Run app with Gunicorn
CMD ["gunicorn", "app:app", "--workers", "4", "--bind", "0.0.0.0:5000", "--timeout", "30"]
