# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose API port (e.g., Flask defaults to 5000)
EXPOSE 10000

# Run the app
CMD ["python", "app.py"]
