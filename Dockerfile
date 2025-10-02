# Use lightweight python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Gunicorn দিয়ে run (Render needs $PORT)
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:$PORT"]
