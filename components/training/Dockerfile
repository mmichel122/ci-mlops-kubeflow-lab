FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script
COPY train.py .

# Output directory for artifacts
RUN mkdir -p model_artifacts

# Entrypoint
ENTRYPOINT ["python", "train.py"]