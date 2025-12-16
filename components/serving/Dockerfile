FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY components/serving /app/components/serving
ENV PYTHONPATH=/app

EXPOSE 8080
CMD ["uvicorn", "components.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
