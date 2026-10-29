# Container for Hugging Face Spaces (Docker SDK), Render, Railway, or any VPS.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf_cache \
    PORT=7860

WORKDIR /srv
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /data/hf_cache && chmod -R 777 /data

EXPOSE 7860
CMD ["sh", "-c", "uvicorn main:app --app-dir app --host 0.0.0.0 --port ${PORT}"]
