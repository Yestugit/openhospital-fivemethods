FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY hospital_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY hospital_service/ ./

EXPOSE 7860

CMD ["uvicorn", "service.asgi:app", "--host", "0.0.0.0", "--port", "7860"]
