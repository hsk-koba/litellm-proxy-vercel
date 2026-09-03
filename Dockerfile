FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LITELLM_PROXY_URL=http://127.0.0.1:4000

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY litellm_config.yml docker-entrypoint.sh ./
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
