FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    BACKEND_PORT=6008 \
    PUBLIC_DEPLOYMENT=true \
    SHELL_TOOL_ENABLED=false \
    READ_FILE_TOOL_ENABLED=true \
    PUBLIC_RATE_LIMIT_PER_MINUTE=10 \
    PUBLIC_MAX_PROMPT_CHARS=8000 \
    LANGFUSE_TRACING_ENABLED=false \
    MYSQL_ENABLED=false \
    REDIS_ENABLED=false \
    RABBITMQ_ENABLED=false

WORKDIR /app

COPY pyproject.toml README.md ./
COPY . .

RUN python -m pip install --upgrade pip \
    && python -m pip install . \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

CMD ["python", "main.py"]
