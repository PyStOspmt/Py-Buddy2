FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src/core:/app/src \
    PYBUDDY_LOCAL_ADMIN=0

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY web ./web
COPY data ./data
COPY scripts/bootstrap_demo_dbs.py ./scripts/bootstrap_demo_dbs.py

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser \
    && python scripts/bootstrap_demo_dbs.py \
    && chown -R appuser:appgroup /app/data /app/src /app/web

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=3)" || exit 1

CMD ["sh", "-c", "python scripts/bootstrap_demo_dbs.py && exec uvicorn server:app --app-dir src/core --host 0.0.0.0 --port 8000"]
