FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300

WORKDIR /app

RUN addgroup --system perturbgate \
    && adduser --system \
       --ingroup perturbgate \
       perturbgate

COPY requirements.lock .

RUN python -m pip install \
    --no-cache-dir \
    --retries 30 \
    -r requirements.lock

COPY app ./app
COPY artifacts ./artifacts

RUN chown -R perturbgate:perturbgate /app

USER perturbgate

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=10s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
