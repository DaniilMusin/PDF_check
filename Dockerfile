# ---------- builder ----------
FROM python:3.10-slim AS builder
WORKDIR /app
COPY pyproject.toml poetry.lock* ./
RUN pip install poetry && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-dev
COPY src/ src/
COPY artifacts/ artifacts/
# ---------- runtime ----------
FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY src/ src/
COPY artifacts/ artifacts/
ENV PYTHONPATH=/app
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "src.app.web:app"]
