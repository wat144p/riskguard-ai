FROM python:3.11-slim

WORKDIR /app

# Dependencies first, so this layer is cached between code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY ml ./ml
COPY flows ./flows
COPY models/release ./models/release

# Run as a non-root user; data/ is the price cache the API writes to
RUN useradd --create-home appuser && mkdir -p /app/data && chown -R appuser /app
USER appuser

# ALPHAVANTAGE_API_KEY is optional and passed at runtime; without it the API uses yfinance.
# Cloud Run sets PORT; locally it defaults to 8000.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
