"""Measures /predict_risk latency against a running server.

Usage:
    uvicorn app.main:app --port 8000        # in one terminal
    python scripts/benchmark_latency.py     # in another (optionally: --url https://... --n 200)

The first request per ticker downloads prices and is reported separately as "cold".
Every later request reads the 24-hour CSV cache, rebuilds the features and predicts.
"""
import argparse
import statistics
import time

import httpx

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=200, help="warm requests to time")
    args = parser.parse_args()

    with httpx.Client(base_url=args.url, timeout=120) as client:
        cold = []
        for ticker in TICKERS:
            start = time.perf_counter()
            client.post("/predict_risk", json={"ticker": ticker}).raise_for_status()
            cold.append((time.perf_counter() - start) * 1000)

        warm = []
        for i in range(args.n):
            start = time.perf_counter()
            client.post("/predict_risk", json={"ticker": TICKERS[i % len(TICKERS)]}).raise_for_status()
            warm.append((time.perf_counter() - start) * 1000)

    warm.sort()
    print(f"target: {args.url}")
    print(f"cold (first request per ticker, n={len(cold)}): median {statistics.median(cold):.0f} ms, max {max(cold):.0f} ms")
    print(
        f"warm (n={len(warm)}): median {statistics.median(warm):.1f} ms, "
        f"p95 {warm[int(0.95 * len(warm)) - 1]:.1f} ms, max {warm[-1]:.1f} ms"
    )


if __name__ == "__main__":
    main()
