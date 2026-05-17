from __future__ import annotations

import argparse
import json
import time
import urllib.request


def post_json(url: str, api_key: str, payload: dict) -> tuple[int, float]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
            "X-Request-ID": "load-probe",
        },
    )
    started_at = time.perf_counter()
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()
        return response.status, (time.perf_counter() - started_at) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description="Small BERTScope API load probe.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8788")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--requests", type=int, default=10)
    args = parser.parse_args()

    latencies = []
    for _ in range(args.requests):
        status, latency_ms = post_json(
            f"{args.base_url}/api/v1/financial-analysis",
            args.api_key,
            {"text": "Revenue growth was strong, but inflation risk remains."},
        )
        if status != 200:
            raise SystemExit(f"Unexpected status: {status}")
        latencies.append(latency_ms)

    average = sum(latencies) / len(latencies)
    print(
        json.dumps(
            {
                "requests": len(latencies),
                "average_latency_ms": round(average, 3),
                "max_latency_ms": round(max(latencies), 3),
            }
        )
    )


if __name__ == "__main__":
    main()
