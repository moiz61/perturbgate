from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen


URL = "http://127.0.0.1:8000/predict"

TOTAL_REQUESTS = 40
CLIENT_CONCURRENCY = 12

PAYLOAD = {
    "telemetry": {
        "voltage_jitter_mv": 40,
        "packet_retransmit_pct": 5,
        "inference_latency_ms": 100,
        "sensor_drift_sigma": 2,
        "cpu_temp_c": 60,
    }
}

BODY = json.dumps(PAYLOAD).encode("utf-8")


def send_request(index: int) -> dict:
    request = Request(
        URL,
        data=BODY,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started = time.perf_counter()

    with urlopen(request, timeout=30) as response:
        body = json.loads(
            response.read().decode("utf-8")
        )

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0

        return {
            "index": index,
            "status": response.status,
            "latency_ms": elapsed_ms,
            "request_id": response.headers.get(
                "X-Request-ID"
            ),
            "decision": body["decision"],
        }


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)

    position = int(
        round(
            (len(ordered) - 1) * p
        )
    )

    return ordered[position]


def main() -> None:
    started = time.perf_counter()

    results = []

    with ThreadPoolExecutor(
        max_workers=CLIENT_CONCURRENCY
    ) as executor:
        futures = [
            executor.submit(
                send_request,
                index,
            )
            for index in range(TOTAL_REQUESTS)
        ]

        for future in as_completed(futures):
            results.append(future.result())

    total_seconds = (
        time.perf_counter() - started
    )

    latencies = [
        item["latency_ms"]
        for item in results
    ]

    successes = sum(
        item["status"] == 200
        for item in results
    )

    unique_request_ids = {
        item["request_id"]
        for item in results
    }

    decisions = {
        decision: sum(
            item["decision"] == decision
            for item in results
        )
        for decision in {
            item["decision"]
            for item in results
        }
    }

    print("=== PERTURBGATE LOAD TEST ===")
    print(f"requests={TOTAL_REQUESTS}")
    print(f"client_concurrency={CLIENT_CONCURRENCY}")
    print(f"http_200={successes}")
    print(
        f"unique_request_ids="
        f"{len(unique_request_ids)}"
    )
    print(f"decisions={decisions}")
    print(
        f"mean_latency_ms="
        f"{statistics.mean(latencies):.3f}"
    )
    print(
        f"p50_latency_ms="
        f"{percentile(latencies, 0.50):.3f}"
    )
    print(
        f"p95_latency_ms="
        f"{percentile(latencies, 0.95):.3f}"
    )
    print(
        f"max_latency_ms="
        f"{max(latencies):.3f}"
    )
    print(
        f"throughput_requests_per_second="
        f"{TOTAL_REQUESTS / total_seconds:.2f}"
    )


if __name__ == "__main__":
    main()
