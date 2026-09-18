from __future__ import annotations

from prometheus_client import Counter, Histogram


HTTP_REQUESTS = Counter(
    "perturbgate_http_requests_total",
    "Total HTTP requests handled by PerturbGate.",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_DURATION = Histogram(
    "perturbgate_http_request_duration_seconds",
    "PerturbGate HTTP request duration in seconds.",
    ["method", "path"],
)

REQUEST_ERRORS = Counter(
    "perturbgate_request_errors_total",
    "Unhandled PerturbGate request errors.",
    ["method", "path"],
)

PREDICTIONS = Counter(
    "perturbgate_predictions_total",
    "PerturbGate prediction decisions.",
    ["decision", "label"],
)
