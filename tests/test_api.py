import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_healthz(client):
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }


def test_readyz(client):
    response = client.get("/readyz")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ready"
    assert len(body["model_version"]) == 12
    assert len(body["artifact_sha256"]) == 64


def test_stable_input_is_accepted(client):
    response = client.post(
        "/predict",
        json={
            "telemetry": {
                "voltage_jitter_mv": 40,
                "packet_retransmit_pct": 5,
                "inference_latency_ms": 100,
                "sensor_drift_sigma": 2,
                "cpu_temp_c": 60,
            }
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["label"] == "healthy"
    assert body["decision"] == "accept"
    assert body["stability"]["fragile"] is False
    assert (
        body["stability"]["perturbations_evaluated"]
        == 10
    )


def test_fragile_input_is_abstained(client):
    response = client.post(
        "/predict",
        json={
            "telemetry": {
                "voltage_jitter_mv": 41.93188438245095,
                "packet_retransmit_pct": 7.8001614283472795,
                "inference_latency_ms": 403.65791719276825,
                "sensor_drift_sigma": 3.380880401569116,
                "cpu_temp_c": 102.42304400705142,
            }
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["decision"] == "abstain"
    assert body["stability"]["fragile"] is True
    assert body["stability"]["agreement"] < 0.90


def test_invalid_telemetry_is_rejected(client):
    response = client.post(
        "/predict",
        json={
            "telemetry": {
                "voltage_jitter_mv": -1,
                "packet_retransmit_pct": 5,
                "inference_latency_ms": 100,
                "sensor_drift_sigma": 2,
                "cpu_temp_c": 60,
            }
        },
    )

    assert response.status_code == 422


def test_unknown_fields_are_rejected(client):
    response = client.post(
        "/predict",
        json={
            "telemetry": {
                "voltage_jitter_mv": 40,
                "packet_retransmit_pct": 5,
                "inference_latency_ms": 100,
                "sensor_drift_sigma": 2,
                "cpu_temp_c": 60,
                "made_up_sensor": 123,
            }
        },
    )

    assert response.status_code == 422


def test_batch_prediction(client):
    response = client.post(
        "/predict-batch",
        json={
            "items": [
                {
                    "telemetry": {
                        "voltage_jitter_mv": 40,
                        "packet_retransmit_pct": 5,
                        "inference_latency_ms": 100,
                        "sensor_drift_sigma": 2,
                        "cpu_temp_c": 60,
                    }
                },
                {
                    "telemetry": {
                        "voltage_jitter_mv": 41.93188438245095,
                        "packet_retransmit_pct": 7.8001614283472795,
                        "inference_latency_ms": 403.65791719276825,
                        "sensor_drift_sigma": 3.380880401569116,
                        "cpu_temp_c": 102.42304400705142,
                    }
                },
            ]
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["count"] == 2
    assert len(body["predictions"]) == 2

    assert (
        body["predictions"][0]["decision"]
        == "accept"
    )

    assert (
        body["predictions"][1]["decision"]
        == "abstain"
    )


def test_response_contains_request_id(client):
    response = client.get("/healthz")

    assert response.status_code == 200

    request_id = response.headers.get("X-Request-ID")

    assert request_id is not None
    assert len(request_id) == 32


def test_response_contains_process_time(client):
    response = client.get("/healthz")

    assert response.status_code == 200

    process_time = response.headers.get(
        "X-Process-Time-Ms"
    )

    assert process_time is not None
    assert float(process_time) >= 0.0


def test_request_ids_are_unique(client):
    first = client.get("/healthz")
    second = client.get("/healthz")

    assert (
        first.headers["X-Request-ID"]
        != second.headers["X-Request-ID"]
    )
