from fastapi.testclient import TestClient

from app.main import app


def test_metrics_endpoint_exposes_http_metrics():
    with TestClient(app) as client:
        health_response = client.get("/healthz")

        assert health_response.status_code == 200

        response = client.get("/metrics")

        assert response.status_code == 200
        assert response.headers[
            "content-type"
        ].startswith("text/plain")

        body = response.text

        assert (
            "perturbgate_http_requests_total"
            in body
        )

        assert (
            "perturbgate_http_request_duration_seconds"
            in body
        )


def test_prediction_is_recorded_in_metrics():
    with TestClient(app) as client:
        prediction = client.post(
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

        assert prediction.status_code == 200
        assert prediction.json()["decision"] == "accept"

        response = client.get("/metrics")

        assert response.status_code == 200

        assert (
            'perturbgate_predictions_total'
            '{decision="accept",label="healthy"}'
            in response.text
        )
