# PerturbGate

[![PerturbGate CI](https://github.com/moiz61/perturbgate/actions/workflows/ci.yml/badge.svg)](https://github.com/moiz61/perturbgate/actions/workflows/ci.yml)

**PerturbGate** is a production-oriented machine learning inference service built with **FastAPI**, **Scikit-Learn**, **Docker**, **GitHub Actions**, and **Prometheus**.

Instead of returning a model prediction without qualification, PerturbGate checks whether that prediction remains stable under small, deterministic changes to the input.

A prediction is returned with one of two decisions:

- **`accept`** — the prediction remains sufficiently stable around the supplied input.
- **`abstain`** — small input perturbations make the prediction too fragile to accept confidently.

The repository demonstrates an end-to-end ML serving workflow including model training, API validation, local robustness testing, bounded inference concurrency, automated regression testing, structured logging, metrics, Docker deployment, CI/CD, model provenance, and container releases.

---

## How PerturbGate Works

For each incoming telemetry sample, the service first obtains the baseline probability from a Scikit-Learn `HistGradientBoostingClassifier`.

It then generates **10 deterministic local perturbations** by moving each of the five input features slightly downward and upward:

| Feature | Local perturbation |
|---|---:|
| `voltage_jitter_mv` | ±3 mV |
| `packet_retransmit_pct` | ±0.6 |
| `inference_latency_ms` | ±10 ms |
| `sensor_drift_sigma` | ±0.25 |
| `cpu_temp_c` | ±1.5 °C |

The baseline and the 10 perturbed samples are evaluated by the model.

Two stability measurements are calculated:

1. **Class agreement** — the fraction of perturbed samples retaining the baseline class.
2. **Maximum probability shift** — the largest change in predicted probability relative to the baseline.

The default gate accepts a prediction when:

```text
agreement >= 0.90
AND
maximum probability shift <= 0.18
```

Otherwise, PerturbGate returns:

```text
abstain
```

Conceptually:

```text
Telemetry
   │
   ▼
Input validation
   │
   ▼
Baseline prediction
   │
   ├──────────────┐
   ▼              ▼
10 nearby     Baseline
perturbations probability
   │              │
   └──────┬───────┘
          ▼
   Stability analysis
          │
     ┌────┴────┐
     ▼         ▼
   accept    abstain
```

---

## Main Features

- FastAPI REST inference service
- Scikit-Learn `HistGradientBoostingClassifier`
- Strict Pydantic input validation
- Deterministic local perturbation testing
- Selective prediction with `accept` / `abstain`
- Single and batch prediction endpoints
- `asyncio.Semaphore`-based bounded inference concurrency
- Thread-pool execution for CPU-bound Scikit-Learn inference
- Health and readiness endpoints
- Structured JSON request logging
- Per-request tracing IDs
- Request latency headers
- Prometheus-compatible metrics
- Docker health checks
- Automated pytest regression suite
- GitHub Actions CI
- GitHub Container Registry releases
- Deterministic semantic model identity
- SHA-256 artifact fingerprinting

---

# Quick Start

There are three main ways to run PerturbGate:

1. Run the published Docker image.
2. Clone the repository and run it with Python.
3. Clone the repository and build the Docker image locally.

The published Docker image is the quickest option.

---

# Option 1 — Run the Published Docker Image

If Docker is already installed, no Python environment or local model training is required.

Pull the verified `0.3.0` release:

```bash
docker pull ghcr.io/moiz61/perturbgate:0.3.0
```

Or pull the latest published image:

```bash
docker pull ghcr.io/moiz61/perturbgate:latest
```

Run PerturbGate:

```bash
docker run -d \
  --name perturbgate-api \
  --restart unless-stopped \
  -p 8000:8000 \
  ghcr.io/moiz61/perturbgate:0.3.0
```

Check the container:

```bash
docker ps --filter name=perturbgate-api
```

Check API readiness:

```bash
curl http://127.0.0.1:8000/readyz
```

The API is available at:

```text
http://127.0.0.1:8000
```

Interactive FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

OpenAPI specification:

```text
http://127.0.0.1:8000/openapi.json
```

Prometheus metrics:

```text
http://127.0.0.1:8000/metrics
```

To stop the service:

```bash
docker stop perturbgate-api
```

To start it again:

```bash
docker start perturbgate-api
```

To remove it:

```bash
docker rm -f perturbgate-api
```

---

# Option 2 — Clone and Run from Source

## Requirements

- Python 3.14
- Git
- `venv`
- pip

Docker is optional when running directly with Python.

## 1. Clone the repository

```bash
git clone https://github.com/moiz61/perturbgate.git
cd perturbgate
```

## Alternative: Download Without Git

1. Open `https://github.com/moiz61/perturbgate`
2. Select **Code**
3. Select **Download ZIP**
4. Extract the ZIP archive
5. Open a terminal inside the extracted `perturbgate` directory

## 2. Create a Python Virtual Environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.lock
```

`requirements.lock` is recommended when reproducing the tested environment.

## 4. Train the Model

The generated model artifact is intentionally not stored in Git.

```bash
python train.py
```

This creates:

```text
artifacts/model.joblib
```

## 5. Start the API

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The API is now available at:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Option 3 — Build and Run Docker Locally

After cloning the repository and generating the model with `python train.py`, build the image:

```bash
docker build -t perturbgate:local .
```

Run it:

```bash
docker run -d \
  --name perturbgate-api \
  --restart unless-stopped \
  -p 8000:8000 \
  perturbgate:local
```

Verify:

```bash
curl http://127.0.0.1:8000/readyz
```

---

# API Usage

## Health Check

```bash
curl http://127.0.0.1:8000/healthz
```

Example:

```json
{
  "status": "ok"
}
```

## Readiness Check

```bash
curl http://127.0.0.1:8000/readyz
```

The readiness endpoint indicates whether the model service has initialized successfully. Current source versions also expose model provenance information, including the semantic model version and exact artifact SHA-256 fingerprint.

## Single Prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "telemetry": {
      "voltage_jitter_mv": 40,
      "packet_retransmit_pct": 5,
      "inference_latency_ms": 100,
      "sensor_drift_sigma": 2,
      "cpu_temp_c": 60
    }
  }'
```

Example accepted response:

```json
{
  "label": "healthy",
  "probability_unstable": 0.029676799765223305,
  "decision": "accept",
  "stability": {
    "agreement": 1.0,
    "max_probability_shift": 0.016093771599878542,
    "perturbations_evaluated": 10,
    "fragile": false
  },
  "model_version": "..."
}
```

## Example Fragile Prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "telemetry": {
      "voltage_jitter_mv": 41.93188438245095,
      "packet_retransmit_pct": 7.8001614283472795,
      "inference_latency_ms": 403.65791719276825,
      "sensor_drift_sigma": 3.380880401569116,
      "cpu_temp_c": 102.42304400705142
    }
  }'
```

This point lies close to the model decision boundary and has demonstrated an `abstain` result because nearby perturbations create substantial prediction instability.

---

# Custom Stability Thresholds

Defaults:

```text
stability_threshold = 0.90
max_probability_shift = 0.18
```

Increasing the agreement threshold or reducing the permitted probability shift makes the gate more conservative.

---

# Input Ranges

| Field | Minimum | Maximum |
|---|---:|---:|
| `voltage_jitter_mv` | 0 | 120 |
| `packet_retransmit_pct` | 0 | 30 |
| `inference_latency_ms` | 5 | 500 |
| `sensor_drift_sigma` | 0 | 8 |
| `cpu_temp_c` | 25 | 105 |

Unknown fields are rejected instead of being silently ignored.

---

# Batch Prediction

Endpoint:

```text
POST /predict-batch
```

Batch size is currently limited to 64 items.

---

# Request Tracing

Each HTTP response includes:

```text
X-Request-ID
X-Process-Time-Ms
```

The request ID also appears in the structured application log, allowing a client request to be correlated with its server-side log record.

---

# Structured Logging

Docker logs:

```bash
docker logs perturbgate-api
```

Follow continuously:

```bash
docker logs -f perturbgate-api
```

---

# Prometheus Metrics

Metrics are exposed at:

```text
GET /metrics
```

PerturbGate-specific metrics include:

```text
perturbgate_http_requests_total
perturbgate_http_request_duration_seconds
perturbgate_request_errors_total
perturbgate_predictions_total
```

---

# Concurrency Control

Scikit-Learn inference is CPU-bound. PerturbGate executes inference through a thread pool rather than directly on the FastAPI event loop.

An application-level:

```python
asyncio.Semaphore(4)
```

limits simultaneous inference operations.

The default can be changed using:

```bash
export PERTURBGATE_MAX_CONCURRENT_INFERENCE=4
```

---

# Model Path Configuration

By default:

```text
artifacts/model.joblib
```

A different artifact can be selected with:

```bash
export PERTURBGATE_MODEL_PATH=/path/to/model.joblib
```

---

# Model Provenance

PerturbGate tracks two related forms of model identity.

## Semantic Model Version

The semantic model fingerprint represents the logical trained model using deterministic training configuration and model outputs.

## Artifact SHA-256

The exact serialized `model.joblib` file is also hashed using SHA-256, identifying the precise binary artifact being served.

This separates:

```text
same logical model
```

from:

```text
same exact serialized artifact
```

---

# Running the Tests

```bash
python train.py
pytest -q
```

The regression suite covers API behavior, validation, accepted and abstained predictions, perturbation generation, stability gating, concurrency limits, request tracing, Prometheus metrics, and model provenance.

---

# Finding a Fragile Input

```bash
python -m scripts.find_abstain
```

---

# Load Testing

```bash
python -m scripts.load_test
```

One local development test produced:

```text
40 total requests
12 concurrent clients
40 HTTP 200 responses
40 unique request IDs
approximately 94 requests/second
```

These values are hardware- and environment-dependent and are not universal performance benchmarks.

---

# Continuous Integration

The repository contains:

```text
.github/workflows/ci.yml
```

Pushes and pull requests to `main` perform model training, compilation, tests, and a Docker build.

---

# Releases

Version tags matching `v*.*.*` trigger `.github/workflows/release.yml` and publish container images to:

```text
ghcr.io/moiz61/perturbgate
```

---

# Deploying on a Server

```bash
docker pull ghcr.io/moiz61/perturbgate:latest

docker run -d \
  --name perturbgate-api \
  --restart unless-stopped \
  -p 8000:8000 \
  ghcr.io/moiz61/perturbgate:latest
```

For a genuinely public deployment, the service should normally be placed behind a reverse proxy or managed platform providing **HTTPS/TLS**, domain routing, access controls, and appropriate firewall rules.

---

# Repository Structure

```text
perturbgate/
├── .github/workflows/
│   ├── ci.yml
│   └── release.yml
├── app/
│   ├── gating.py
│   ├── logging_config.py
│   ├── main.py
│   ├── metrics.py
│   ├── model.py
│   ├── perturb.py
│   └── schemas.py
├── artifacts/
│   └── .gitkeep
├── scripts/
│   ├── find_abstain.py
│   └── load_test.py
├── tests/
├── Dockerfile
├── README.md
├── requirements.lock
├── requirements.txt
└── train.py
```

---

# Project Scope and Limitations

The current classifier is trained on a **synthetic telemetry dataset** generated by `train.py`.

PerturbGate is primarily a demonstration of robustness-aware ML inference, selective prediction and abstention, API engineering, testing, observability, reproducibility, containerization, and CI/CD.

The current model and telemetry ranges should **not** be interpreted as a validated real-world failure-prediction system.

For real operational use, the model, features, perturbation sizes, decision thresholds, and evaluation procedure would need to be validated for the intended application domain.

---

# Why Abstention?

A conventional classifier normally returns its prediction even when the input lies close to a decision boundary.

PerturbGate adds another question:

> **Does the prediction remain stable if the input changes slightly?**

If not, the service can abstain instead of presenting the prediction as equally reliable.

---

# Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| API | FastAPI, Uvicorn |
| Validation | Pydantic |
| ML | Scikit-Learn, NumPy |
| Model persistence | Joblib |
| Concurrency | asyncio, semaphore, thread pool |
| Testing | pytest |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Registry | GitHub Container Registry |
| Metrics | Prometheus client |
| Logging | Structured JSON logging |
| Version control | Git / GitHub |

---

# License

No license has currently been specified for this repository.

Until a license is added, normal copyright rules apply and the source should not be assumed to grant unrestricted reuse, modification, or redistribution rights.

---

# Author

**Abdul Moiz Zagham**

GitHub: [@moiz61](https://github.com/moiz61)

---

# Project Status

PerturbGate is an actively developed demonstration project focused on robust and reproducible machine-learning inference infrastructure.

The verified published container release is currently `0.3.0`, while the `main` branch may contain newer development work before the next tagged release.
