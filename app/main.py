from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool

from app.gating import evaluate_stability
from app.logging_config import configure_logging
from app.metrics import (
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS,
    PREDICTIONS,
    REQUEST_ERRORS,
)
from app.model import LoadedModel, load_model_bundle, predict_probabilities
from app.perturb import generate_local_perturbations, telemetry_to_array
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
    Stability,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"
logger = configure_logging()


def get_model_path() -> Path:
    configured_path = os.getenv("PERTURBGATE_MODEL_PATH")

    if configured_path:
        return Path(configured_path).expanduser().resolve()

    return DEFAULT_MODEL_PATH


def get_max_concurrent_inference() -> int:
    value = int(
        os.getenv(
            "PERTURBGATE_MAX_CONCURRENT_INFERENCE",
            "4",
        )
    )

    if value < 1:
        raise ValueError(
            "PERTURBGATE_MAX_CONCURRENT_INFERENCE must be >= 1."
        )

    return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = get_model_path()

    # Load the trained model exactly once when FastAPI starts.
    loaded_model = load_model_bundle(model_path)

    # Bound simultaneous model inference work.
    inference_semaphore = asyncio.Semaphore(
        get_max_concurrent_inference()
    )

    app.state.loaded_model = loaded_model
    app.state.model_version = loaded_model.model_sha256[:12]
    app.state.inference_semaphore = inference_semaphore
    app.state.ready = True

    try:
        yield
    finally:
        app.state.ready = False


app = FastAPI(
    title="PerturbGate",
    version="0.3.0",
    description=(
        "A stability-gated FastAPI inference service "
        "with local perturbation testing."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = uuid.uuid4().hex
    request.state.request_id = request_id

    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_seconds = time.perf_counter() - started

        route = request.scope.get("route")
        route_path = getattr(
            route,
            "path",
            request.url.path,
        )

        REQUEST_ERRORS.labels(
            method=request.method,
            path=route_path,
        ).inc()

        logger.exception(
            "request_failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": route_path,
                "duration_ms": round(
                    duration_seconds * 1000.0,
                    3,
                ),
            },
        )
        raise

    duration_seconds = time.perf_counter() - started

    route = request.scope.get("route")
    route_path = getattr(
        route,
        "path",
        request.url.path,
    )

    HTTP_REQUESTS.labels(
        method=request.method,
        path=route_path,
        status_code=str(response.status_code),
    ).inc()

    HTTP_REQUEST_DURATION.labels(
        method=request.method,
        path=route_path,
    ).observe(duration_seconds)

    duration_ms = duration_seconds * 1000.0

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = (
        f"{duration_ms:.3f}"
    )

    logger.info(
        "request_complete",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": route_path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 3),
        },
    )

    return response


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(
        content=generate_latest(),
        headers={
            "Content-Type": CONTENT_TYPE_LATEST,
        },
    )


def predict_one_sync(
    loaded_model: LoadedModel,
    model_version: str,
    prediction_request: PredictionRequest,
) -> PredictionResponse:
    baseline = telemetry_to_array(
        prediction_request.telemetry
    )

    perturbations = generate_local_perturbations(
        baseline
    )

    # One matrix call:
    # row 0 = original input
    # rows 1-10 = the 10 local perturbations
    feature_matrix = np.vstack(
        [
            baseline,
            perturbations,
        ]
    )

    probabilities = predict_probabilities(
        loaded_model,
        feature_matrix,
    )

    baseline_probability = float(
        probabilities[0]
    )

    perturbed_probabilities = probabilities[1:]

    gate = evaluate_stability(
        baseline_probability=baseline_probability,
        perturbed_probabilities=perturbed_probabilities,
        stability_threshold=(
            prediction_request.stability_threshold
        ),
        max_probability_shift=(
            prediction_request.max_probability_shift
        ),
    )

    return PredictionResponse(
        label=gate.label,
        probability_unstable=baseline_probability,
        decision=gate.decision,
        stability=Stability(
            agreement=gate.agreement,
            max_probability_shift=(
                gate.max_probability_shift
            ),
            perturbations_evaluated=(
                gate.perturbations_evaluated
            ),
            fragile=gate.fragile,
        ),
        model_version=model_version,
    )


async def predict_one(
    request: Request,
    prediction_request: PredictionRequest,
) -> PredictionResponse:
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(
            status_code=503,
            detail="Model service is not ready.",
        )

    loaded_model = request.app.state.loaded_model
    model_version = request.app.state.model_version
    semaphore = request.app.state.inference_semaphore

    # Only a bounded number of requests may enter
    # model inference at the same time.
    async with semaphore:
        result = await run_in_threadpool(
            predict_one_sync,
            loaded_model,
            model_version,
            prediction_request,
        )

    PREDICTIONS.labels(
        decision=result.decision,
        label=result.label,
    ).inc()

    return result


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {
        "status": "ok",
    }


@app.get("/readyz")
async def readyz(request: Request) -> dict[str, str]:
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(
            status_code=503,
            detail="Model service is not ready.",
        )

    return {
        "status": "ready",
        "model_version": request.app.state.model_version,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    return await predict_one(
        request,
        payload,
    )


@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
)
async def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
) -> BatchPredictionResponse:
    predictions = await asyncio.gather(
        *[
            predict_one(request, item)
            for item in payload.items
        ]
    )

    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions),
        model_version=request.app.state.model_version,
    )
