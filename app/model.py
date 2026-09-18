from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


EXPECTED_FEATURE_NAMES = (
    "voltage_jitter_mv",
    "packet_retransmit_pct",
    "inference_latency_ms",
    "sensor_drift_sigma",
    "cpu_temp_c",
)


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    feature_names: tuple[str, ...]
    model_version: str
    artifact_sha256: str
    metrics: dict[str, float]


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_model_bundle(path: Path) -> LoadedModel:
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}"
        )

    artifact_sha256 = calculate_sha256(path)
    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        raise ValueError(
            "Model artifact must contain a dictionary bundle."
        )

    if "model" not in bundle:
        raise ValueError(
            "Model bundle is missing 'model'."
        )

    if "feature_names" not in bundle:
        raise ValueError(
            "Model bundle is missing 'feature_names'."
        )

    feature_names = tuple(
        bundle["feature_names"]
    )

    if feature_names != EXPECTED_FEATURE_NAMES:
        raise ValueError(
            "Model feature order does not match "
            "the API feature order.\n"
            f"Expected: {EXPECTED_FEATURE_NAMES}\n"
            f"Found:    {feature_names}"
        )

    model = bundle["model"]

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Loaded model does not provide predict_proba()."
        )

    metrics = {
        key: float(value)
        for key, value
        in bundle.get("metrics", {}).items()
    }

    # New artifacts contain a deterministic semantic version.
    # Old artifacts fall back to their file SHA for compatibility.
    model_version = str(
        bundle.get(
            "model_version",
            artifact_sha256,
        )
    )

    return LoadedModel(
        model=model,
        feature_names=feature_names,
        model_version=model_version,
        artifact_sha256=artifact_sha256,
        metrics=metrics,
    )


def predict_probabilities(
    loaded_model: LoadedModel,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(
        feature_matrix,
        dtype=np.float64,
    )

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    if matrix.ndim != 2:
        raise ValueError(
            "Feature matrix must be two-dimensional."
        )

    if matrix.shape[1] != len(
        EXPECTED_FEATURE_NAMES
    ):
        raise ValueError(
            f"Expected {len(EXPECTED_FEATURE_NAMES)} "
            f"features, received {matrix.shape[1]}."
        )

    probabilities = (
        loaded_model.model.predict_proba(matrix)[:, 1]
    )

    return np.asarray(
        probabilities,
        dtype=np.float64,
    )
