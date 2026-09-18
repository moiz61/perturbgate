from __future__ import annotations

import numpy as np

from app.schemas import Telemetry


FEATURE_MINIMUMS = np.array(
    [0.0, 0.0, 5.0, 0.0, 25.0],
    dtype=np.float64,
)

FEATURE_MAXIMUMS = np.array(
    [120.0, 30.0, 500.0, 8.0, 105.0],
    dtype=np.float64,
)

PERTURBATION_SIZES = np.array(
    [3.0, 0.6, 10.0, 0.25, 1.5],
    dtype=np.float64,
)


def telemetry_to_array(telemetry: Telemetry) -> np.ndarray:
    return np.array(
        [
            telemetry.voltage_jitter_mv,
            telemetry.packet_retransmit_pct,
            telemetry.inference_latency_ms,
            telemetry.sensor_drift_sigma,
            telemetry.cpu_temp_c,
        ],
        dtype=np.float64,
    )


def generate_local_perturbations(
    baseline: np.ndarray,
) -> np.ndarray:
    """
    Generate exactly 10 deterministic local perturbations.

    Each of the five features is perturbed once downward
    and once upward while all other features remain fixed.
    """

    baseline = np.asarray(baseline, dtype=np.float64)

    if baseline.shape != (5,):
        raise ValueError(
            f"Expected baseline shape (5,), received {baseline.shape}."
        )

    perturbations = []

    for feature_index, step in enumerate(PERTURBATION_SIZES):
        lower = baseline.copy()
        lower[feature_index] -= step
        lower = np.clip(
            lower,
            FEATURE_MINIMUMS,
            FEATURE_MAXIMUMS,
        )
        perturbations.append(lower)

        upper = baseline.copy()
        upper[feature_index] += step
        upper = np.clip(
            upper,
            FEATURE_MINIMUMS,
            FEATURE_MAXIMUMS,
        )
        perturbations.append(upper)

    result = np.vstack(perturbations)

    if result.shape != (10, 5):
        raise RuntimeError(
            f"Expected perturbation matrix shape (10, 5), "
            f"received {result.shape}."
        )

    return result
