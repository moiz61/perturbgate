from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.gating import evaluate_stability
from app.model import load_model_bundle, predict_probabilities
from app.perturb import (
    FEATURE_MAXIMUMS,
    FEATURE_MINIMUMS,
    generate_local_perturbations,
)


MODEL_PATH = Path("artifacts/model.joblib")

FEATURE_NAMES = [
    "voltage_jitter_mv",
    "packet_retransmit_pct",
    "inference_latency_ms",
    "sensor_drift_sigma",
    "cpu_temp_c",
]


def main() -> None:
    loaded_model = load_model_bundle(MODEL_PATH)

    rng = np.random.default_rng(42)

    # Generate many valid telemetry points at once.
    candidates = rng.uniform(
        low=FEATURE_MINIMUMS,
        high=FEATURE_MAXIMUMS,
        size=(20_000, 5),
    )

    baseline_probabilities = predict_probabilities(
        loaded_model,
        candidates,
    )

    # Search points nearest the 0.50 decision boundary first.
    order = np.argsort(
        np.abs(baseline_probabilities - 0.5)
    )

    for index in order:
        baseline = candidates[index]
        baseline_probability = float(
            baseline_probabilities[index]
        )

        perturbations = generate_local_perturbations(
            baseline
        )

        perturbed_probabilities = predict_probabilities(
            loaded_model,
            perturbations,
        )

        gate = evaluate_stability(
            baseline_probability=baseline_probability,
            perturbed_probabilities=perturbed_probabilities,
            stability_threshold=0.90,
            max_probability_shift=0.18,
        )

        if gate.fragile:
            telemetry = {
                name: float(value)
                for name, value in zip(
                    FEATURE_NAMES,
                    baseline,
                )
            }

            payload = {
                "telemetry": telemetry,
            }

            print("=== ABSTAIN CANDIDATE FOUND ===")
            print()
            print("Baseline probability:")
            print(f"{baseline_probability:.6f}")
            print()
            print("Agreement:")
            print(f"{gate.agreement:.3f}")
            print()
            print("Maximum probability shift:")
            print(f"{gate.max_probability_shift:.6f}")
            print()
            print("Expected decision:")
            print(gate.decision)
            print()
            print("=== API PAYLOAD ===")
            print(
                json.dumps(
                    payload,
                    indent=2,
                )
            )
            return

    raise RuntimeError(
        "No fragile point found in candidate search."
    )


if __name__ == "__main__":
    main()
