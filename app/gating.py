from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CLASSIFICATION_THRESHOLD = 0.50


@dataclass(frozen=True)
class GateResult:
    label: str
    agreement: float
    max_probability_shift: float
    perturbations_evaluated: int
    fragile: bool
    decision: str


def evaluate_stability(
    baseline_probability: float,
    perturbed_probabilities: np.ndarray,
    stability_threshold: float,
    max_probability_shift: float,
) -> GateResult:
    """
    Decide whether the baseline prediction is locally stable.

    The gate checks two things:

    1. Agreement:
       What fraction of perturbed predictions keep the same class
       as the baseline prediction?

    2. Probability shift:
       How far does any perturbed probability move away from the
       baseline probability?

    The prediction is accepted only when BOTH checks pass.
    """

    baseline_probability = float(baseline_probability)

    perturbed_probabilities = np.asarray(
        perturbed_probabilities,
        dtype=np.float64,
    ).reshape(-1)

    if perturbed_probabilities.size == 0:
        raise ValueError(
            "At least one perturbed probability is required."
        )

    if not 0.0 <= baseline_probability <= 1.0:
        raise ValueError(
            "Baseline probability must be between 0 and 1."
        )

    if np.any(
        (perturbed_probabilities < 0.0)
        | (perturbed_probabilities > 1.0)
    ):
        raise ValueError(
            "Perturbed probabilities must be between 0 and 1."
        )

    baseline_class = int(
        baseline_probability >= CLASSIFICATION_THRESHOLD
    )

    perturbed_classes = (
        perturbed_probabilities >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    agreement = float(
        np.mean(perturbed_classes == baseline_class)
    )

    observed_max_shift = float(
        np.max(
            np.abs(
                perturbed_probabilities
                - baseline_probability
            )
        )
    )

    agreement_failed = agreement < stability_threshold

    shift_failed = (
        observed_max_shift > max_probability_shift
    )

    fragile = bool(
        agreement_failed or shift_failed
    )

    decision = (
        "abstain"
        if fragile
        else "accept"
    )

    label = (
        "unstable"
        if baseline_class == 1
        else "healthy"
    )

    return GateResult(
        label=label,
        agreement=agreement,
        max_probability_shift=observed_max_shift,
        perturbations_evaluated=int(
            perturbed_probabilities.size
        ),
        fragile=fragile,
        decision=decision,
    )
