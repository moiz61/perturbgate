import numpy as np

from app.gating import evaluate_stability


def test_stable_prediction_is_accepted():
    perturbed = np.array(
        [
            0.71,
            0.69,
            0.74,
            0.66,
            0.73,
            0.68,
            0.72,
            0.70,
            0.67,
            0.75,
        ]
    )

    result = evaluate_stability(
        baseline_probability=0.70,
        perturbed_probabilities=perturbed,
        stability_threshold=0.90,
        max_probability_shift=0.18,
    )

    assert result.label == "unstable"
    assert result.decision == "accept"
    assert result.fragile is False
    assert result.agreement == 1.0
    assert result.perturbations_evaluated == 10


def test_class_flips_trigger_abstention():
    perturbed = np.array(
        [
            0.40,
            0.42,
            0.44,
            0.46,
            0.48,
            0.52,
            0.54,
            0.56,
            0.58,
            0.60,
        ]
    )

    result = evaluate_stability(
        baseline_probability=0.51,
        perturbed_probabilities=perturbed,
        stability_threshold=0.90,
        max_probability_shift=0.18,
    )

    assert result.decision == "abstain"
    assert result.fragile is True
    assert result.agreement < 0.90


def test_large_probability_shift_triggers_abstention():
    perturbed = np.array(
        [
            0.61,
            0.62,
            0.63,
            0.64,
            0.65,
            0.66,
            0.67,
            0.68,
            0.69,
            0.95,
        ]
    )

    result = evaluate_stability(
        baseline_probability=0.70,
        perturbed_probabilities=perturbed,
        stability_threshold=0.90,
        max_probability_shift=0.18,
    )

    assert result.agreement == 1.0
    assert result.max_probability_shift > 0.18
    assert result.decision == "abstain"
    assert result.fragile is True
