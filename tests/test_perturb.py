import numpy as np

from app.perturb import generate_local_perturbations


def test_generates_exactly_ten_perturbations():
    baseline = np.array(
        [40.0, 5.0, 100.0, 2.0, 60.0]
    )

    result = generate_local_perturbations(baseline)

    assert result.shape == (10, 5)


def test_expected_perturbation_values():
    baseline = np.array(
        [40.0, 5.0, 100.0, 2.0, 60.0]
    )

    result = generate_local_perturbations(baseline)

    expected = np.array(
        [
            [37.0, 5.0, 100.0, 2.0, 60.0],
            [43.0, 5.0, 100.0, 2.0, 60.0],
            [40.0, 4.4, 100.0, 2.0, 60.0],
            [40.0, 5.6, 100.0, 2.0, 60.0],
            [40.0, 5.0, 90.0, 2.0, 60.0],
            [40.0, 5.0, 110.0, 2.0, 60.0],
            [40.0, 5.0, 100.0, 1.75, 60.0],
            [40.0, 5.0, 100.0, 2.25, 60.0],
            [40.0, 5.0, 100.0, 2.0, 58.5],
            [40.0, 5.0, 100.0, 2.0, 61.5],
        ]
    )

    np.testing.assert_allclose(result, expected)


def test_perturbations_are_clipped_to_valid_domain():
    baseline = np.array(
        [0.0, 30.0, 5.0, 8.0, 105.0]
    )

    result = generate_local_perturbations(baseline)

    assert np.all(result[:, 0] >= 0.0)
    assert np.all(result[:, 1] <= 30.0)
    assert np.all(result[:, 2] >= 5.0)
    assert np.all(result[:, 3] <= 8.0)
    assert np.all(result[:, 4] <= 105.0)
