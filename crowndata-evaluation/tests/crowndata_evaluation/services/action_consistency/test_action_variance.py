import numpy as np
import pytest
from crowndata_evaluation.services.action_consistency.action_variance_calculator import (
    ActionVarianceCalculator,
)


@pytest.mark.parametrize(
    "traj_a, traj_b, variance",
    [
        [
            [
                [
                    -2.89342165,
                    -0.198712066,
                    0.126990348,
                    -2.89334679,
                    -0.201124683,
                    0.126930386,
                ]
            ],
            [
                [
                    -2.89342165,
                    -0.198712066,
                    0.126990348,
                    -2.89334679,
                    -0.201124683,
                    0.126930386,
                ]
            ],
            0,
        ]
    ],
)
def test_same_trajectory_action_variance(traj_a, traj_b, variance):
    calculator = ActionVarianceCalculator(epsilon=0.1)
    combined_traj = np.array(traj_a + traj_b)
    assert calculator.calculate_action_variance(combined_traj) == variance


@pytest.mark.parametrize(
    "trajectories, epsilon, expected_variance",
    [
        (
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
                [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                [10.1, 11.1, 12.1, 13.1, 14.1, 15.1],
            ],
            100.0,  # Large epsilon
            121.515,  # Updated expected variance
        ),
    ],
)
def test_large_epsilon_action_variance(trajectories, epsilon, expected_variance):
    calculator = ActionVarianceCalculator(epsilon=epsilon)
    calculated_variance = calculator.calculate_action_variance(np.array(trajectories))
    np.testing.assert_almost_equal(calculated_variance, expected_variance, decimal=3)


@pytest.mark.parametrize(
    "trajectories, epsilon, expected_variance",
    [
        (
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
                [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                [10.1, 11.1, 12.1, 13.1, 14.1, 15.1],
            ],
            1e-6,  # Very small epsilon
            121.515,
        ),
    ],
)
def test_small_epsilon_action_variance(trajectories, epsilon, expected_variance):
    calculator = ActionVarianceCalculator(epsilon=epsilon)
    calculated_variance = calculator.calculate_action_variance(np.array(trajectories))
    np.testing.assert_almost_equal(calculated_variance, expected_variance, decimal=3)


@pytest.mark.parametrize(
    "trajectories, epsilon, expected_variance",
    [
        (
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ],
            0.1,  # Small epsilon
            0.0,  # Expected variance is zero for identical trajectories
        ),
    ],
)
def test_identical_trajectories_action_variance(
    trajectories, epsilon, expected_variance
):
    calculator = ActionVarianceCalculator(epsilon=epsilon)
    calculated_variance = calculator.calculate_action_variance(np.array(trajectories))
    np.testing.assert_almost_equal(calculated_variance, expected_variance, decimal=3)
