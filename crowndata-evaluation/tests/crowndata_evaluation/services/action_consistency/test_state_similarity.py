import pytest
import numpy as np
from crowndata_evaluation.services.action_consistency.state_similarity_calculator import (
    StateSimilarityCalculator,
)


@pytest.fixture(scope="module")
def generate_random_data():
    """
    Fixture to generate random data simulating Cartesian pose data.

    Returns
    -------
    tuple of ndarray
        Two randomly generated datasets, each with 100 rows and 6 columns representing
        (x, y, z, roll, pitch, yaw).
    """
    data1 = np.random.rand(100, 6)  # First dataset
    data2 = np.random.rand(100, 6)  # Second dataset

    return data1, data2


@pytest.mark.parametrize("epsilon", [0.5, 1.0])
def test_state_similarity_single_trajectory(epsilon, generate_random_data):
    """
    Test the state similarity for a single randomly generated trajectory.

    Parameters
    ----------
    epsilon : float
        The epsilon value used by the StateSimilarityCalculator to compute similarity.
    generate_random_data : fixture
        Provides the randomly generated data from the fixture.
    """
    data1, _ = generate_random_data

    # Initialize StateSimilarityCalculator with the given epsilon
    calc = StateSimilarityCalculator(epsilon)

    # Compute similarity for the dataset
    similarity = calc.compute_similarity(data1)

    # Ensure similarity score is within the valid range [0, 1]
    assert (
        0 <= similarity <= 1
    ), f"Expected similarity to be between 0 and 1, but got {similarity}"


@pytest.mark.parametrize("epsilon", [0.5])
def test_combined_trajectory_similarity(epsilon, generate_random_data):
    """
    Test similarity between two combined randomly generated trajectories.

    Parameters
    ----------
    epsilon : float
        The epsilon value used by the StateSimilarityCalculator to compute similarity.
    generate_random_data : fixture
        Provides the randomly generated data from the fixture.
    """
    data1, data2 = generate_random_data

    # Initialize StateSimilarityCalculator with the given epsilon
    calc = StateSimilarityCalculator(epsilon)

    # Combine both datasets and compute similarity
    combined_data = np.vstack((data1, data2))
    combined_similarity = calc.compute_similarity(combined_data)

    # Ensure combined similarity score is within the valid range [0, 1]
    assert (
        0 <= combined_similarity <= 1
    ), f"Expected combined similarity to be between 0 and 1, but got {combined_similarity}"


def test_empty_data_similarity():
    """
    Test the state similarity with an empty dataset.

    This test checks whether the similarity calculator can handle an empty dataset correctly,
    returning a similarity score of 0.

    """
    # Create an empty dataset
    empty_data = np.array([]).reshape(0, 6)

    # Initialize StateSimilarityCalculator with a fixed epsilon value
    calc = StateSimilarityCalculator(epsilon=0.5)

    # Compute similarity for the empty dataset
    similarity = calc.compute_similarity(empty_data)

    # Ensure the similarity score is 0 for the empty dataset
    assert similarity == 0, "Expected similarity for empty data to be 0"
