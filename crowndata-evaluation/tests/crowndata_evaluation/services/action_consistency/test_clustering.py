import pytest
import numpy as np
from crowndata_evaluation.services.action_consistency.clustering import (
    define_clusters,
    sklearn_cluster_wrapper,
)


@pytest.mark.parametrize(
    "data, epsilon, expected_num_clusters",
    [
        # Edge case: large epsilon = 1 big cluster
        [
            np.random.rand(100, 6),
            10000,
            1,
        ],
        # Edge case: small epsilon = 100 small clusters
        [
            np.random.rand(100, 6),
            0,
            100,
        ],
        [
            np.array(
                [
                    np.ones(6),
                    np.ones(6) + 0.1,
                ]
            ),
            0.25,
            1,
        ],
        [
            np.array(
                [
                    np.ones(6),
                    np.ones(6) + 0.1,
                ]
            ),
            0.24,
            2,
        ],
    ],
)
def test_define_clusters(data, epsilon, expected_num_clusters):
    """Test the define_clusters function."""
    clusters = define_clusters(data, epsilon)
    assert len(clusters) == expected_num_clusters


@pytest.mark.parametrize(
    "data, method_name, args, expected_num_clusters",
    [
        # Edge case: all data points are the same, expect 1 cluster regardless of n_clusters
        [
            np.ones((100, 6)),
            "KMeans",
            {
                "n_clusters": 20,
            },
            1,
        ],
        # Normal case: random data, expect the specified number of clusters
        [
            np.random.rand(100, 6),
            "KMeans",
            {
                "n_clusters": 20,
            },
            20,
        ],
        # Case with some duplicate points, expect fewer clusters than specified
        [
            np.array(
                [
                    np.ones(6),
                    np.ones(6),
                    np.ones(6) + 1,
                    np.ones(6) + 1 + 1e-5,
                ]
            ),
            "KMeans",
            {
                "n_clusters": 4,
            },
            3,
        ],
        #TODO: Edge case: empty data
    ],
)
def test_sklearn_cluster_wrapper(data, method_name, args, expected_num_clusters):
    """Test the sklearn_cluster_wrapper function."""
    clusters = sklearn_cluster_wrapper(data, method_name, args)
    assert len(clusters) == expected_num_clusters