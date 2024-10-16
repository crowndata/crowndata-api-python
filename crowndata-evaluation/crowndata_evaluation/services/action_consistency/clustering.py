import numpy as np
from typing import Dict
import sklearn.cluster
from scipy.spatial import cKDTree


def sklearn_cluster_wrapper(
    data: np.ndarray, class_name: str, args: Dict
) -> Dict[int, np.ndarray]:
    """Use sklearn.cluster to cluster data.

    Parameters
    ----------
    data : np.ndarray
        Data to cluster. (number of samples, state_dimension)
    class_name : str
        Name of the Clustering Algorithm class in sklearn.cluster to use.
    args : Dict
        Arguments to pass to the class or function.

    Returns
    -------
    Dict[int, np.ndarray]
        A dictionary where each key is a cluster index, and the value is the
        states or actions in that cluster.
    """
    # check if given name is in sklearn.cluster
    if not hasattr(sklearn.cluster, class_name):
        raise ValueError(f"{class_name} is not in sklearn.cluster")
    # check if class name has caps, otherwise it's a method
    if class_name.islower():
        raise ValueError(
            f"{class_name} does not contain any uppercase letters. Use the class version instead of the method."
        )

    # TODO: Edge case empty data
    clusters = {}
    clustering_alg = getattr(sklearn.cluster, class_name)(**args)
    labels = clustering_alg.fit_predict(data)

    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    for key in clusters:
        clusters[key] = data[clusters[key]]

    return clusters


def define_clusters(data: np.ndarray, epsilon: float) -> Dict[int, np.ndarray]:
    """Cluster data points based on epsilon distance using scipy's cKDTree.

    This algorithm groups data points into clusters where points within each
    cluster are at most 'epsilon' distance apart.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The input data to cluster.
    epsilon : float
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.

    Returns
    -------
    clusters : dict
        A dictionary where each key is a cluster label (integer) and the value
        is an array of shape (n_samples_in_cluster, n_features) containing the
        data points in that cluster.

    Notes
    -----
    - Uses scipy's cKDTree for efficient nearest neighbor queries.
    - Empty input data will return an empty dictionary.

    The algorithm can be described mathematically as follows:

    1. For each point x_i in the dataset X:
       - Find all points x_j such that ||x_i - x_j|| <= epsilon
       - These points form a neighborhood N_i

    2. A cluster C_k is formed by the union of neighborhoods that intersect:
       C_k = Union(N_i) for all i where N_i intersects with any N_j in C_k

    3. The process continues until all points are assigned to a cluster.

    Examples
    --------
    >>> data = np.array([[1, 1, 1, 1, 1, 1], [1.5, 1, 1, 1, 1, 1]])
    >>> epsilon = 1
    >>> define_clusters(data, epsilon)
    {0: array([[1, 1, 1, 1, 1, 1],
               [1.5, 1, 1, 1, 1, 1]])}
    """

    if len(data) == 0:
        return {}

    tree = cKDTree(data)
    neighbors = tree.query_ball_tree(tree, r=epsilon)

    clusters: Dict[int, np.ndarray] = {}
    visited = np.zeros(len(data), dtype=bool)
    cluster_idx = 0

    for i in range(len(data)):
        if not visited[i]:
            cluster_points = []
            stack = [i]
            visited[i] = True

            while stack:
                current = stack.pop()
                cluster_points.append(current)

                for neighbor in neighbors[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)

            cluster_points.sort()
            clusters[cluster_idx] = data[cluster_points]
            cluster_idx += 1

    return clusters
