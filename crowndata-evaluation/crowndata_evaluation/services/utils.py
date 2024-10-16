import json
import numpy as np


def read_trajectory_json(data_name: str) -> np.ndarray:
    """
    Reads a JSON file and returns the data as a NumPy array.
    Args:
        data_name (str): Data Name.
    Returns:
        np.ndarray: An array containing the x, y, z, roll, pitch, and yaw values.
    Raises:
        TypeError: If the input file path is not a string.
        ValueError: If the JSON data is invalid or if data cannot be converted to floats.
    """
    # Input type check
    if not isinstance(data_name, str):
        raise TypeError(
            f"Expected data_name to be a string, but got {type(data_name).__name__}"
        )

    file_path = (
        f"./public/data/{data_name}/trajectories/cartesian_position__trajectory.json"
    )

    with open(file_path, "r") as file:
        data = json.load(file)

    # Validate that data is a list of dictionaries
    if not isinstance(data, list):
        raise ValueError("Expected JSON data to be a list of dictionaries.")

    xyzrpy_array = np.array(
        [
            [
                entry.get("x"),
                entry.get("y"),
                entry.get("z"),
                entry.get("roll"),
                entry.get("pitch"),
                entry.get("yaw"),
            ]
            for entry in data
            if isinstance(entry, dict)  # Ensure each entry is a dictionary
        ]
    )

    # Output type check
    if not isinstance(xyzrpy_array, np.ndarray):
        raise TypeError(
            f"Expected output to be a NumPy array, but got {type(xyzrpy_array).__name__}"
        )

    return xyzrpy_array


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Ensure both arrays have the same shape
    if a.shape != b.shape:
        raise ValueError("Arrays must have the same shape.")

    # Calculate the dot product of the two arrays
    dot_product = np.dot(a, b)

    # Calculate the magnitude (norm) of each array
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Return the cosine similarity
    return dot_product / (norm_a * norm_b)
