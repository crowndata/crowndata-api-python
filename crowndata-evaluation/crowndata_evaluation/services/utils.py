import json
import os
from typing import Dict

import numpy as np
from fastapi import HTTPException

# Get the EVALUATION_API_ENDPOINT environment variable
data_dir = os.getenv("DATA_DIR", "./public")


def read_trajectory_json(data_name: str, joint: str) -> np.ndarray:
    """
    Reads a JSON file and returns the data as a NumPy array.
    Args:
        data_name (str): Data Name.
        joint: Joint Name.
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

    file_path = f"{data_dir}/assets/{data_name}/trajectories/{joint}__trajectory.json"

    with open(file_path, "r") as file:
        data = json.load(file)

    xyzrpy_array = np.array(data.get("data"))

    # Output type check
    if not isinstance(xyzrpy_array, np.ndarray):
        raise TypeError(
            f"Expected output to be a NumPy array, but got {type(xyzrpy_array).__name__}"
        )

    return xyzrpy_array


def fetch_trajectory_json(data_name: str, joint: str) -> np.ndarray:
    """
    Reads a JSON file and returns the data as a NumPy array.

    Args:
        data_name (str): Data Name.
        joint(str): Joint Name.

    Returns:
        np.ndarray: An array containing the x, y, z, roll, pitch, and yaw values.

    Raises:
        HTTPException: If fetch failed.
    """

    try:
        return read_trajectory_json(data_name=data_name, joint=joint)

    except TypeError as e:
        # If the data doesn't exist, raise an HTTPException
        raise HTTPException(status_code=400, detail=str(e))

    except ValueError as e:
        # If the data doesn't exist, raise an HTTPException
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except FileNotFoundError:
        # If the data doesn't exist, raise an HTTPException
        raise HTTPException(
            status_code=400,
            detail="Data not exist",
        )

    except Exception as e:
        # Optionally catch other exceptions and return a custom message
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error: " + str(e),
        )


def fetch_information_json(data_name: str) -> Dict:
    """
    Reads a JSON file and returns the information.

    Args:
        data_name (str): The name of the data set.

    Returns:
        Dict: Information.

    Raises:
        FileNotFoundError: If the file is not found.
    """

    try:
        # Ensure the data_name is a string
        if not isinstance(data_name, str):
            raise TypeError(
                f"Expected data_name to be a string, but got {type(data_name).__name__}"
            )

        # Define the file path
        file_path = f"{data_dir}/assets/{data_name}/information.json"

        # Open and read the JSON file
        with open(file_path, "r") as file:
            return json.load(file)

    except FileNotFoundError:
        # Return the default sample rate if the file is not found
        return {}
    except json.JSONDecodeError:
        # Handle JSON decode errors (if the file is corrupted or not properly formatted)
        return {}


def fetch_joint_json(data_name: str) -> Dict:
    """
    Reads a JSON file and returns the information.

    Args:
        data_name (str): The name of the data set.

    Returns:
        Dict: Joint data.

    Raises:
        FileNotFoundError: If the file is not found.
    """

    try:
        # Ensure the data_name is a string
        if not isinstance(data_name, str):
            raise TypeError(
                f"Expected data_name to be a string, but got {type(data_name).__name__}"
            )

        # Define the file path
        file_path = f"{data_dir}/assets/{data_name}/trajectories/joint_positions.json"

        # Open and read the JSON file
        with open(file_path, "r") as file:
            return json.load(file)

    except FileNotFoundError:
        # Return the default sample rate if the file is not found
        return {}
    except json.JSONDecodeError:
        # Handle JSON decode errors (if the file is corrupted or not properly formatted)
        return {}


default_sample_rate = 10


def fetch_trajectory_sample_rate(data_name: str) -> float:
    """
    Reads a JSON file and returns the trajectory sample rate.

    Args:
        data_name (str): The name of the data set.

    Returns:
        float: The sample rate in Hz or the default of 10 Hz if not found.

    Raises:
        FileNotFoundError: If the file is not found.
    """

    try:
        # Ensure the data_name is a string
        if not isinstance(data_name, str):
            raise TypeError(
                f"Expected data_name to be a string, but got {type(data_name).__name__}"
            )

        # Define the file path
        file_path = f"{data_dir}/assets/{data_name}/information.json"

        # Open and read the JSON file
        with open(file_path, "r") as file:
            data = json.load(file)

            # Retrieve the sample rate, default to default_sample_rate if not present
            return data.get("armSamplingRate", default_sample_rate)

    except FileNotFoundError:
        # Return the default sample rate if the file is not found
        return default_sample_rate
    except json.JSONDecodeError:
        # Handle JSON decode errors (if the file is corrupted or not properly formatted)
        return default_sample_rate


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
