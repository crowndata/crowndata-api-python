import os

import h5py

# Get the EVALUATION_API_ENDPOINT environment variable
data_dir = os.getenv("DATA_DIR", "./public")

urdf_map = {
    "rm_65_gazebo_dual_gripper": {
        "urdf_file_path": f"{data_dir}/geometries/ros2_rm_robot/rm_description/urdf/rm_65_gazebo_dual_gripper.urdf",
        "movable_joints": [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "Joint_finger1",
            "joint1_2",
            "joint2_2",
            "joint3_2",
            "joint4_2",
            "joint5_2",
            "joint6_2",
            "Joint_finger1_2",
        ],
        "joint_names": ["joint8", "joint8_2"],
    },
    "droid": {
        "urdf_file_path": f"{data_dir}/geometries/DROID/panda.urdf",
        "movable_joints": [f"panda_joint{i}" for i in range(1, 8)],
        "joint_names": ["panda_joint8"],
    },
}


def get_urdf_map():
    return urdf_map


def get_urdf_dict(urdf: str):
    return urdf_map.get(urdf, None)


def get_urdf_file_path(urdf: str):
    if get_urdf_dict(urdf) is not None:
        return urdf_map.get(urdf, None).get("urdf_file_path", None)


def h5_to_dict(data):
    """Recursively loads an h5py group into a dictionary."""
    data = {}
    if isinstance(data, h5py.Group):
        for key, item in data.items():
            if isinstance(
                item, h5py.Dataset
            ):  # If it's a dataset, read it into the dictionary
                data[key] = item[:]
            elif isinstance(item, h5py.Group):  # If it's a group, call recursively
                data[key] = h5_to_dict(item)
    return data
