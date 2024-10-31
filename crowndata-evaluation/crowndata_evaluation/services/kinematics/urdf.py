from typing import Dict, List

import numpy as np
from urdf_parser_py.urdf import URDF


def get_robot_from_urdf(file_path: str):
    # Load the URDF model
    return URDF.from_xml_file(file_path)


def get_transform_from_origin(origin):
    """Compute a transformation matrix from URDF origin."""
    xyz = origin.xyz if origin and origin.xyz else [0, 0, 0]
    rpy = origin.rpy if origin and origin.rpy else [0, 0, 0]
    # Convert to transformation matrix
    T = np.eye(4)
    T[:3, 3] = xyz
    R = rpy_to_matrix(rpy)
    T[:3, :3] = R
    return T


def rpy_to_matrix(rpy: List[float]) -> np.ndarray:
    """Convert roll, pitch, yaw to rotation matrix."""
    roll, pitch, yaw = rpy
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )
    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    Rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    return Rz @ Ry @ Rx


def matrix_to_rpy(matrix: np.ndarray) -> List[float]:
    """Convert rotation matrix to roll, pitch, yaw."""
    if matrix[2, 0] != 1 and matrix[2, 0] != -1:
        pitch = -np.arcsin(matrix[2, 0])
        roll = np.arctan2(matrix[2, 1] / np.cos(pitch), matrix[2, 2] / np.cos(pitch))
        yaw = np.arctan2(matrix[1, 0] / np.cos(pitch), matrix[0, 0] / np.cos(pitch))
    else:
        # Gimbal lock case
        yaw = 0  # Or set to a different fixed value if needed
        if matrix[2, 0] == -1:
            pitch = np.pi / 2
            roll = np.arctan2(matrix[0, 1], matrix[0, 2])
        else:
            pitch = -np.pi / 2
            roll = np.arctan2(-matrix[0, 1], -matrix[0, 2])

    return [float(roll), float(pitch), float(yaw)]


def joint_transform(joint, angle: float) -> np.ndarray:
    """Compute the transformation matrix for a joint."""
    T_origin = get_transform_from_origin(joint.origin)

    if joint.type == "revolute" or joint.type == "continuous":
        # For a revolute joint, rotate around the joint axis
        axis = np.array(joint.axis)
        angle = angle  # in radians
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        R = np.eye(3) * cos_a + (1 - cos_a) * np.outer(axis, axis)
        R += (
            np.array(
                [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
            )
            * sin_a
        )
        T_rotation = np.eye(4)
        T_rotation[:3, :3] = R
        return T_origin @ T_rotation
    elif joint.type == "prismatic":
        # For prismatic joints, translate along the joint axis
        T_translation = np.eye(4)
        T_translation[:3, 3] = np.array(joint.axis) * angle
        return T_origin @ T_translation
    else:
        # Fixed joints simply use the origin transformation
        return T_origin


def forward_kinematics(
    robot, joint_records: List[Dict[str, float]], link_name: str
) -> List[List[float]]:
    """Compute the positions of each joint and link based on joint angles."""
    T = np.eye(4)  # Starting from the base frame
    positions = []

    ancestors = robot.get_chain(robot.get_root(), link_name)
    joint_map = {joint.name: joint for joint in robot.joints}

    for joint_record in joint_records:
        for joint_name in ancestors:
            if joint_name in joint_map.keys():
                joint = joint_map[joint_name]
                angle = joint_record.get(joint.name, 0.0)  # Get angle or default to 0
                angle = joint_record.get(joint.name, 0.0)  # Get angle or default to 0
                T = T @ joint_transform(joint, angle)  # Apply joint transformation

                if joint.child == link_name:
                    position = [
                        round(num, 4) for num in T[:3, 3].tolist() + matrix_to_rpy(T)
                    ]
                    positions.append(position)  # Store the position of each link
                    break

    return positions


def find_joint(robot, joint_name):
    for joint in robot.joints:
        if joint.name == joint_name:
            return joint
