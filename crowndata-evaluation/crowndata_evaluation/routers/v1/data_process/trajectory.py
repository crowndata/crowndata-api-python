import os
import subprocess
import time
from typing import List, Literal, Optional

import cv2
import h5py
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from crowndata_evaluation.services.kinematics.resources import (
    get_urdf_dict,
    get_urdf_file_path,
    h5_to_dict,
)
from crowndata_evaluation.services.kinematics.urdf import (
    find_joint,
    forward_kinematics,
    get_robot_from_urdf,
)

# Get the EVALUATION_API_ENDPOINT environment variable
data_dir = os.getenv("DATA_DIR", "./public")

trajectory_router = APIRouter()


# Request model
class TrajectoryRequest(BaseModel):
    urdf: str = Field(None, example="rm_65_gazebo_dual_gripper")
    sourcePath: str = Field(
        None, example="data/rm65_abc_20241031_010921/rm65_abc_20241031_010921.h5"
    )
    targetDir: str = Field(None, example="assets/rm65_abc_20241031_010921")
    cameras: List[str] = Field(
        None, example=["cam_high", "cam_left_wrist", "cam_right_wrist", "cam_low"]
    )


# Response model
class TrajectoryResponse(BaseModel):
    status: Literal["success", "failure", None] = Field(None, example="success")
    messages: Optional[List[str]] = Field(
        None, example=["Cannot process trajectory. Or Missing Field"]
    )


@trajectory_router.post(
    "",
    summary="Trajectory",
    description="Retrieve Trajectory",
    response_model=TrajectoryResponse,
)
async def post(request: TrajectoryRequest):
    start_time = time.time()
    source_path = f"{data_dir}/{request.sourcePath}"
    target_dir = f"{data_dir}/{request.targetDir}"
    cameras = request.cameras
    urdf = request.urdf

    # Load the URDF model
    urdf_file_path = get_urdf_file_path(urdf=urdf)
    robot = get_robot_from_urdf(urdf_file_path)

    movable_joints = get_urdf_dict(urdf=urdf).get("movable_joints", [])
    joint_names = get_urdf_dict(urdf=urdf).get("joint_names", [])

    messages = []
    #### process trajectory ####
    try:
        trajectory_data = {}
        with h5py.File(source_path, "r") as file:
            for key in file.keys():
                if isinstance(file[key], h5py.Dataset):
                    trajectory_data[key] = file[key][:]
                elif isinstance(file[key], h5py.Group):
                    trajectory_data[key] = h5_to_dict(file[key])

        subprocess.call(["mkdir", "-p", f"{target_dir}"])
        subprocess.call(["mkdir", "-p", f"{target_dir}/trajectories"])
        # joint position
        joint_positions = trajectory_data["action"]
        # convert to radian
        joint_positions[:, :6] = np.radians(joint_positions[:, :6])
        joint_positions[:, 7:13] = np.radians(joint_positions[:, :6])
        # convert to meter
        joint_positions[:, 6] = joint_positions[:, 6] / 1000.0 * 60
        joint_positions[:, 13] = joint_positions[:, 13] / 1000.0 * 60

        columns = [f"joint{i}" for i in range(1, len(trajectory_data["action"][0]) + 1)]
        if len(trajectory_data["action"][0]) == len(movable_joints):
            columns = movable_joints

        df = pd.DataFrame(
            joint_positions,
            columns=columns,
        )
        df_file_path = f"{target_dir}/trajectories/joint_positions.json"
        df.to_json(
            df_file_path,
            orient="split",
            index=False,
            double_precision=3,
        )
        messages.append(f"Created New file for joint positions: {df_file_path}")

        joint_records = df.to_dict(orient="records")
        trajectories = {}
        for joint_name in joint_names:
            link_name = find_joint(robot, joint_name).child
            trajectories[joint_name] = forward_kinematics(
                robot=robot, joint_records=joint_records, link_name=link_name
            )
            df = pd.DataFrame(
                trajectories[joint_name],
                columns=["x", "y", "z", "roll", "pitch", "yaw"],
            )
            df_file_path = f"{target_dir}/trajectories/{joint_name}__trajectory.json"
            df.to_json(
                df_file_path,
                orient="split",
                index=False,
                double_precision=3,
            )
            messages.append(
                f"Created New trajectory file for joint {joint_name}: {df_file_path}"
            )

        #### Process Images ####
        images = {}
        for i, camera in enumerate(cameras):
            camera_images = trajectory_data["observations"][f"images/{camera}"]
            if camera not in images.keys():
                images[camera] = []
                output_folder = f"{target_dir}/images"
                subprocess.call(["mkdir", "-p", output_folder])
            for camera_image in camera_images:
                decoded_image = cv2.imdecode(camera_image, cv2.IMREAD_COLOR)

                # Get the current width and height of the image
                height, width, channels = decoded_image.shape

                # Calculate the new width while keeping the aspect ratio
                aspect_ratio = width / height
                new_height = 50
                new_width = int(new_height * aspect_ratio)

                # Resize the image
                resized_image = cv2.resize(decoded_image, (new_width, new_height))

                # Save or show the resized image
                images[camera].append(resized_image)
            df_img = pd.DataFrame()

        for camera in images:
            for j, image in enumerate(images[camera]):
                # Save the image in WebP format
                cv2.imwrite(f"{target_dir}/images/{camera}__image_{j:08d}.webp", image)
            df_img[camera] = [f"{camera}__image_{j:08d}.webp" for j in range(len(df))]
            messages.append(
                f"Created New file of downsized images for camera: {camera}."
            )

        df_file_path = f"{target_dir}/images/camera_images.json"
        df_img.to_json(
            df_file_path,
            orient="split",
            index=False,
        )
        messages.append(f"Created New file for Image path: {df_file_path}")
        messages.append(f"Finished: {(time.time() - start_time):.2f} seconds")

    except (
        Exception
    ) as e:  # Raise an HTTPException with a 400 status code for bad request
        raise HTTPException(
            status_code=400,
            detail=f"Error: {e}",
        )

    return {"status": "success", "messages": messages}
