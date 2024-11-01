import os
from typing import List

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field

from crowndata_evaluation.services.kinematics.resources import get_urdf_file_path
from crowndata_evaluation.services.kinematics.urdf import (
    forward_kinematics,
    get_robot_from_urdf,
)
from crowndata_evaluation.services.utils import fetch_joint_json

# Get the EVALUATION_API_ENDPOINT environment variable
data_dir = os.getenv("DATA_DIR", "./public")

pos_router = APIRouter()


# Request model
class PosRequest(BaseModel):
    urdf: str = Field(None, example="droid")
    dataName: str = Field(None, example="droid_00000000")
    linkName: str = Field(None, example="robotiq_85_adapter_link")


# Response model
class PosResponse(BaseModel):
    columns: List[str]
    data: List[List[float]]


@pos_router.post(
    "",
    summary="Pos",
    description="Retrieve Pos",
    response_model=PosResponse,
)
async def post(request: PosRequest):
    urdf_file_path = get_urdf_file_path(request.urdf)
    robot = get_robot_from_urdf(f"{urdf_file_path}")

    joint_data = fetch_joint_json(data_name=request.dataName)
    joint_records = pd.DataFrame(
        columns=joint_data.get("columns"), data=joint_data.get("data")
    ).to_dict(orient="records")
    link_name = request.linkName

    data = forward_kinematics(
        robot=robot, joint_records=joint_records, link_name=link_name
    )

    columns = ["x", "y", "z", "roll", "pitch", "yaw"]

    return {
        "columns": columns,
        "data": data,
    }
