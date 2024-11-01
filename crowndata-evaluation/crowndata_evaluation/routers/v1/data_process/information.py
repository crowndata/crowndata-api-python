import json
import os
import subprocess
import time
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from crowndata_evaluation.services.kinematics.resources import get_urdf_dict

# Get the EVALUATION_API_ENDPOINT environment variable
data_dir = os.getenv("DATA_DIR", "./public")

data_process_information_router = APIRouter()


# Request model
class InformationRequest(BaseModel):
    targetDir: Optional[str] = Field(None, example="assets/rm65_abc_20241031_010921")
    dataName: Optional[str] = Field(..., example="robot_data_example")
    startTime: Optional[str] = Field(..., example="20230417T144805.000Z")
    endTime: Optional[str] = Field(..., example="20230417T145051.000Z")
    robotEmbodiment: Optional[str] = Field(..., example="rm_65_gazebo_dual_gripper")
    robotSerialNumber: Optional[str] = Field(..., example="rm_65_123456")
    videoSamplingRate: Optional[int] = Field(..., example=50)
    armSamplingRate: Optional[int] = Field(..., example=50)
    sensorSamplingRate: Optional[int] = Field(..., example=50)
    operatorName: Optional[str] = Field(..., example="Crown Data")
    taskDescription: Optional[str] = Field(..., example="Sample Task")
    subtaskDescription: Optional[str] = Field(..., example="Subtask Description")
    taskState: Optional[str] = Field(..., example="SUCCESS")
    subtaskState: Optional[str] = Field(..., example="SUCCESS")
    dataLength: Optional[int] = Field(default=0, example=0)
    durationInSeconds: Optional[int] = Field(default=0, example=0)
    cameras: Optional[List[str]] = Field(
        ..., example=["cam_high", "cam_left_wrist", "cam_right_wrist", "cam_low"]
    )
    dataSource: Optional[str] = Field(
        ..., example="https://droid-dataset.github.io/droid/the-droid-dataset"
    )


# Response model
class InformationResponse(BaseModel):
    status: Literal["success", "failure", None] = Field(None, example="success")
    messages: Optional[List[str]] = Field(
        None, example=["Cannot process trajectory. Or Missing Field"]
    )


@data_process_information_router.post(
    "",
    summary="Trajectory",
    description="Retrieve Trajectory",
    response_model=InformationResponse,
)
async def post(request: InformationRequest):
    start_time = time.time()
    target_dir = f"{data_dir}/{request.targetDir}"
    urdf = request.robotEmbodiment

    if urdf:
        joints = get_urdf_dict(urdf=urdf).get("joint_names", [])
    if not joints:
        raise HTTPException(
            status_code=400,
            detail=f"Robot Embodiment Not Supported: {request.robotEmbodiment}",
        )

    messages = []

    information = {}
    if request.dataName:
        information["dataName"] = request.dataName
    if request.startTime:
        information["startTime"] = request.startTime
    if request.endTime:
        information["endTime"] = request.endTime
    if request.robotEmbodiment:
        information["robotEmbodiment"] = request.robotEmbodiment
    if request.robotSerialNumber:
        information["robotSerialNumber"] = request.robotSerialNumber
    if request.videoSamplingRate:
        information["videoSamplingRate"] = request.videoSamplingRate
    if request.armSamplingRate:
        information["armSamplingRate"] = request.armSamplingRate
    if request.sensorSamplingRate:
        information["sensorSamplingRate"] = request.sensorSamplingRate
    if request.operatorName:
        information["operatorName"] = request.operatorName
    if request.taskDescription:
        information["taskDescription"] = request.taskDescription
    if request.subtaskDescription:
        information["subtaskDescription"] = request.subtaskDescription
    if request.taskState:
        information["taskState"] = request.taskState
    if request.subtaskState:
        information["subtaskState"] = request.subtaskState
    if request.dataLength:
        information["dataLength"] = request.dataLength
    if request.durationInSeconds:
        information["durationInSeconds"] = request.durationInSeconds
    if request.cameras:
        information["cameras"] = request.cameras
    if request.dataSource:
        information["dataSource"] = request.dataSource
    if joints:
        information["joints"] = joints

    subprocess.call(["mkdir", "-p", f"{target_dir}"])
    with open(f"{target_dir}/information.json", "w") as json_file:
        json.dump(information, json_file, indent=2)
    messages.append(f"Information: {information}")
    messages.append(f"Finished: {(time.time() - start_time):.2f} seconds")

    return {"status": "success", "messages": messages}
