import base64
import os
from typing import List

from fastapi import APIRouter, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

# Get the EVALUATION_API_ENDPOINT environment variable
data_dir = os.getenv("DATA_DIR", "./public")
open_api_key = os.getenv("OPENAI_API_KEY")

client = None
if open_api_key:
    client = OpenAI()

analyze_image_router = APIRouter()


# Request model
class AnalyzeImageRequest(BaseModel):
    imgs: List[str] = Field(
        None,
        example=[
            "assets/droid_00000000/images/wrist_image_left__image_00000005.webp",
            "assets/droid_00000000/images/wrist_image_left__image_00000165.webp",
        ],
    )
    prompt: str = Field(None, example="Has the item been moved? yes or no only.")


# Response model
class AnalyzeImageResponse(BaseModel):
    columns: List[str]
    data: List[List[float]]


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@analyze_image_router.post(
    "",
    summary="Openai analyze picture",
    description="Analyze picture with prompt",
)
async def post(request: AnalyzeImageRequest):
    if len(request.imgs) == 0:
        raise HTTPException(
            status_code=400,
            detail="Data validation error: No image found",
        )
    if len(request.imgs) >= 3:
        raise HTTPException(
            status_code=400,
            detail="Data validation error: To many image found. Pass less than three images",
        )

    base64_images = [encode_image(f"{data_dir}/{img}") for img in request.imgs]

    if not base64_images:
        raise HTTPException(
            status_code=400,
            detail="Data validation error: No image found",
        )

    image_urls = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        }
        for base64_image in base64_images
    ]
    response = {}
    if client:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{request.prompt}",
                        },
                    ]
                    + image_urls,
                }
            ],
        )

    return response.choices[0]
