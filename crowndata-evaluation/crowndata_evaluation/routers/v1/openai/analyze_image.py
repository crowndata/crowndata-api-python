import base64
import os
from typing import List, Optional

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
    imgs: Optional[List[str]] = Field(
        None,
        example=[
            "assets/droid_00000000/images/wrist_image_left__image_00000005.webp",
            "assets/droid_00000000/images/wrist_image_left__image_00000165.webp",
        ],
    )
    base64EncodeImgs: Optional[List[str]] = Field(
        None,
        example=[
            "UklGRlgFAABXRUJQVlA4IEwFAAAQGQCdASpYADIAPm0wk0ekIqGhKrIMAIANiWQApLkAPwA6eT0fUvyI9k+zeAAQDYc9ZH4w9abbIdFDeZN6Wn7jZrvX2e/DRTF9x4J6VloreZL9bYmYgu97UdyPjQ9KbAIyR4T48/eJVb9TgRHlMdLVu/JTa0DD2uGIBt1J0V7sm6PYkDjNTgeXhVM56wCxLXrHLICfTzK2m3cpTPf+lU/MR411mxipG8SxCfPIlDTLboIk7sCsH2f1Of0uVJKA7WbtLVJtJore0AmO/aweVZySAAD+xNPxowIbP1nuOcAcyGffVemqRA95CqxFfwole+Jiu07GBcIP/0xW5vpfyzk+Jnvh7zIAiZR2zs9FfTJbHMzHkKoBcZV7y+nAGAa4gQG4kjAHbz2WHGBx7fedCJG690sELbZ7hVtbgqSDcuVKNVc14Q/cV/aIrlWsY1ZEpAv6IbEyDLj5f0ATn+rU33q8FUVKmns5mO5E+y/X9JBH3FQdjhv07vSgE85xl/41ajY+1b9xCkx5DzaLqxw5GNhagMeE1JbaotihkWA5dcjDHdASbe9qy8mWHqHpmQh/a6UPmeSg5KdfMc55TWdqjEpsZfswxrcCLACW2qxFPnse2MDKluzatP70uM9f4DdbI25epwPpvXKXO8T6AMVbSDs5sHCGTCZN3JpAiZBE5PtqA/BBWK6avO9rGRMvszBpS6CFhq/wVf5r8Spo4GUcSvt7DfpDUgz5w2qv+eKP5zpsIjRI+vNmM6AmW6H9cOKCD08bNXkmfIbP4qJE+Ol5uoqikib1rNkLdPK9dtwPSLlUyfpsxT4q7oU1KXZxNU+L3A9ZTMYvuqqv90GN7rcEBan+Tf+XKVWRICWEGL1p4uKL/isUfyHyidKpF+eHRiHiQfRfQhLxqZI5TZ+3ejhpOgvmFbDTvhBjgO08vVDD565HgQojibAPoMxCG93E3wm3DwwREBlRW6jk4JhERQ644YRYpzctFc7rayrw+khUFjjI3cDMASG8df3TAGbKxZdFEcmfnI5igFem9suq3ziL2BYd1f9ZWwf6eN1AnhAROztGt/i0HxR9PiPZ3uyBs7zoGnYW68NsttDOumi8MXNKQFa3qkP4BJzFRwltcjvBg5g44INdepwEiNs2MDTY/7mnEmkNKeA+bdIO6MOk7csZwyaEiZjdpkLhfmloM6f6zR9rU0DcPgbsMr3uEscT9OHEFgd71HjCBUM9iSgsaRbIAqXOAzqIZZParpWRt5oC30LSkuKRIABqQAh+SIrBeK1DF8O6cT8pTCwswgQ5RbsNUXWcOn2CFLCe0MOBQW7QdQn3D/lQ2fdjxudkUkmpTDuU/S3he/88KhF4S94sKjY69gne83qoOeXnga57RHMjQaq7h7zw6aErSO6ru8FIqi9QRnNuFDOGH4U0i8jx3UkGW96hgydd517aKHldZVV/+VFIhJtn4kZ9SagEsGyhJupREt11XenjdWbQSaxLHSf7+bue/hCmE1RKXAH4nEE90A3ZWD28vWdOWp8/P/gXLm97aopiJowyfR/MYosyAKaUfRce9siOePfpjf8i7AU1kOMyeOjqhxuvCbfLB9nAaHpIDmkjtjXOp/BsgqY4T05t576t5y+c0sk6q7yVqDkx59Rech+A0wucVPcHjgsabyUDO08nhZp4ix7gP+zqOLpXKcnN+NwnPqYu0J9+7MpITrHT/A1fKr38DWXVWLhRIfIsji4uy8tk6H0/bAnxjxwZg0onY1G8B0QnAVXmMMpRnGqHf+R0noTY97W5kdlsOqZ4OQ8TPAaaB8odaykAAAA=",
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
    imgs = request.imgs or []
    base64_encoded_images = request.base64EncodeImgs or []
    if len(imgs) == 0 and len(base64_encoded_images) == 0:
        raise HTTPException(
            status_code=400,
            detail="Data validation error: No image found",
        )
    if len(imgs) + len(base64_encoded_images) > 3:
        raise HTTPException(
            status_code=400,
            detail="Data validation error: To many image found. Pass less than three images",
        )

    if imgs:
        base64_encoded_images = [
            encode_image(f"{data_dir}/{img}") for img in request.imgs
        ] + base64_encoded_images

    if not base64_encoded_images:
        raise HTTPException(
            status_code=400,
            detail="Data validation error: No image found",
        )

    image_urls = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_encoded_image}"},
        }
        for base64_encoded_image in base64_encoded_images
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
