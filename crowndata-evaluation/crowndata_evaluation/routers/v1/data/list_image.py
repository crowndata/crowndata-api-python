import glob
import os

from fastapi import APIRouter, Query

# Get the EVALUATION_API_ENDPOINT environment variable
data_dir = os.getenv("DATA_DIR", "./public")

data_list_image_router = APIRouter()


def get_images(dataName):
    root = f"{data_dir}"
    exts = ["webp", "jpg", "png"]
    matches = []

    if dataName is not None:
        for ext in exts:
            matches.extend(
                glob.glob(f"{root}/assets/{dataName}/images/*.{ext}", recursive=True)
            )
        results = [os.path.relpath(d, root) for d in matches]
        results.sort()
        return results

    return []


# GET endpoint
@data_list_image_router.get(
    "",
    summary="Image List",
    description="Retrieve Image List for a data",
)
async def get(dataName: str = Query(None, description="Name of the data set")):
    results = get_images(dataName=dataName)

    return {"data": results, "length": len(results)}
