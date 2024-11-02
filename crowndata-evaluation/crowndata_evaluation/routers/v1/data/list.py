import glob
import os

from fastapi import APIRouter

# Get the EVALUATION_API_ENDPOINT environment variable
data_dir = os.getenv("DATA_DIR", "./public")

data_list_router = APIRouter()


# POST endpoint for evaluating metrics
@data_list_router.get(
    "",
    summary="Data List",
    description="Retrieve Data List",
)
async def get():
    root = f"{data_dir}/assets"
    matches = glob.glob(f"{root}/*", recursive=True)
    one_level_directories = [
        os.path.relpath(d, root) for d in matches if os.path.isdir(d)
    ]
    one_level_directories.sort()

    return {"data": one_level_directories, "length": len(one_level_directories)}
