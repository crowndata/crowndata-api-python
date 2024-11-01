from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from crowndata_evaluation.routers.v1.data.information import information_router
from crowndata_evaluation.routers.v1.data_process.information import (
    data_process_information_router,
)
from crowndata_evaluation.routers.v1.data_process.trajectory import trajectory_router
from crowndata_evaluation.routers.v1.evaluation.compare_metric_router import (
    compare_metric_router,
)
from crowndata_evaluation.routers.v1.evaluation.compare_single_to_group_metric_route import (
    compare_single_to_group_router,
)
from crowndata_evaluation.routers.v1.evaluation.group_compare_metric_router import (
    group_compare_metric_router,
)
from crowndata_evaluation.routers.v1.evaluation.group_metric_router import (
    group_metric_router,
)
from crowndata_evaluation.routers.v1.evaluation.metric_router import metric_router
from crowndata_evaluation.routers.v1.kinematics.pos_router import pos_router
from crowndata_evaluation.routers.v1.openai.analyze_image import analyze_image_router

app = FastAPI(
    title="Crowndata Evaluation API",
    description="API for evaluating metrics from Crowndata",
    version="1.0.0",
    docs_url="/docs/api/crowndata-api-python/crowndata-evaluation/",  # Custom path for Swagger UI
    redoc_url="/redoc/api/crowndata-api-python/crowndata-evaluation/",  # Custom path for ReDoc
)

# Allow all origins or specify domains like ['http://localhost:3000'] for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Change "*" to specific domains in production for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom middleware for adding "Access-Control-Allow-Origin" header globally
class CustomCORSHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response: Response = await call_next(request)
        # Add CORS headers to the response
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"

        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response(status_code=204)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "*"

        return response


app.add_middleware(CustomCORSHeaderMiddleware)


app.include_router(metric_router, prefix="/v1/evaluation/metrics", tags=["Evaluation"])
app.include_router(
    compare_metric_router,
    prefix="/v1/evaluation/compare-metrics",
    tags=["Evaluation"],
)
app.include_router(
    group_metric_router,
    prefix="/v1/evaluation/group-metrics",
    tags=["Evaluation"],
)
app.include_router(
    group_compare_metric_router,
    prefix="/v1/evaluation/group-compare-metrics",
    tags=["Evaluation"],
)
app.include_router(
    compare_single_to_group_router,
    prefix="/v1/evaluation/compare-single-to-group-metrics",
    tags=["Evaluation"],
)

app.include_router(
    information_router,
    prefix="/v1/data/information",
    tags=["Data"],
)


app.include_router(
    pos_router,
    prefix="/v1/kinematics/pos",
    tags=["Kinematics"],
)

app.include_router(
    analyze_image_router,
    prefix="/v1/openai/analyze-image",
    tags=["OpenAI"],
)

app.include_router(
    trajectory_router,
    prefix="/v1/data-process/trajectory",
    tags=["Data Process"],
)

app.include_router(
    data_process_information_router,
    prefix="/v1/data-process/information",
    tags=["Data Process"],
)
