from fastapi import APIRouter
from .endpoints import anomaly, stfpm, efficientad

api_router = APIRouter()
api_router.include_router(anomaly.router, prefix="/anomaly", tags=["anomaly"])
api_router.include_router(stfpm.router, prefix="/stfpm", tags=["stfpm"])
api_router.include_router(efficientad.router, prefix="/efficientad", tags=["efficientad"])
