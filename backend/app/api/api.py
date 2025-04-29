from fastapi import APIRouter
from .endpoints import anomaly

api_router = APIRouter()
api_router.include_router(anomaly.router, prefix="/anomaly", tags=["anomaly"]) 