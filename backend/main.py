from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from starlette.middleware.base import BaseHTTPMiddleware
import os

from core.config import get_settings
from api.api import api_router

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allowing all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for serving result images
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR) 