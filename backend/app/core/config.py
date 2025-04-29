from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from pathlib import Path

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Wood Anomaly Detection API"
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:5173",  # Vite default
        "http://localhost:3000",  # React default
        "http://localhost:8000",  # Backend default
    ]
    
    # Static files
    STATIC_DIR: str = str(Path(__file__).resolve().parent.parent.parent / "static")
    
    class Config:
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings() 