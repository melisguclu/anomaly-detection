from fastapi import APIRouter, UploadFile, File, HTTPException
from schemas.anomaly import AnomalyDetectionResult
from services.efficientad_detection import EfficientADDetectionService
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter()

# Set an environment variable to disable EfficientAD if it causes server crashes
ENABLE_EFFICIENTAD = os.environ.get("ENABLE_EFFICIENTAD", "true").lower() == "true"

try:
    if ENABLE_EFFICIENTAD:
        efficientad_service = EfficientADDetectionService()
        logger.info("EfficientAD service initialized successfully")
    else:
        efficientad_service = None
        logger.info("EfficientAD service disabled via environment variable")
except Exception as e:
    logger.error(f"Failed to initialize EfficientAD service: {str(e)}")
    efficientad_service = None

@router.post("/detect", response_model=AnomalyDetectionResult)
async def detect_anomaly(file: UploadFile = File(...)) -> AnomalyDetectionResult:
    """
    EfficientAD anomaly detection for wood surface images.
    """
    if efficientad_service is None:
        raise HTTPException(
            status_code=503, 
            detail="EfficientAD service is unavailable. Please check server logs for details."
        )
        
    try:
        contents = await file.read()
        score, result_image = await efficientad_service.detect_anomaly(contents)
        return AnomalyDetectionResult(
            score=score,
            result_image=f"/static/{result_image}"
        )
    except Exception as e:
        logger.error(f"EfficientAD API error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
