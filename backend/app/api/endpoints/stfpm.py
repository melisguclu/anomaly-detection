from fastapi import APIRouter, UploadFile, File, HTTPException
from ...schemas.anomaly import AnomalyDetectionResult
from ...services.stfpm_detection import STFPMDetectionService
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter()

# Set an environment variable to disable STFPM if it causes server crashes
ENABLE_STFPM = os.environ.get("ENABLE_STFPM", "true").lower() == "true"

try:
    if ENABLE_STFPM:
        stfpm_service = STFPMDetectionService()
        logger.info("STFPM service initialized successfully")
    else:
        stfpm_service = None
        logger.info("STFPM service disabled via environment variable")
except Exception as e:
    logger.error(f"Failed to initialize STFPM service: {str(e)}")
    stfpm_service = None

@router.post("/detect", response_model=AnomalyDetectionResult)
async def detect_anomaly(file: UploadFile = File(...)) -> AnomalyDetectionResult:
    """
    Detect anomalies in wood surface images using the STFPM model.
    """
    if stfpm_service is None:
        raise HTTPException(
            status_code=503, 
            detail="STFPM service is unavailable. Please check server logs for details."
        )
        
    try:
        contents = await file.read()
        score, result_image = await stfpm_service.detect_anomaly(contents)
        
        return AnomalyDetectionResult(
            score=score,
            result_image=f"/static/{result_image}"
        )
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 