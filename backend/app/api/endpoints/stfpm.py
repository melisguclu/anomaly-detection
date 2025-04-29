from fastapi import APIRouter, UploadFile, File, HTTPException
from ...schemas.anomaly import AnomalyDetectionResult
from ...services.stfpm_detection import STFPMDetectionService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
stfpm_service = STFPMDetectionService()

@router.post("/detect", response_model=AnomalyDetectionResult)
async def detect_anomaly(file: UploadFile = File(...)) -> AnomalyDetectionResult:
    """
    Detect anomalies in wood surface images using the STFPM model.
    """
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