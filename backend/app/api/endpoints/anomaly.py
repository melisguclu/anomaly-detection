from fastapi import APIRouter, UploadFile, File, HTTPException
from ...schemas.anomaly import AnomalyDetectionResult
from ...services.anomaly_detection import AnomalyDetectionService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/detect", response_model=AnomalyDetectionResult)
async def detect_anomaly(file: UploadFile = File(...)) -> AnomalyDetectionResult:
    """
    Detect anomalies in wood surface images using the PaDiM model.
    """
    try:
        contents = await file.read()
        score, result_image = await AnomalyDetectionService.detect_anomaly(contents)
        
        return AnomalyDetectionResult(
            score=score,
            result_image=f"/static/{result_image}"
        )
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 