from pydantic import BaseModel

class AnomalyDetectionResult(BaseModel):
    score: float
    result_image: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "score": 0.75,
                "result_image": "/static/abc123_result.png"
            }
        } 