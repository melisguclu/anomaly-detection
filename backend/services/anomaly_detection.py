import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import logging
import io

# Add the models directory to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "models" / "padim"))

from inference import run_inference

logger = logging.getLogger(__name__)

class AnomalyDetectionService:
    @staticmethod
    async def detect_anomaly(image_contents: bytes) -> tuple[float, str]:
        try:
            # Convert image bytes to numpy array
            image = Image.open(io.BytesIO(image_contents))
            image = image.convert('RGB')
            image_array = np.array(image) / 255.0  # Normalize to [0, 1]
            
            logger.debug(f"Processing image with shape: {image_array.shape}")
            
            # Run inference using PaDiM model
            score, result_image = run_inference(image_array)
            
            return score, result_image
            
        except Exception as e:
            logger.error(f"Error in anomaly detection service: {str(e)}", exc_info=True)
            raise 