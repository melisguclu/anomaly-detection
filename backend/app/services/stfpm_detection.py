import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import logging
import io
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

# Add the models directory to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = str(PROJECT_ROOT / "models")
if MODELS_DIR not in sys.path:
    sys.path.append(MODELS_DIR)

from stfpm.main import ResNet18_MS3, MVTecDataset

logger = logging.getLogger(__name__)

class STFPMDetectionService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher = ResNet18_MS3(pretrained=True).to(self.device)
        self.student = ResNet18_MS3(pretrained=False).to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    async def detect_anomaly(self, image_contents: bytes) -> tuple[float, str]:
        try:
            # Convert image bytes to PIL Image
            image = Image.open(io.BytesIO(image_contents))
            image = image.convert('RGB')
            
            # Get original image size
            original_size = image.size
            
            # Transform image
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                teacher_features = self.teacher(image_tensor)
                student_features = self.student(image_tensor)
                
                # Calculate anomaly score using multiple feature maps
                score_map = 1.0
                for t_feat, s_feat in zip(teacher_features, student_features):
                    # Normalize features
                    t_feat = F.normalize(t_feat, dim=1)
                    s_feat = F.normalize(s_feat, dim=1)
                    
                    # Calculate similarity map
                    sm = torch.sum((t_feat - s_feat) ** 2, 1, keepdim=True)
                    sm = F.interpolate(sm, size=(64, 64), mode='bilinear', align_corners=False)
                    score_map = score_map * sm
                
                # Get final anomaly score
                score = score_map.squeeze().mean().cpu().item()
                
                # Normalize score to [0, 1] range
                score = min(max(score, 0), 1)
            
            # Save result image with heatmap visualization
            score_map_np = score_map.squeeze().cpu().numpy()
            score_map_np = (score_map_np - score_map_np.min()) / (score_map_np.max() - score_map_np.min() + 1e-8)
            score_map_np = (score_map_np * 255).astype(np.uint8)
            
            # Convert to heatmap
            import cv2
            heatmap = cv2.applyColorMap(score_map_np, cv2.COLORMAP_JET)
            
            # Resize heatmap to match original image size
            heatmap = cv2.resize(heatmap, original_size)
            
            # Convert original image to numpy array and ensure correct format
            original_np = np.array(image)
            original_np = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
            
            # Ensure both images have the same size and number of channels
            if original_np.shape != heatmap.shape:
                logger.warning(f"Size mismatch: original={original_np.shape}, heatmap={heatmap.shape}")
                heatmap = cv2.resize(heatmap, (original_np.shape[1], original_np.shape[0]))
            
            # Blend images
            blended = cv2.addWeighted(original_np, 0.7, heatmap, 0.3, 0)
            
            # Save result
            result_image = f"stfpm_result_{hash(image_contents)}.png"
            result_path = PROJECT_ROOT / "backend" / "static" / result_image
            cv2.imwrite(str(result_path), blended)
            
            return score, result_image
            
        except Exception as e:
            logger.error(f"Error in STFPM detection service: {str(e)}", exc_info=True)
            raise 