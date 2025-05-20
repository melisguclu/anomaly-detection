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
import cv2
# Add the models directory to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = str(PROJECT_ROOT / "models")
if MODELS_DIR not in sys.path:
    sys.path.append(MODELS_DIR)

from stfpm.main import ResNet18_MS3, MVTecDataset

logger = logging.getLogger(__name__)

class STFPMDetectionService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Modelleri başlat
        self.teacher = ResNet18_MS3(pretrained=True).to(self.device).eval()
        self.student = ResNet18_MS3(pretrained=False).to(self.device)

        # Model checkpoint dosyasının yolu
        checkpoint_path = PROJECT_ROOT / "models" / "stfpm" / "snapshots" / "wood" / "best.pth.tar"

        # Checkpoint yükle ve student modeline uygula
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.student.load_state_dict(checkpoint['state_dict'])
            self.student.eval()
        else:
            raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

        # Görüntü transform
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    async def detect_anomaly(self, image_contents: bytes) -> tuple[float, str]:
        try:
            # Convert image bytes to PIL Image
            image = Image.open(io.BytesIO(image_contents)).convert('RGB')
            original_size = image.size

            # Transform image
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                teacher_features = self.teacher(image_tensor)
                student_features = self.student(image_tensor)

                score_map = 1.0
                for t_feat, s_feat in zip(teacher_features, student_features):
                    t_feat = F.normalize(t_feat, dim=1)
                    s_feat = F.normalize(s_feat, dim=1)
                    sm = torch.sum((t_feat - s_feat) ** 2, 1, keepdim=True)
                    sm = F.interpolate(sm, size=(64, 64), mode='bilinear', align_corners=False)
                    score_map *= sm

                score = score_map.squeeze().mean().cpu().item()
                score = min(max(score, 0), 1)  # Clamp between 0-1

            # Convert score map to heatmap
            score_map_np = score_map.squeeze().cpu().numpy()
            score_map_norm = (score_map_np - score_map_np.min()) / (score_map_np.max() - score_map_np.min() + 1e-8)
            heatmap_img = (score_map_norm * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, original_size)

            # Original image as numpy
            original_np = np.array(image)
            original_np = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)

            # Overlay (blend)
            overlay = cv2.addWeighted(original_np, 0.7, heatmap, 0.3, 0)

            # Composite: original | heatmap | overlay
            composite = np.concatenate((original_np, heatmap, overlay), axis=1)

            # Save result
            result_image = f"stfpm_result_{hash(image_contents)}.png"
            result_path = PROJECT_ROOT / "backend" / "static" / result_image
            cv2.imwrite(str(result_path), composite)

            return score, result_image

        except Exception as e:
            logger.error(f"Error in STFPM detection service: {str(e)}", exc_info=True)
            raise
