import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import io
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import logging
from torch.serialization import add_safe_globals
from torch.nn import Sequential

# Add Sequential to safe globals for model loading
add_safe_globals([Sequential])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "EfficientAD-main"

if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

from common import get_autoencoder, get_pdn_small

logger = logging.getLogger(__name__)

class EfficientADDetectionService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create empty models
        self.teacher = get_pdn_small(384).to(self.device).eval()
        self.student = get_pdn_small(768).to(self.device).eval()
        self.autoencoder = get_autoencoder(384).to(self.device).eval()

        try:
            # Try to load models directly first
            teacher_path = MODEL_DIR / "teacher_final.pth"
            student_path = MODEL_DIR / "student_final.pth"
            autoencoder_path = MODEL_DIR / "autoencoder_final.pth"
            
            # Check if files exist
            if not teacher_path.exists() or not student_path.exists() or not autoencoder_path.exists():
                raise FileNotFoundError(f"Model files not found in {MODEL_DIR}")
                
            # Load models with safer approach
            loaded_teacher = torch.load(teacher_path, map_location=self.device, weights_only=False)
            loaded_student = torch.load(student_path, map_location=self.device, weights_only=False)
            loaded_autoencoder = torch.load(autoencoder_path, map_location=self.device, weights_only=False)
            
            # Check if loaded objects are models or state dictionaries
            if isinstance(loaded_teacher, torch.nn.Module):
                self.teacher = loaded_teacher.to(self.device).eval()
            else:
                self.teacher.load_state_dict(loaded_teacher, strict=False)
                
            if isinstance(loaded_student, torch.nn.Module):
                self.student = loaded_student.to(self.device).eval()
            else:
                self.student.load_state_dict(loaded_student, strict=False)
                
            if isinstance(loaded_autoencoder, torch.nn.Module):
                self.autoencoder = loaded_autoencoder.to(self.device).eval()
            else:
                self.autoencoder.load_state_dict(loaded_autoencoder, strict=False)
                
            logger.info("Successfully loaded EfficientAD models")
            
        except Exception as e:
            logger.error(f"Failed to load EfficientAD models: {e}")
            raise RuntimeError(f"Model dosyaları yüklenemedi: {e}")

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    async def detect_anomaly(self, image_contents: bytes) -> tuple[float, str]:
        try:
            image = Image.open(io.BytesIO(image_contents)).convert("RGB")
            original_size = image.size
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                t_feat = self.teacher(image_tensor)
                s_feat = self.student(image_tensor)
                ae_feat = self.autoencoder(image_tensor)

                map_st = torch.mean((t_feat - s_feat[:, :384]) ** 2, dim=1, keepdim=True)
                map_ae = torch.mean((ae_feat - s_feat[:, 384:]) ** 2, dim=1, keepdim=True)
                anomaly_map = 0.5 * map_st + 0.5 * map_ae
                score = anomaly_map.mean().item()

            # Normalize ve heatmap oluştur
            anomaly_map = F.interpolate(anomaly_map, size=(original_size[1], original_size[0]), mode="bilinear")
            anomaly_map = anomaly_map[0, 0].cpu().numpy()
            heatmap = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
            heatmap_img = np.uint8(heatmap * 255)
            heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)

            orig_np = np.array(image)
            orig_np = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(orig_np, 0.7, heatmap_img, 0.3, 0)

            result_img = np.concatenate((orig_np, heatmap_img, overlay), axis=1)
            result_filename = f"efficientad_result_{hash(image_contents)}.png"
            result_path = PROJECT_ROOT / "backend" / "static" / result_filename
            cv2.imwrite(str(result_path), result_img)

            return score, result_filename

        except Exception as e:
            logger.error(f"EfficientAD hata: {str(e)}", exc_info=True)
            raise
