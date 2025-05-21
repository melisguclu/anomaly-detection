import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys


sys.path.append('/content/anomaly-detection/models')

from stfpm.main import ResNet18_MS3
from stfpm.evaluate import roc

# === Dataset loader ===
class SimpleTestDataset(torch.utils.data.Dataset):
    def __init__(self, defect_dir, good_dir):
        self.data = []

        for p in Path(defect_dir).rglob("*"):
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                self.data.append((p, 1))  # 1 = defect

        for p in Path(good_dir).rglob("*"):
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                self.data.append((p, 0))  # 0 = good

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), path.name, label

# === Görsel sonuçları kaydeden fonksiyon ===
def save_heatmap(img_tensor, heatmap, filename):
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap).resize((256, 256)).convert("L")
    overlay = Image.blend(
        Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)),
        heatmap_img.convert("RGB"),
        alpha=0.5
    )
    overlay.save(filename)


def main():
    print("🚀 STFPM test script started.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ckpt_path = Path("/content/anomaly-detection/models/stfpm/snapshots/wood/best.pth.tar")
    test_defect = Path("/content/drive/MyDrive/wood_dataset/wood/test/defect")
    test_good   = Path("/content/drive/MyDrive/wood_dataset/wood/test/good")
    output_dir  = Path("/content/drive/MyDrive/anomaly-results/stfpm")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Test klasörleri:\n - defect: {test_defect}\n - good:   {test_good}")
    print(f"📁 Çıktı klasörü: {output_dir}")

    teacher = ResNet18_MS3(pretrained=True).to(device).eval()
    student = ResNet18_MS3(pretrained=False).to(device).eval()
    checkpoint = torch.load(ckpt_path, map_location=device)
    student.load_state_dict(checkpoint["state_dict"])
    print("✅ Model checkpoint yüklendi.")

    dataset = SimpleTestDataset(test_defect, test_good)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"🖼️ Toplam test görseli: {len(dataset)}")

    anomaly_scores = []
    image_labels = []
    processed = 0

    for img, name, label in dataloader:
        img = img.to(device)

        with torch.no_grad():
            feat_s = student(img)
            feat_t = teacher(img)

            score_maps = []
            target_size = None

            for idx, (f_s, f_t) in enumerate(zip(feat_s, feat_t)):
                s = F.mse_loss(f_s, f_t, reduction='none').mean(dim=1, keepdim=True)

                if idx == 0:
                    target_size = s.shape[-2:]

                if s.shape[-2:] != target_size:
                    s = F.interpolate(s, size=target_size, mode='bilinear', align_corners=False)

                score_maps.append(s)

            score_map = torch.mean(torch.cat(score_maps, dim=1), dim=1).squeeze().cpu().numpy()
            score = np.max(score_map)

            anomaly_scores.append(score)
            image_labels.append(int(label.item()))

            save_path = output_dir / f"overlay_{name[0]}"
            save_heatmap(img.squeeze().cpu(), score_map, save_path)

            print(f"✔ İşlendi: {name[0]}  →  Label: {label.item()}  →  Score: {score:.4f}")
            processed += 1

    fpr, tpr, auc_score = roc(image_labels, anomaly_scores)
    print(f"\n📈 AUC Score: {auc_score:.4f}")
    print(f"✅ Toplam işlendi: {processed}")
    print(f"📁 Sonuçlar kaydedildi: {output_dir}")

if __name__ == "__main__":
    main()
