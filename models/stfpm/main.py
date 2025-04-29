import argparse
import os
import cv2
import numpy as np
import pandas as pd

import torch
import torch.optim
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from .evaluate import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from glob import glob
from PIL import Image
from matplotlib import pyplot as plt
from .evaluate import roc as compute_roc


class MVTecDataset(object):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.ToTensor()
        self.dataset = self.load_dataset()

    def load_dataset(self):
        return [Image.open(p).convert('RGB') for p in self.image_list]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.transform(self.dataset[idx])
        return self.image_list[idx], image


class ResNet18_MS3(nn.Module):

    def __init__(self, pretrained=False):
        super(ResNet18_MS3, self).__init__()
        net = models.resnet18(pretrained=pretrained)
        self.model = torch.nn.Sequential(*(list(net.children())[:-2]))

    def forward(self, x):
        res = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in ['4', '5', '6']:
                res.append(x)
        return res


def load_gt(root, cls, image_paths):
    gt = []
    for img_path in image_paths:
        fname = os.path.basename(img_path)  # örn: 100000000.jpg
        defect_type = os.path.basename(os.path.dirname(img_path))  # örn: defect
        mask_name = fname.replace(".jpg", "_mask.jpg")
        gt_path = os.path.join(root, cls, "ground_truth", defect_type, mask_name)
        
        if not os.path.exists(gt_path):
            print(f"Uyarı: Mask bulunamadı: {gt_path}")
            temp = np.zeros((256, 256), dtype=np.bool_)
        else:
            temp = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if temp is None:
                print(f"Uyarı: Mask okunamadı: {gt_path}")
                temp = np.zeros((256, 256), dtype=np.bool_)
            else:
                temp = (cv2.resize(temp, (256, 256)) > 30).astype(np.bool_)  # 30 yerine 10-50 arası deneyebilirsin
        
        gt.append(temp[None, ...])
    
    return np.concatenate(gt, 0)






def calculate_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description="Anomaly Detection")
    parser.add_argument("split", nargs="?", choices=["train", "test"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--category", type=str , default='leather')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoint-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--result-path", type=str, default='results')
    parser.add_argument("--save-fig", action='store_true')
    parser.add_argument("--mvtec-ad", type=str, default='mvtec_anomaly_detection')
    parser.add_argument('--model-save-path', type=str, default='snapshots')

    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.split == 'train':
        image_list = sorted(glob(os.path.join(args.mvtec_ad, args.category, 'train', 'good', '*.jpg')))
        train_image_list, val_image_list = train_test_split(image_list, test_size=0.2, random_state=0)
        train_dataset = MVTecDataset(train_image_list, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_dataset = MVTecDataset(val_image_list, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    elif args.split == 'test':
        test_neg_image_list = sorted(glob(os.path.join(args.mvtec_ad, args.category, 'test', 'good', '*.jpg')))
        test_pos_image_list = sorted(list(set(glob(os.path.join(args.mvtec_ad, args.category, 'test', '*', '*.jpg'))) - set(test_neg_image_list)))
        test_neg_dataset = MVTecDataset(test_neg_image_list, transform=transform)
        test_pos_dataset = MVTecDataset(test_pos_image_list, transform=transform)
        test_neg_loader = DataLoader(test_neg_dataset, batch_size=1, shuffle=False, drop_last=False)
        test_pos_loader = DataLoader(test_pos_dataset, batch_size=1, shuffle=False, drop_last=False)

    teacher = ResNet18_MS3(pretrained=True)
    student = ResNet18_MS3(pretrained=False)
    teacher.to(device)
    student.to(device)

    if args.split == 'train':
        train_val(teacher, student, train_loader, val_loader, args, device)
    elif args.split == 'test':
        saved_dict = torch.load(args.checkpoint)
        category = args.category
        gt = load_gt(args.mvtec_ad, category, test_pos_image_list)


        print('load ' + args.checkpoint)
        student.load_state_dict(saved_dict['state_dict'])

        pos = test(teacher, student, test_pos_loader, device)
        neg = test(teacher, student, test_neg_loader, device)

        scores = []
        for i in range(len(pos)):
            temp = cv2.resize(pos[i], (256, 256))
            scores.append(temp)
        for i in range(len(neg)):
            temp = cv2.resize(neg[i], (256, 256))
            scores.append(temp)

        scores = np.stack(scores)
        neg_gt = np.zeros((len(neg), 256, 256), dtype=np.bool_)
        gt_pixel = np.concatenate((gt, neg_gt), 0)
        gt_image = np.concatenate((np.ones(pos.shape[0], dtype=np.bool_), np.zeros(neg.shape[0], dtype=np.bool_)), 0)

        pro = evaluate(gt_pixel, scores, metric='pro')
        auc_pixel = evaluate(gt_pixel.flatten(), scores.flatten(), metric='roc')
        auc_image_max = evaluate(gt_image, scores.max(-1).max(-1), metric='roc')
        print(f'Catergory: {category}\tPixel-AUC: {auc_pixel:.6f}\tImage-AUC: {auc_image_max:.6f}\tPRO: {pro:.6f}')

        fpr_pixel, tpr_pixel, auc_pixel_val = compute_roc(gt_pixel.flatten(), scores.flatten())
        fpr_image, tpr_image, auc_image_val = compute_roc(gt_image, scores.max(-1).max(-1))

        roc_dir = os.path.join(args.result_path, args.category)
        os.makedirs(roc_dir, exist_ok=True)

        # Pixel-level ROC
        plt.figure()
        plt.plot(fpr_pixel, tpr_pixel, label=f'Pixel AUC = {auc_pixel_val:.4f}')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Pixel-level ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(roc_dir, "pixel_roc_curve.png"))
        plt.close()

        # Image-level ROC
        plt.figure()
        plt.plot(fpr_image, tpr_image, label=f'Image AUC = {auc_image_val:.4f}')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Image-level ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(roc_dir, "image_roc_curve.png"))
        plt.close()

        print("ROC eğrileri kaydedildi.")

        # F1 Score & IoU
        threshold = 0.5
        pred_binary = scores > threshold
        gt_binary = gt_pixel.astype(bool)

        f1 = f1_score(gt_binary.flatten(), pred_binary.flatten())
        print(f"F1 Score (threshold={threshold}): {f1:.4f}")

        ious = [calculate_iou(pred_binary[i], gt_binary[i]) for i in range(len(pred_binary))]
        mean_iou = np.mean(ious)
        print(f"Mean IoU: {mean_iou:.4f}")

        # CSV output
        image_names = test_pos_image_list + test_neg_image_list
        image_names = [os.path.basename(p) for p in image_names]

        df = pd.DataFrame({
            "Image": image_names,
            "F1 Score": [f1_score(gt_binary[i].flatten(), pred_binary[i].flatten()) for i in range(len(pred_binary))],
            "IoU": ious,
            "Anomaly Score (max)": [s.max() for s in scores]
        })

        os.makedirs(os.path.join(args.result_path, args.category), exist_ok=True)
        df.to_csv(os.path.join(args.result_path, args.category, "per_image_report.csv"), index=False)
        print("CSV dosyası oluşturuldu.")

        # Görsel çıktıları kaydet (overlay dahil)
        if args.save_fig:
            from matplotlib import pyplot as plt

            save_dir = os.path.join(args.result_path, args.category, "vis")
            os.makedirs(save_dir, exist_ok=True)

            for i, score_map in enumerate(scores):
                vis = (score_map - score_map.min()) / (score_map.max() - score_map.min()) * 255
                vis = vis.astype(np.uint8)
                vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

                filename = os.path.basename(image_names[i]).replace('.jpg', '_score.jpg')
                cv2.imwrite(os.path.join(save_dir, filename), vis_color)

                # Orijinal görüntüyü yükle
                if i < len(test_pos_image_list):
                    img_path = test_pos_image_list[i]
                else:
                    img_path = test_neg_image_list[i - len(test_pos_image_list)]

                original_img = cv2.imread(img_path)
                original_img = cv2.resize(original_img, (256, 256))

                # Overlay: score map ile orijinal görseli birleştir
                overlayed = cv2.addWeighted(original_img, 0.6, vis_color, 0.4, 0)
                overlay_filename = os.path.basename(image_names[i]).replace('.jpg', '_overlay.jpg')
                cv2.imwrite(os.path.join(save_dir, overlay_filename), overlayed)

                # Ground truth mask
                if i < len(gt):
                    gt_mask = gt[i][0].astype(np.uint8) * 255
                    gt_mask = cv2.resize(gt_mask, (256, 256))
                    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
                else:
                    gt_mask = np.zeros((256, 256, 3), dtype=np.uint8)

                # Composite görsel
                composite = np.concatenate((original_img, vis_color, overlayed, gt_mask), axis=1)
                composite_filename = os.path.basename(image_names[i]).replace('.jpg', '_composite.jpg')
                cv2.imwrite(os.path.join(save_dir, composite_filename), composite)

            print(f"Görseller kaydedildi (overlay dahil): {save_dir}")




def test(teacher, student, loader, device):
    teacher.eval()
    student.eval()
    loss_map = np.zeros((len(loader.dataset), 64, 64))
    i = 0
    for batch_data in loader:
        _, batch_img = batch_data
        batch_img = batch_img.to(device)
        with torch.no_grad():
            t_feat = teacher(batch_img)
            s_feat = student(batch_img)
        score_map = 1.
        for j in range(len(t_feat)):
            t_feat[j] = F.normalize(t_feat[j], dim=1)
            s_feat[j] = F.normalize(s_feat[j], dim=1)
            sm = torch.sum((t_feat[j] - s_feat[j]) ** 2, 1, keepdim=True)
            sm = F.interpolate(sm, size=(64, 64), mode='bilinear', align_corners=False)
            score_map = score_map * sm
        loss_map[i: i + batch_img.size(0)] = score_map.squeeze().cpu().data.numpy()
        i += batch_img.size(0)
    return loss_map


def train_val(teacher, student, train_loader, val_loader, args, device):
    min_err = 10000
    teacher.eval()
    student.train()

    optimizer = torch.optim.SGD(student.parameters(), 0.4, momentum=0.9, weight_decay=1e-4)
    for epoch in range(args.epochs):
        student.train()
        for batch_data in train_loader:
            _, batch_img = batch_data
            batch_img = batch_img.to(device)

            with torch.no_grad():
                t_feat = teacher(batch_img)
            s_feat = student(batch_img)

            loss = 0
            for i in range(len(t_feat)):
                t_feat[i] = F.normalize(t_feat[i], dim=1)
                s_feat[i] = F.normalize(s_feat[i], dim=1)
                loss += torch.sum((t_feat[i] - s_feat[i]) ** 2, 1).mean()

            print('[%d/%d] loss: %f' % (epoch, args.epochs, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        err = test(teacher, student, val_loader, device).mean()
        print('Valid Loss: {:.7f}'.format(err.item()))
        if err < min_err:
            min_err = err
            save_name = os.path.join(args.model_save_path, args.category, 'best.pth.tar')
            dir_name = os.path.dirname(save_name)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            state_dict = {
                'category': args.category,
                'state_dict': student.state_dict()
            }
            torch.save(state_dict, save_name)


if __name__ == "__main__":
    main()
