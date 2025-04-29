import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import datasets.mvtec as mvtec

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/wood_dataset')
    parser.add_argument('--save_path', type=str, default='/content/drive/MyDrive/wood_results')
    parser.add_argument('--arch', type=str, default='resnet18')
    return parser.parse_args(

    )

def main():
    args = parse_args()

    # Model setup
    model = resnet18(pretrained=True, progress=True)
    t_d = 448
    d = 100
    model.to(device)
    model.eval()

    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))

    outputs = []
    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    os.makedirs(os.path.join(args.save_path, f'temp_{args.arch}'), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    for class_name in mvtec.CLASS_NAMES:
        train_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
        test_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        train_feature_filepath = os.path.join(args.save_path, f'temp_{args.arch}', f'train_{class_name}.pkl')
        if not os.path.exists(train_feature_filepath):
            for (x, _, _) in tqdm(train_dataloader, f'| feature extraction | train | {class_name} |'):
                with torch.no_grad():
                    _ = model(x.to(device))
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v.cpu().detach())
                outputs = []
            for k in train_outputs.keys():
                train_outputs[k] = torch.cat(train_outputs[k], 0)

            embedding_vectors = train_outputs['layer1']
            for layer_name in ['layer2', 'layer3']:
                embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])

            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)
            mean = torch.mean(embedding_vectors, dim=0).numpy()
            cov = np.zeros((C, C, H * W))
            I = np.identity(C)
            for i in range(H * W):
                cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
            with open(train_feature_filepath, 'wb') as f:
                torch.save((mean, cov), f)
            train_mean, train_cov = mean, cov    
        else:
            print(f'load train set feature from: {train_feature_filepath}')
            with open(train_feature_filepath, 'rb') as f:
                train_mean, train_cov = torch.load(f)
        gt_list = []
        gt_mask_list = []
        test_imgs = []

        for (x, y, mask) in tqdm(test_dataloader, f'| feature extraction | test | {class_name} |'):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            with torch.no_grad():
                _ = model(x.to(device))
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            outputs = []

        for k in test_outputs.keys():
            test_outputs[k] = torch.cat(test_outputs[k], 0)

        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)

        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        for i in range(H * W):
            mean = train_mean[:, i]
            conv_inv = np.linalg.inv(train_cov[:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)
        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear', align_corners=False).squeeze().numpy()
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        max_score, min_score = score_map.max(), score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)

        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        total_roc_auc.append(img_roc_auc)
        print(f'image ROCAUC: {img_roc_auc:.3f}')
        fig_img_rocauc.plot(fpr, tpr, label=f'{class_name} img_ROCAUC: {img_roc_auc:.3f}')

        gt_mask = np.asarray(gt_mask_list)
        gt_mask = (gt_mask > 0.5).astype(np.uint8)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print(f'pixel ROCAUC: {per_pixel_rocauc:.3f}')

        fig_pixel_rocauc.plot(fpr, tpr, label=f'{class_name} ROCAUC: {per_pixel_rocauc:.3f}')

        # IoU and F1 calculation per image
        from sklearn.metrics import f1_score, jaccard_score
        import pandas as pd

        iou_list = []
        f1_list = []
        true_classes = []

        for i in range(len(gt_mask_list)):
            gt = (gt_mask_list[i].squeeze().flatten() > 0.5).astype(np.uint8)
            pred = scores[i].flatten()
            pred_mask = (pred > threshold).astype(np.uint8)

            if gt.sum() > 0:
                iou = jaccard_score(gt, pred_mask, zero_division=0)
                f1_val = f1_score(gt, pred_mask, zero_division=0)
                true_classes.append("defect")
            else:
                iou = jaccard_score(gt, pred_mask, zero_division=0)
                f1_val = f1_score(gt, pred_mask, zero_division=0)
                true_classes.append("good")

            iou_list.append(iou)
            f1_list.append(f1_val)

        df = pd.DataFrame({
            "Image Index": list(range(len(gt_mask_list))),
            "True Class": true_classes,
            "F1 Score": f1_list,
            "IoU Score": iou_list
        })
        df.to_csv(os.path.join(args.save_path, f'prediction_metrics.csv'), index=False)

        save_dir = os.path.join(args.save_path, f'pictures_{args.arch}')
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)

    print(f'Average ROCAUC: {np.mean(total_roc_auc):.3f}')
    fig_img_rocauc.title.set_text(f'Average image ROCAUC: {np.mean(total_roc_auc):.3f}')
    fig_img_rocauc.legend(loc="lower right")

    print(f'Average pixel ROCUAC: {np.mean(total_pixel_roc_auc):.3f}')
    fig_pixel_rocauc.title.set_text(f'Average pixel ROCAUC: {np.mean(total_pixel_roc_auc):.3f}')
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')

        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')

        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        cb.set_label('Anomaly Score', fontdict={'size': 8})

        fig_img.savefig(os.path.join(save_dir, class_name + f'_{i}.png'), dpi=100)
        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z

if __name__ == '__main__':
    main()

