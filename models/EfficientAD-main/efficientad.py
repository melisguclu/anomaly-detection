import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import random
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
from common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from sklearn.metrics import roc_auc_score, f1_score
from PIL import Image
from sklearn.metrics import precision_score, recall_score



# Teacher normalization function
@torch.no_grad()
def teacher_normalization(teacher, train_loader):
    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco'])
    parser.add_argument('-s', '--subdataset', default='wood')
    parser.add_argument('-o', '--output_dir', default='output/1')
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights', default='models/teacher_small.pth')
    parser.add_argument('-i', '--imagenet_train_path',
                        default='none')
    parser.add_argument('-a', '--mvtec_ad_path',
                        default='./mvtec_anomaly_detection')
    parser.add_argument('-b', '--mvtec_loco_path',
                        default='./mvtec_loco_anomaly_detection')
    parser.add_argument('-t', '--train_steps', type=int, default=70000)
    return parser.parse_args()

# constants
seed = 42
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])

def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))

from skimage.io import imread
from skimage.transform import resize

def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0.0

def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
         desc='Running inference'):
    y_true = []
    y_score = []
    iou_scores = []

    for image, target, path in tqdm(test_set, desc=desc):
        orig_width = image.width
        orig_height = image.height
        image = default_transform(image)
        image = image[None]
        if on_gpu:
            image = image.cuda()
        map_combined, _, _ = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()

        defect_class = os.path.basename(os.path.dirname(path))
        img_nm = os.path.split(path)[1].split('.')[0]
        output_path = os.path.join(test_output_dir, defect_class)
        os.makedirs(output_path, exist_ok=True)
        Image.fromarray((map_combined * 255).astype(np.uint8)).save(os.path.join(output_path, img_nm + '.png'))

        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)

        # IoU hesaplama
        if defect_class == 'defect':
            pred_binary = (map_combined > 0.5).astype(np.uint8)
            gt_path = path.replace("/test/", "/ground_truth/").replace(".jpg", "_mask.jpg")
            if os.path.exists(gt_path):
                gt_mask = imread(gt_path, as_gray=True)
                if gt_mask.shape != pred_binary.shape:
                    gt_mask = resize(gt_mask, pred_binary.shape, order=0, preserve_range=True).astype(np.uint8)
                gt_mask = (gt_mask > 0.5).astype(np.uint8)
                iou = compute_iou(pred_binary, gt_mask)
                iou_scores.append(iou)

    # F1, AUC, IoU metrikleri (threshold = 0.5)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    y_pred = [1 if s > 0.5 else 0 for s in y_score]
    f1 = f1_score(y_true, y_pred)
    avg_iou = np.mean(iou_scores) if iou_scores else 0.0

    print(f"\nFinal Results:")
    print(f"  AUC Score: {auc * 100:.2f}")
    print(f"  F1 Score (threshold=0.5): {f1:.4f}")
    print(f"  IoU Score: {avg_iou:.4f}")

    # 🔍 Threshold optimizasyonu
    best_threshold = 0.0
    best_f1 = 0.0
    for threshold in np.arange(0.0, 1.01, 0.01):
        y_pred_opt = [1 if s > threshold else 0 for s in y_score]
        f1_opt = f1_score(y_true, y_pred_opt)
        if f1_opt > best_f1:
            best_f1 = f1_opt
            best_threshold = threshold

    precision = precision_score(y_true, [1 if s > best_threshold else 0 for s in y_score])
    recall = recall_score(y_true, [1 if s > best_threshold else 0 for s in y_score])

    print(f"\nOptimal Threshold Analysis:")
    print(f"  Best Threshold: {best_threshold:.2f}")
    print(f"  Optimized F1 Score: {best_f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")

    return auc * 100

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization'):
    maps_st = []
    maps_ae = []
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end


@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae

def main():
    args = get_argparse()

    dataset_path = args.mvtec_ad_path if args.dataset == 'mvtec_ad' else args.mvtec_loco_path

    train_output_dir = os.path.join(args.output_dir, 'trainings', args.dataset, args.subdataset)
    test_output_dir = os.path.join(args.output_dir, 'anomaly_maps', args.dataset, args.subdataset, 'test')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, args.subdataset, 'train'),
        transform=transforms.Lambda(train_transform))
    train_size = int(0.9 * len(full_train_set))
    validation_size = len(full_train_set) - train_size
    rng = torch.Generator().manual_seed(seed)
    train_set, validation_set = torch.utils.data.random_split(full_train_set, [train_size, validation_size], rng)

    test_set = ImageFolderWithPath(os.path.join(dataset_path, args.subdataset, 'test'))

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=1)

    if args.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    else:
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)

    teacher.load_state_dict(torch.load(args.weights, map_location='cpu'))
    autoencoder = get_autoencoder(out_channels)

    teacher.eval()
    student.train()
    autoencoder.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

    optimizer = torch.optim.Adam(itertools.chain(student.parameters(), autoencoder.parameters()), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.95 * args.train_steps), gamma=0.1)

    tqdm_obj = tqdm(range(args.train_steps))
    for iteration, (image_st, image_ae) in zip(tqdm_obj, train_loader_infinite):
        if on_gpu:
            image_st = image_st.cuda()
            image_ae = image_ae.cuda()

        with torch.no_grad():
            teacher_output_st = teacher(image_st)
            teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std

        student_output_st = student(image_st)[:, :out_channels]
        distance_st = (teacher_output_st - student_output_st) ** 2
        d_hard = torch.quantile(distance_st, q=0.999)
        loss_hard = torch.mean(distance_st[distance_st >= d_hard])

        ae_output = autoencoder(image_ae)
        with torch.no_grad():
            teacher_output_ae = teacher(image_ae)
            teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std

        student_output_ae = student(image_ae)[:, out_channels:]
        distance_ae = (teacher_output_ae - ae_output)**2
        distance_stae = (ae_output - student_output_ae)**2
        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)

        loss_total = loss_hard + loss_ae + loss_stae

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            tqdm_obj.set_description(f"Loss: {loss_total.item():.4f}")

        if iteration % 1000 == 0:
            torch.save(teacher, os.path.join(train_output_dir, 'teacher_tmp.pth'))
            torch.save(student, os.path.join(train_output_dir, 'student_tmp.pth'))
            torch.save(autoencoder, os.path.join(train_output_dir, 'autoencoder_tmp.pth'))

    teacher.eval()
    student.eval()
    autoencoder.eval()

    torch.save(teacher, os.path.join(train_output_dir, 'teacher_final.pth'))
    torch.save(student, os.path.join(train_output_dir, 'student_final.pth'))
    torch.save(autoencoder, os.path.join(train_output_dir, 'autoencoder_final.pth'))

    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(validation_loader, teacher, student, autoencoder, teacher_mean, teacher_std, desc='Final map normalization')
    test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std, q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=test_output_dir, desc='Final inference')
if __name__ == '__main__':
    main()
