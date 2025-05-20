import os
import urllib.request
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis
from PIL import Image
from skimage.transform import resize
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# model ve hook
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device).eval()

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

# sabit parametreler (ResNet50)
t_d, d = 1664, 550
idx = torch.randperm(t_d)[:d].clone().detach()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

file_path = os.path.join(os.path.dirname(__file__), "train_wood.pkl")

if not os.path.exists(file_path):
    logger.info("Downloading pretrained model from Hugging Face...")
    url = "https://huggingface.co/melisgcl/padim/resolve/main/train_wood.pkl"
    urllib.request.urlretrieve(url, file_path)
    logger.info("Download completed.")

with open(file_path, "rb") as f:
    loaded = torch.load(f, weights_only=False)
    if len(loaded) == 3:
        train_mean, train_cov, threshold = loaded
    else:
        train_mean, train_cov = loaded
        threshold = 0.5

    logger.info(f"Loaded train_mean shape: {train_mean.shape}")
    logger.info(f"Loaded train_cov shape: {train_cov.shape}")

def run_inference(image_array: np.ndarray) -> tuple[float, str]:
    try:
        logger.debug(f"Input image shape: {image_array.shape}")

        # Resize image to 256x256 and center crop to 224x224
        if image_array.shape[:2] != (256, 256):
            image_array = resize(image_array, (256, 256), anti_aliasing=True, preserve_range=True)
        image = Image.fromarray((image_array * 255).astype(np.uint8))
        width, height = image.size
        image = image.crop(((width - 224) // 2, (height - 224) // 2, (width + 224) // 2, (height + 224) // 2))

        image_tensor = normalize(to_tensor(image)).unsqueeze(0).to(device)

        outputs = []
        def hook(module, input, output):
            outputs.append(output)

        hooks = [
            model.layer1[-1].register_forward_hook(hook),
            model.layer2[-1].register_forward_hook(hook),
            model.layer3[-1].register_forward_hook(hook),
        ]

        with torch.no_grad():
            _ = model(image_tensor)

        embedding = outputs[0]
        for out in outputs[1:]:
            embedding = embedding_concat(embedding, out)
        for h in hooks:
            h.remove()

        if embedding.size(1) > t_d:
            embedding = embedding[:, :t_d, :, :]

        embedding = torch.index_select(embedding, 1, idx)
        B, C, H, W = embedding.size()
        embedding = embedding.view(B, C, -1).cpu().numpy()

        if embedding.shape[2] != train_mean.shape[1]:
            logger.warning("Embedding dimension mismatch with train mean")
            embedding = resize(embedding[0].T, (train_mean.shape[1], train_mean.shape[0])).T
            embedding = embedding.reshape(1, *embedding.shape)

        dist_list = []
        for i in range(embedding.shape[2]):
            mean = train_mean[:, i]
            cov_inv = np.linalg.inv(train_cov[:, :, i])
            dist = mahalanobis(embedding[0][:, i], mean, cov_inv)
            dist_list.append(dist)

        feature_map_size = int(np.sqrt(embedding.shape[2]))
        dist_map = np.array(dist_list).reshape(feature_map_size, feature_map_size)
        dist_map = gaussian_filter(dist_map, sigma=4)
        score = dist_map.max()
        score_norm = (dist_map - dist_map.min()) / (dist_map.max() - dist_map.min())

        mask = (score_norm > threshold).astype(np.uint8)
        original_shape = image_array.shape[:2]
        if mask.shape != original_shape:
            mask = resize(mask, original_shape, order=0, preserve_range=True).astype(np.uint8)
            score_norm = resize(score_norm, original_shape, order=1, preserve_range=True)

        vis = mark_boundaries(image_array, mask, color=(1, 0, 0))
        file_id = str(uuid.uuid4())[:8]
        static_dir = os.path.join(os.path.dirname(__file__), "../../backend/static")
        os.makedirs(static_dir, exist_ok=True)
        save_path = os.path.join(static_dir, f"{file_id}_result.png")

        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].imshow(image_array); ax[0].set_title("Original"); ax[0].axis("off")
        ax[1].imshow(score_norm, cmap="jet"); ax[1].set_title("Anomaly Heatmap"); ax[1].axis("off")
        ax[2].imshow(mask, cmap="gray"); ax[2].set_title("Predicted Mask"); ax[2].axis("off")
        ax[3].imshow(vis); ax[3].set_title("Segmentation Result"); ax[3].axis("off")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        return float(score), os.path.basename(save_path)

    except Exception as e:
        logger.error(f"Error in run_inference: {str(e)}", exc_info=True)
        raise