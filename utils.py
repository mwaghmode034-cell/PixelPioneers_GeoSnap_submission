import torch
import tifffile as tiff
import numpy as np
from PIL import Image
from torchvision import transforms

rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def preprocess_rgb(image):
    image = image.convert("RGB")
    x = rgb_transform(image)
    return x.unsqueeze(0)


def preprocess_tiff(file, band_mean, band_std):

    img = tiff.imread(file)

    img = img.astype(np.float32)

    img = (img - band_mean[:, None, None]) / (
            band_std[:, None, None] + 1e-8
    )

    x = torch.tensor(img).float()

    return x.unsqueeze(0)