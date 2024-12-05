import numpy as np
import torch

def dice_coefficient(preds, targets, smooth=1e-5):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()

def normalize(img: np.ndarray) -> np.ndarray:
    """Percentile-based normalization of the gray levels."""
    img = img.astype("float32") / np.percentile(img, 99)
    img[img > 1.0] = 0.975
    return img