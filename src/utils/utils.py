import torch

def dice_coefficient(preds, targets, smooth=1e-5):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    intersection = (preds * targets).sum(dim=(1,2,3,4))
    union = preds.sum(dim=(1,2,3,4)) + targets.sum(dim=(1,2,3,4))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()
