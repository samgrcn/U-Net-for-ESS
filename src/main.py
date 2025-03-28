import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.unet import UNet3D
from src.datasets.dataset import VolumeDataset
from src.utils.utils import dice_coefficient
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 1
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
PIN_MEMORY = True

# PATHS
PATIENT_DIR = '../data/cropped_paris_data/'

def get_file_paths(patient_dir):
    patient_paths = [os.path.join(patient_dir, d) for d in os.listdir(patient_dir) if
                     os.path.isdir(os.path.join(patient_dir, d))]
    image_paths = []
    mask_paths = []
    for patient_path in patient_paths:
        files_in_patient = os.listdir(patient_path)
        image_file = None
        for fname in [' mDIXON-Quant_BH_v3.nii', ' mDIXON-Quant_BH.nii', ' mDIXON-Quant_BH.nii.gz']:
            if fname in files_in_patient:
                image_file = fname
                break
        if image_file is None:
            print(f"No image file found in {patient_path}")
            continue

        image_path = os.path.join(patient_path, image_file)

        mask_file = None
        for mname in ['erector.nii', 'erector.nii.gz']:
            if mname in files_in_patient:
                mask_file = mname
                break
        if mask_file is None:
            print(f"No mask file found in {patient_path}")
            continue

        mask_path = os.path.join(patient_path, mask_file)
        image_paths.append(image_path)
        mask_paths.append(mask_path)
    return image_paths, mask_paths

# Get image and mask paths
image_paths, mask_paths = get_file_paths(PATIENT_DIR)

# Split data
train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

target_spacing = (1.75, 1.75, 3.0)
desired_size = (32, 128, 128)

# Datasets
train_dataset = VolumeDataset(train_img_paths, train_mask_paths, desired_size=desired_size, target_spacing=target_spacing, augment=True)
val_dataset = VolumeDataset(val_img_paths, val_mask_paths, desired_size=desired_size, target_spacing=target_spacing, augment=False)

# Data loaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)

# MODEL
model = UNet3D(n_channels=1, n_classes=1, bilinear=True).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

CHECKPOINT_DIR = 'outputs/checkpoints/Simple-Unet3D-cropped-augment/'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(state, filepath)
    logging.info(f"Model checkpoint saved at {filepath}")

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, total=len(loader))
    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        loop.set_description(f"Training")
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def evaluate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    dice_score = 0.0
    with torch.no_grad():
        loop = tqdm(loader, total=len(loader))
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)

            dice = dice_coefficient(outputs, masks)
            dice_score += dice.item() * images.size(0)

            loop.set_description(f"Validation")
            loop.set_postfix(loss=loss.item(), dice=dice.item())

    avg_loss = val_loss / len(loader.dataset)
    avg_dice = dice_score / len(loader.dataset)
    return avg_loss, avg_dice

def main():
    best_val_dice = 0.0
    train_losses = []
    val_losses = []
    val_dices = []

    for epoch in range(NUM_EPOCHS):
        logging.info(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        logging.info(f"Training Loss: {train_loss:.4f}")
        train_losses.append(train_loss)

        val_loss, val_dice = evaluate(model, val_loader, criterion, DEVICE)
        logging.info(f"Validation Loss: {val_loss:.4f}, Validation Dice: {val_dice:.4f}")
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        scheduler.step(val_loss)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_dice': val_dice,
            }, filename='best_model.pth.tar')

        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_dice': val_dice,
            }, filename=f'checkpoint_epoch_{epoch + 1}.pth.tar')

    # Plotting
    epochs = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_dices, 'g', label='Validation Dice')
    plt.title('Validation Dice over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(CHECKPOINT_DIR, 'training_plot.png')
    plt.savefig(plot_path)
    logging.info(f"Training plot saved at {plot_path}")
    plt.show()

if __name__ == '__main__':
    main()
