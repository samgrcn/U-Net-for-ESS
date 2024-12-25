import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Imports from your project (you may adjust relative paths)
# -----------------------------------------------------------------------------
from src.models.unet import UNet3D  # Make sure this is your 3D U-Net with n_channels=2
from src.datasets.dataset import VolumeDataset  # Your dataset returning water, fat, and mask each shape (D,H,W)
from src.utils.utils import dice_coefficient

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 2
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
PIN_MEMORY = True

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATIENT_DIR = '../data/cropped_full_paris_data_cleaned/'  # Update to your folder containing the patients

def get_file_paths_2chan(patient_dir):
    """
    Returns three lists:
       - water_image_paths
       - fat_image_paths
       - mask_paths
    for all patients in 'patient_dir'.

    The function expects:
        - Water image named ' mDIXON-Quant_BH_v3.nii'
                               or ' mDIXON-Quant_BH.nii'
                               or ' mDIXON-Quant_BH.nii.gz'
          (with the leading space).
        - Fat image named 'fat.nii.gz'
        - Mask image named 'erector.nii' or 'erector.nii.gz'
    """
    patient_paths = [
        os.path.join(patient_dir, d)
        for d in os.listdir(patient_dir)
        if os.path.isdir(os.path.join(patient_dir, d))
    ]

    water_paths = []
    fat_paths = []
    mask_paths = []

    for patient_path in patient_paths:
        files_in_patient = os.listdir(patient_path)

        # -- Water
        water_file = None
        for wname in [' mDIXON-Quant_BH_v3.nii',
                      ' mDIXON-Quant_BH.nii',
                      ' mDIXON-Quant_BH.nii.gz']:
            if wname in files_in_patient:
                water_file = wname
                break
        if water_file is None:
            print(f"No water file found in {patient_path}, skipping.")
            continue
        water_path = os.path.join(patient_path, water_file)

        # -- Fat
        fat_file = 'fat.nii.gz'
        if fat_file not in files_in_patient:
            print(f"No fat file found in {patient_path}, skipping.")
            continue
        fat_path = os.path.join(patient_path, fat_file)

        # -- Mask
        mask_file = None
        for mname in ['erector.nii', 'erector.nii.gz']:
            if mname in files_in_patient:
                mask_file = mname
                break
        if mask_file is None:
            print(f"No mask file found in {patient_path}, skipping.")
            continue
        mask_path = os.path.join(patient_path, mask_file)

        water_paths.append(water_path)
        fat_paths.append(fat_path)
        mask_paths.append(mask_path)

    return water_paths, fat_paths, mask_paths

# -----------------------------------------------------------------------------
# Data Preparation
# -----------------------------------------------------------------------------
water_paths, fat_paths, mask_paths = get_file_paths_2chan(PATIENT_DIR)

(
    train_water_paths, val_water_paths,
    train_fat_paths,   val_fat_paths,
    train_mask_paths,  val_mask_paths
) = train_test_split(
    water_paths, fat_paths, mask_paths,
    test_size=0.2, random_state=42
)

target_spacing = (1.75, 1.75, 3.0)
desired_size = (32, 128, 128)  # (D, H, W)

train_dataset = VolumeDataset(
    train_water_paths, train_fat_paths, train_mask_paths,
    desired_size=desired_size,
    target_spacing=target_spacing,
    augment=True
)
val_dataset = VolumeDataset(
    val_water_paths, val_fat_paths, val_mask_paths,
    desired_size=desired_size,
    target_spacing=target_spacing,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

# -----------------------------------------------------------------------------
# Model, Loss, Optimizer, Scheduler
# -----------------------------------------------------------------------------
model = UNet3D(n_channels=2, n_classes=1, bilinear=True).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

CHECKPOINT_DIR = 'outputs/checkpoints/Simple-Unet3D-cropped-augment-99-2channel/'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(state, filepath)
    logging.info(f"Model checkpoint saved at {filepath}")

# -----------------------------------------------------------------------------
# Training & Validation
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, total=len(loader))

    for water, fat, masks in loop:
        # water, fat, masks => shape (B, D, H, W) from VolumeDataset
        water = water.to(device)
        fat   = fat.to(device)
        masks = masks.to(device)

        # Unsqueeze channel dimension => (B,1,D,H,W)
        water = water.unsqueeze(1)
        fat   = fat.unsqueeze(1)
        masks = masks.unsqueeze(1)

        # Combine water+fat => (B,2,D,H,W)
        inputs = torch.cat([water, fat], dim=1)

        outputs = model(inputs)  # (B,1,D,H,W)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * water.size(0)
        loop.set_description("Training")
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def evaluate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    dice_score = 0.0

    with torch.no_grad():
        loop = tqdm(loader, total=len(loader))
        for water, fat, masks in loop:
            water = water.to(device)
            fat   = fat.to(device)
            masks = masks.to(device)

            # Unsqueeze channel dimension
            water = water.unsqueeze(1)
            fat   = fat.unsqueeze(1)
            masks = masks.unsqueeze(1)

            inputs = torch.cat([water, fat], dim=1)
            outputs = model(inputs)

            loss = criterion(outputs, masks)
            val_loss += loss.item() * water.size(0)

            dice = dice_coefficient(outputs, masks)
            dice_score += dice.item() * water.size(0)

            loop.set_description("Validation")
            loop.set_postfix(loss=loss.item(), dice=dice.item())

    avg_loss = val_loss / len(loader.dataset)
    avg_dice = dice_score / len(loader.dataset)
    return avg_loss, avg_dice

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
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

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_dice': val_dice,
            }, filename='best_model.pth.tar')

        # Save checkpoint every 10 epochs
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
