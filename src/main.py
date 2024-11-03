# main.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.unet import UNet
from datasets.dataset import SliceDataset
from utils.utils import dice_coefficient
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Configurations and Hyperparameters
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


BATCH_SIZE = 16  # You can adjust this based on your GPU memory
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
PIN_MEMORY = True

# Paths to your data
IMAGE_DIR = '../data/images/'  # Update this path
MASK_DIR = '../data/masks/'    # Update this path

def get_file_paths(image_dir, mask_dir):
    image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]
    mask_paths = [os.path.join(mask_dir, f) for f in sorted(os.listdir(mask_dir))]
    return image_paths, mask_paths

# Get image and mask paths
image_paths, mask_paths = get_file_paths(IMAGE_DIR, MASK_DIR)

# Split data into training and validation sets
train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

# Create datasets
train_dataset = SliceDataset(train_img_paths, train_mask_paths)
val_dataset = SliceDataset(val_img_paths, val_mask_paths)

# Test data loading
sample_image, sample_mask = train_dataset[5]
print(f"Sample image shape: {sample_image.shape}")
print(f"Sample mask shape: {sample_mask.shape}")

image = sample_image.numpy()[0]  # Remove channel dimension
mask = sample_mask.numpy()[0]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Input Image')
plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title('Ground Truth Mask')
plt.show()

# Create data loaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)


# Initialize model
model = UNet(n_channels=1, n_classes=1, bilinear=True).to(DEVICE)

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# Directory to save model checkpoints
CHECKPOINT_DIR = 'outputs/checkpoints/'
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

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Update progress bar
        loop.set_description(f"Training")
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader)
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
            val_loss += loss.item()

            # Calculate Dice score
            dice = dice_coefficient(outputs, masks)
            dice_score += dice.item()

            # Update progress bar
            loop.set_description(f"Validation")
            loop.set_postfix(loss=loss.item(), dice=dice.item())

    avg_loss = val_loss / len(loader)
    avg_dice = dice_score / len(loader)
    return avg_loss, avg_dice

def main():
    best_val_dice = 0.0

    for epoch in range(NUM_EPOCHS):
        logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")

        # Training
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        logging.info(f"Training Loss: {train_loss:.4f}")

        # Validation
        val_loss, val_dice = evaluate(model, val_loader, criterion, DEVICE)
        logging.info(f"Validation Loss: {val_loss:.4f}, Validation Dice: {val_dice:.4f}")

        # Step the scheduler
        scheduler.step(val_loss)

        # Check if this is the best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_dice': val_dice,
            }, filename='best_model.pth.tar')

        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_dice': val_dice,
            }, filename=f'checkpoint_epoch_{epoch+1}.pth.tar')

if __name__ == '__main__':
    main()