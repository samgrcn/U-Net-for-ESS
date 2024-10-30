import torch
from torch.utils.data import DataLoader
from models.unet import UNet
from datasets.dataset import MuscleDataset

# Initialize model, loss function, optimizer
model = UNet(n_channels=1, n_classes=1, bilinear=True)
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Prepare data loaders
train_dataset = MuscleDataset(train_images, train_masks, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

val_dataset = MuscleDataset(val_images, val_masks, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

num_epochs = 50

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model.to(device)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Compute average training loss
    avg_train_loss = train_loss / len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save the model checkpoint if needed
    # torch.save(model.state_dict(), 'outputs/models/unet_epoch_{epoch+1}.pth')