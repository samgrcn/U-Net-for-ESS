import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from models.unet import UNet

def load_nifti_image(nifti_path):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    print(f"Original data shape: {data.shape}")
    data = data.astype(np.float32)
    # Normalize the image
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data, img.affine

def load_model(checkpoint_path, device):
    model = UNet(n_channels=1, n_classes=1, bilinear=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Model loaded successfully.")
    return model


def predict_volume(model, image_data, device, threshold=0.5):
    H, W, D = image_data.shape
    predicted_masks = np.zeros((H, W, D), dtype=np.uint8)

    for i in range(D):
        image_slice = image_data[:, :, i]
        # Rotate if necessary (apply same rotation as during training)
        # image_slice = np.rot90(image_slice, k=-1)

        # Convert to tensor
        image_tensor = torch.from_numpy(image_slice).unsqueeze(0).unsqueeze(0).float().to(device)  # Shape: [1, 1, H, W]

        # Run the model
        with torch.no_grad():
            output = model(image_tensor)

        # Post-process
        probability_map = torch.sigmoid(output).cpu().numpy()[0, 0, :, :]
        predicted_mask = (probability_map > threshold).astype(np.uint8)

        # Store the predicted mask
        predicted_masks[:, :, i] = predicted_mask

    return predicted_masks


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths to the model checkpoint and input image
    CHECKPOINT_PATH = 'outputs/checkpoints/best_model.pth.tar'  # Update if necessary
    IMAGE_PATH = '../data/images/FLARE22_Tr_0001_0000.nii.gz'  # Update this path
    OUTPUT_MASK_PATH = 'preds/pred.nii.gz'  # Update this path

    # Load the model
    model = load_model(CHECKPOINT_PATH, device)

    # Load the image
    image_data, affine = load_nifti_image(IMAGE_PATH)  # image_data shape: (H, W, D)

    # Predict the mask
    predicted_masks = predict_volume(model, image_data, device, threshold=0.5)

    # Save the predicted masks as a NIfTI file
    predicted_mask_nifti = nib.Nifti1Image(predicted_masks.astype(np.float32), affine)
    nib.save(predicted_mask_nifti, OUTPUT_MASK_PATH)
    print(f"Predicted mask saved at {OUTPUT_MASK_PATH}")

    # Optional: Visualize a slice
    slice_index = image_data.shape[2] // 2  # Middle slice
    image_slice = image_data[:, :, slice_index]
    predicted_mask_slice = predicted_masks[:, :, slice_index]

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image_slice, cmap='gray')
    plt.title('Input Image Slice')

    plt.subplot(1, 3, 2)
    plt.imshow(predicted_mask_slice, cmap='gray')
    plt.title('Predicted Mask Slice')

    plt.subplot(1, 3, 3)
    plt.imshow(image_slice, cmap='gray')
    plt.imshow(predicted_mask_slice, cmap='jet', alpha=0.5)
    plt.title('Overlay')

    plt.show()

    if __name__ == '__main__':
        main()