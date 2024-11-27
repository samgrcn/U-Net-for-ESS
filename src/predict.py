# predict.py
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
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data, img.affine

def load_model(checkpoint_path, device):
    model = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Model loaded successfully.")
    return model

def predict_volume(model, image_data, device, threshold=0.5):
    H, W, D = image_data.shape
    predicted_masks = np.zeros((H, W, D), dtype=np.uint8)

    for i in range(D):
        # Neighboring slices
        prev_slice = image_data[:, :, max(i - 1, 0)]
        current_slice = image_data[:, :, i]
        next_slice = image_data[:, :, min(i + 1, D - 1)]

        # Stack slices for 3-channel input
        image_stack = np.stack([prev_slice, current_slice, next_slice], axis=0)  # Shape: [3, H, W]

        # Tensor
        image_tensor = torch.from_numpy(image_stack).unsqueeze(0).float().to(device)  # Shape: [1, 3, H, W]

        # Run the model
        with torch.no_grad():
            output = model(image_tensor)

        # Post-process
        probability_map = torch.sigmoid(output).cpu().numpy()[0, 0, :, :]
        predicted_mask = (probability_map > threshold).astype(np.uint8)

        # Store predicted mask
        predicted_masks[:, :, i] = predicted_mask

        # Print progress
        if (i + 1) % 10 == 0 or (i + 1) == D:
            print(f"Processed slice {i + 1}/{D}")

    return predicted_masks

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # PATHS
    CHECKPOINT_PATH = 'outputs/checkpoints/best_model.pth.tar'  # Update if necessary
    IMAGE_PATH = '../data/images_unlabeled/601_water_ax_idiq_bh_inspi_dos_l3.nii'  # Update this path
    OUTPUT_MASK_PATH = 'outputs/preds/pred_belgium2.nii.gz'  # Update this path

    model = load_model(CHECKPOINT_PATH, device)

    image_data, affine = load_nifti_image(IMAGE_PATH)  # image_data shape: (H, W, D)

    predicted_masks = predict_volume(model, image_data, device, threshold=0.5)

    # Save the predicted masks as a NIfTI file
    predicted_mask_nifti = nib.Nifti1Image(predicted_masks.astype(np.float32), affine)
    nib.save(predicted_mask_nifti, OUTPUT_MASK_PATH)
    print(f"Predicted mask saved at {OUTPUT_MASK_PATH}")

    # Visualize middle slice
    slice_index = image_data.shape[2] // 2  # Middle slice
    current_slice = image_data[:, :, slice_index]
    predicted_mask_slice = predicted_masks[:, :, slice_index]

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(current_slice, cmap='gray')
    plt.title('Input Image Slice')

    plt.subplot(1, 3, 2)
    plt.imshow(predicted_mask_slice, cmap='gray')
    plt.title('Predicted Mask Slice')

    plt.subplot(1, 3, 3)
    plt.imshow(current_slice, cmap='gray')
    plt.imshow(predicted_mask_slice, cmap='jet', alpha=0.5)
    plt.title('Overlay')

    plt.show()

if __name__ == '__main__':
    main()
