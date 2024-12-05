# predict.py
import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from models.unet import UNet
from scipy.ndimage import zoom
from skimage.transform import resize

def get_test_data_paths(test_dir, has_masks=True):
    patient_dirs = [os.path.join(test_dir, d) for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    image_paths = []
    mask_paths = []
    for patient_dir in patient_dirs:
        files_in_patient = os.listdir(patient_dir)
        # Identify image file
        image_file = None
        for fname in [' mDIXON-Quant_BH_v3.nii', ' mDIXON-Quant_BH.nii']:
            if fname in files_in_patient:
                image_file = fname
                break
        if image_file is None:
            print(f"No image file found in {patient_dir}")
            continue
        image_path = os.path.join(patient_dir, image_file)
        image_paths.append(image_path)
        if has_masks:
            mask_file = 'erector.nii'
            mask_path = os.path.join(patient_dir, mask_file)
            if not os.path.exists(mask_path):
                print(f"No mask file found in {patient_dir}")
                continue
            mask_paths.append(mask_path)
    if has_masks:
        return image_paths, mask_paths
    else:
        return image_paths

def load_nifti_image(nifti_path, target_spacing):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    header = img.header
    affine = img.affine

    # Get current voxel spacing
    voxel_spacing = header.get_zooms()  # (X, Y, Z)
    voxel_spacing = (voxel_spacing[2], voxel_spacing[1], voxel_spacing[0])  # Convert to (Z, Y, X)

    data = data.astype(np.float32)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Resample volume to target_spacing
    zoom_factors = (
        voxel_spacing[0] / target_spacing[0],
        voxel_spacing[1] / target_spacing[1],
        voxel_spacing[2] / target_spacing[2]
    )
    data_resampled = zoom(data, zoom_factors, order=1)

    return data_resampled, affine

def load_nifti_mask(nifti_path, target_spacing):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    header = img.header

    # Get current voxel spacing
    voxel_spacing = header.get_zooms()
    voxel_spacing = (voxel_spacing[2], voxel_spacing[1], voxel_spacing[0])  # Convert to (Z, Y, X)

    data = data.astype(np.float32)
    # Binarize mask
    data = (data > 0).astype(np.uint8)

    # Resample volume to target_spacing
    zoom_factors = (
        voxel_spacing[0] / target_spacing[0],
        voxel_spacing[1] / target_spacing[1],
        voxel_spacing[2] / target_spacing[2]
    )
    data_resampled = zoom(data, zoom_factors, order=0)  # Nearest-neighbor interpolation for masks

    return data_resampled

def load_model(checkpoint_path, device):
    model = UNet(n_channels=1, n_classes=1, bilinear=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Model loaded successfully.")
    return model

def predict_volume(model, image_data, device, desired_size=(256, 256), threshold=0.5):
    H, W, D = image_data.shape
    predicted_masks = np.zeros((H, W, D), dtype=np.uint8)

    for i in range(D):
        current_slice = image_data[:, :, i]

        # Resize slice to desired_size
        image_resized = resize(
            current_slice,
            desired_size,
            mode='reflect',
            anti_aliasing=True
        )

        # Add batch and channel dimensions
        image_tensor = torch.from_numpy(image_resized).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(image_tensor)

        probability_map = torch.sigmoid(output).cpu().numpy()[0, 0, :, :]
        predicted_mask_resized = (probability_map > threshold).astype(np.uint8)

        # Resize mask back to original slice size
        predicted_mask_original_size = resize(
            predicted_mask_resized,
            (current_slice.shape[0], current_slice.shape[1]),
            order=0,
            preserve_range=True,
            anti_aliasing=False
        ).astype(np.uint8)

        predicted_masks[:, :, i] = predicted_mask_original_size

        # Print progress
        if (i + 1) % 10 == 0 or (i + 1) == D:
            print(f"Processed slice {i + 1}/{D}")

    return predicted_masks

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # PATHS
    CHECKPOINT_PATH = 'outputs/checkpoints/best_model.pth.tar'  # Update with your checkpoint path
    TEST_PARIS_DIR = 'data/test_paris_data/'
    TEST_BELGIUM_DIR = 'data/test_belgium_data/'
    OUTPUT_DIR = 'outputs/predictions/your_model_name/'  # Replace 'your_model_name' with appropriate name
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define target_spacing and desired_size
    target_spacing = (3.0, 1.7188, 1.7188)  # (Z, Y, X)
    desired_size = (256, 256)

    model = load_model(CHECKPOINT_PATH, device)

    # Get test data paths
    paris_image_paths, paris_mask_paths = get_test_data_paths(TEST_PARIS_DIR, has_masks=True)
    belgium_image_paths, belgium_mask_paths = get_test_data_paths(TEST_BELGIUM_DIR, has_masks=True)

    # Select the 4 test images
    paris_image_paths = paris_image_paths[:4]
    paris_mask_paths = paris_mask_paths[:4]
    belgium_image_paths = belgium_image_paths[:4]
    belgium_mask_paths = belgium_mask_paths[:4]

    # Lists to hold slices for plotting
    paris_slices = []
    belgium_slices = []

    # Process Paris images
    for idx, (image_path, mask_path) in enumerate(zip(paris_image_paths, paris_mask_paths)):
        # Load image data
        image_data, affine = load_nifti_image(image_path, target_spacing)
        # Load mask data
        mask_data = load_nifti_mask(mask_path, target_spacing)
        # Predict mask
        predicted_masks = predict_volume(model, image_data, device, desired_size=desired_size, threshold=0.5)
        # Save predicted mask as NIfTI
        predicted_mask_nifti = nib.Nifti1Image(predicted_masks.astype(np.float32), affine)
        output_mask_path = os.path.join(OUTPUT_DIR, f'paris_patient_{idx+1}_pred_mask.nii.gz')
        nib.save(predicted_mask_nifti, output_mask_path)
        print(f"Predicted mask saved at {output_mask_path}")

        # Get middle slice
        middle_slice_idx = image_data.shape[2] // 2
        image_slice = image_data[:, :, middle_slice_idx]
        mask_slice = mask_data[:, :, middle_slice_idx]
        predicted_mask_slice = predicted_masks[:, :, middle_slice_idx]

        # Collect slices for plotting
        paris_slices.append((image_slice, mask_slice, predicted_mask_slice))

    # Process Belgium images
    for idx, (image_path, mask_path) in enumerate(zip(belgium_image_paths, belgium_mask_paths)):
        # Load image data
        image_data, affine = load_nifti_image(image_path, target_spacing)
        # Load mask data
        mask_data = load_nifti_mask(mask_path, target_spacing)
        # Predict mask
        predicted_masks = predict_volume(model, image_data, device, desired_size=desired_size, threshold=0.5)
        # Save predicted mask as NIfTI
        predicted_mask_nifti = nib.Nifti1Image(predicted_masks.astype(np.float32), affine)
        output_mask_path = os.path.join(OUTPUT_DIR, f'belgium_patient_{idx+1}_pred_mask.nii.gz')
        nib.save(predicted_mask_nifti, output_mask_path)
        print(f"Predicted mask saved at {output_mask_path}")

        # Get middle slice
        middle_slice_idx = image_data.shape[2] // 2
        image_slice = image_data[:, :, middle_slice_idx]
        mask_slice = mask_data[:, :, middle_slice_idx]
        predicted_mask_slice = predicted_masks[:, :, middle_slice_idx]

        # Collect slices for plotting
        belgium_slices.append((image_slice, mask_slice, predicted_mask_slice))

    # Now create plots
    num_paris = len(paris_slices)
    num_belgium = len(belgium_slices)

    # Plot Paris slices
    fig_paris, axes_paris = plt.subplots(num_paris, 4, figsize=(20, 5 * num_paris))
    for idx, (image_slice, mask_slice, predicted_mask_slice) in enumerate(paris_slices):
        # Original image
        axes_paris[idx, 0].imshow(image_slice, cmap='gray', aspect='auto')
        axes_paris[idx, 0].set_title(f'Paris Patient {idx+1} - Image')
        axes_paris[idx, 0].axis('off')
        # Ground truth mask
        axes_paris[idx, 1].imshow(mask_slice, cmap='gray', aspect='auto')
        axes_paris[idx, 1].set_title(f'Paris Patient {idx+1} - Ground Truth Mask')
        axes_paris[idx, 1].axis('off')
        # Predicted mask
        axes_paris[idx, 2].imshow(predicted_mask_slice, cmap='gray', aspect='auto')
        axes_paris[idx, 2].set_title(f'Paris Patient {idx+1} - Predicted Mask')
        axes_paris[idx, 2].axis('off')
        # Overlay
        axes_paris[idx, 3].imshow(image_slice, cmap='gray', aspect='auto')
        # Create an overlay of ground truth and predicted masks
        overlay = np.zeros((*image_slice.shape, 3))
        # Ground truth mask in green
        overlay[mask_slice == 1] = [0, 1, 0]
        # Predicted mask in red
        overlay[predicted_mask_slice == 1] = [1, 0, 0]
        # Where both masks overlap (correct predictions), color in yellow
        overlay[(mask_slice == 1) & (predicted_mask_slice == 1)] = [1, 1, 0]
        axes_paris[idx, 3].imshow(overlay, alpha=0.5)
        axes_paris[idx, 3].set_title(f'Paris Patient {idx+1} - Overlay')
        axes_paris[idx, 3].axis('off')
    plt.tight_layout()
    plot_paris_path = os.path.join(OUTPUT_DIR, 'paris_predictions.png')
    plt.savefig(plot_paris_path)
    print(f"Paris predictions plot saved at {plot_paris_path}")

    # Plot Belgium slices
    fig_belgium, axes_belgium = plt.subplots(num_belgium, 4, figsize=(20, 5 * num_belgium))
    for idx, (image_slice, mask_slice, predicted_mask_slice) in enumerate(belgium_slices):
        # Original image
        axes_belgium[idx, 0].imshow(image_slice, cmap='gray', aspect='auto')
        axes_belgium[idx, 0].set_title(f'Belgium Patient {idx+1} - Image')
        axes_belgium[idx, 0].axis('off')
        # Ground truth mask
        axes_belgium[idx, 1].imshow(mask_slice, cmap='gray', aspect='auto')
        axes_belgium[idx, 1].set_title(f'Belgium Patient {idx+1} - Ground Truth Mask')
        axes_belgium[idx, 1].axis('off')
        # Predicted mask
        axes_belgium[idx, 2].imshow(predicted_mask_slice, cmap='gray', aspect='auto')
        axes_belgium[idx, 2].set_title(f'Belgium Patient {idx+1} - Predicted Mask')
        axes_belgium[idx, 2].axis('off')
        # Overlay
        axes_belgium[idx, 3].imshow(image_slice, cmap='gray', aspect='auto')
        # Create an overlay of ground truth and predicted masks
        overlay = np.zeros((*image_slice.shape, 3))
        # Ground truth mask in green
        overlay[mask_slice == 1] = [0, 1, 0]
        # Predicted mask in red
        overlay[predicted_mask_slice == 1] = [1, 0, 0]
        # Where both masks overlap (correct predictions), color in yellow
        overlay[(mask_slice == 1) & (predicted_mask_slice == 1)] = [1, 1, 0]
        axes_belgium[idx, 3].imshow(overlay, alpha=0.5)
        axes_belgium[idx, 3].set_title(f'Belgium Patient {idx+1} - Overlay')
        axes_belgium[idx, 3].axis('off')
    plt.tight_layout()
    plot_belgium_path = os.path.join(OUTPUT_DIR, 'belgium_predictions.png')
    plt.savefig(plot_belgium_path)
    print(f"Belgium predictions plot saved at {plot_belgium_path}")

if __name__ == '__main__':
    main()
