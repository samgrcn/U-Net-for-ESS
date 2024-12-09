import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from models.unet import UNet
from scipy.ndimage import zoom
from skimage.transform import resize
import glob

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_test_data_paths(test_dir, dataset_type, has_masks=True):
    """
    Retrieves paths to water, fat fraction, and mask images based on the dataset type.

    Parameters:
    - test_dir (str): Directory containing the test data subfolders.
    - dataset_type (str): Type of dataset ('paris' or 'belgium').
    - has_masks (bool): Whether the dataset includes mask files.

    Returns:
    - Tuple of lists containing paths to water images, fat fraction images, and masks (if applicable).
    """
    patient_dirs = [os.path.join(test_dir, d) for d in os.listdir(test_dir)
                    if os.path.isdir(os.path.join(test_dir, d))]
    water_paths = []
    fat_paths = []
    mask_paths = []

    for patient_dir in patient_dirs:
        files_in_patient = os.listdir(patient_dir)

        if dataset_type.lower() == 'paris':
            water_file = ' mDIXON-Quant_BH.nii.gz'  # Leading space
            fat_file = ' mDIXON-Quant_BH_fat.nii.gz'  # Leading space
        elif dataset_type.lower() == 'belgium':
            water_file = ' mDIXON-Quant_BH_v3.nii'  # Leading space
            fat_file_pattern = '*fatfrac*.nii*'  # Pattern to match fat fraction files
            mask_file = 'erector.nii.gz'
        else:
            raise ValueError("Unsupported dataset type. Choose 'paris' or 'belgium'.")

        # Locate water file
        if dataset_type.lower() == 'paris':
            if water_file not in files_in_patient:
                print(f"Missing water image ({water_file}) in {patient_dir}")
                continue
            water_path = os.path.join(patient_dir, water_file)

            # Locate fat fraction file
            if fat_file not in files_in_patient:
                print(f"Missing fat fraction image ({fat_file}) in {patient_dir}")
                continue
            fat_path = os.path.join(patient_dir, fat_file)

        elif dataset_type.lower() == 'belgium':
            if water_file not in files_in_patient:
                print(f"Missing water image ({water_file}) in {patient_dir}")
                continue
            water_path = os.path.join(patient_dir, water_file)

            # Use glob to find fat fraction files matching the pattern
            fat_files = glob.glob(os.path.join(patient_dir, fat_file_pattern))
            if not fat_files:
                print(f"Missing fat fraction image matching pattern '{fat_file_pattern}' in {patient_dir}")
                continue
            # Assuming there's only one fat fraction file per patient directory
            fat_path = fat_files[0]

        if has_masks:
            # For Belgium dataset only
            if dataset_type.lower() == 'belgium':
                if mask_file not in files_in_patient:
                    print(f"Missing mask file ({mask_file}) in {patient_dir}")
                    continue
                mask_path = os.path.join(patient_dir, mask_file)
            else:
                # For datasets with masks but not Belgium, define mask_file accordingly
                # Currently, only Belgium has masks
                mask_path = None

            if dataset_type.lower() == 'belgium':
                water_paths.append(water_path)
                fat_paths.append(fat_path)
                mask_paths.append(mask_path)
        else:
            water_paths.append(water_path)
            fat_paths.append(fat_path)

    if has_masks and dataset_type.lower() == 'belgium':
        return water_paths, fat_paths, mask_paths
    else:
        return water_paths, fat_paths


def load_nifti_image(nifti_path, target_spacing):
    """
    Loads a NIfTI image, normalizes it, and resamples to the target spacing.

    Parameters:
    - nifti_path (str): Path to the NIfTI file.
    - target_spacing (tuple): Desired voxel spacing (Z, Y, X).

    Returns:
    - Tuple containing the resampled image data and the affine matrix.
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()
    header = img.header

    voxel_spacing = header.get_zooms()  # (X, Y, Z)
    voxel_spacing = (voxel_spacing[2], voxel_spacing[1], voxel_spacing[0])  # (Z, Y, X)

    data = data.astype(np.float32)
    # Normalize to [0, 1]
    data = (data - np.min(data)) / (np.ptp(data))

    # Calculate zoom factors for resampling
    zoom_factors = (
        voxel_spacing[0] / target_spacing[0],
        voxel_spacing[1] / target_spacing[1],
        voxel_spacing[2] / target_spacing[2]
    )
    data_resampled = zoom(data, zoom_factors, order=1)
    return data_resampled, img.affine


def load_nifti_mask(nifti_path, target_spacing):
    """
    Loads a NIfTI mask, binarizes it, and resamples to the target spacing.

    Parameters:
    - nifti_path (str): Path to the NIfTI mask file.
    - target_spacing (tuple): Desired voxel spacing (Z, Y, X).

    Returns:
    - Resampled binary mask data.
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()
    header = img.header

    voxel_spacing = header.get_zooms()
    voxel_spacing = (voxel_spacing[2], voxel_spacing[1], voxel_spacing[0])  # (Z, Y, X)

    data = data.astype(np.float32)
    # Binarize mask
    data = (data > 0).astype(np.uint8)

    zoom_factors = (
        voxel_spacing[0] / target_spacing[0],
        voxel_spacing[1] / target_spacing[1],
        voxel_spacing[2] / target_spacing[2]
    )
    data_resampled = zoom(data, zoom_factors, order=0)
    return data_resampled


def load_model(checkpoint_path, device):
    """
    Loads the trained U-Net model from the checkpoint.

    Parameters:
    - checkpoint_path (str): Path to the model checkpoint.
    - device (torch.device): Device to load the model on.

    Returns:
    - Loaded U-Net model in evaluation mode.
    """
    model = UNet(n_channels=2, n_classes=1, bilinear=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Model loaded successfully.")
    return model


def predict_volume(model, water_data, fat_data, device, desired_size=(256, 256), threshold=0.5):
    """
    Predicts the segmentation mask for a 3D volume slice-by-slice.

    Parameters:
    - model (torch.nn.Module): Trained U-Net model.
    - water_data (numpy.ndarray): 3D water image data.
    - fat_data (numpy.ndarray): 3D fat fraction image data.
    - device (torch.device): Device to perform computations on.
    - desired_size (tuple): Desired 2D slice size (H, W) for the model.
    - threshold (float): Threshold for converting probability maps to binary masks.

    Returns:
    - Predicted binary mask for the entire 3D volume.
    """
    H, W, D = water_data.shape
    predicted_masks = np.zeros((H, W, D), dtype=np.uint8)

    for i in range(D):
        water_slice = water_data[:, :, i]
        fat_slice = fat_data[:, :, i]

        # Resize slices to desired size
        water_resized = resize(water_slice, desired_size, mode='reflect', anti_aliasing=True)
        fat_resized = resize(fat_slice, desired_size, mode='reflect', anti_aliasing=True)

        # Stack channels and create tensor
        image_tensor = np.stack([water_resized, fat_resized], axis=0)  # (2, H, W)
        image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).float().to(device)  # (1, 2, H, W)

        with torch.no_grad():
            output = model(image_tensor)

        # Apply sigmoid and threshold
        probability_map = torch.sigmoid(output).cpu().numpy()[0, 0, :, :]
        predicted_mask_resized = (probability_map > threshold).astype(np.uint8)

        # Resize back to original slice size
        predicted_mask_original_size = resize(
            predicted_mask_resized,
            (H, W),
            order=0,
            preserve_range=True,
            anti_aliasing=False
        ).astype(np.uint8)

        predicted_masks[:, :, i] = predicted_mask_original_size

        # Progress update
        if (i + 1) % 10 == 0 or (i + 1) == D:
            print(f"Processed slice {i + 1}/{D}")

    return predicted_masks


def plot_predictions(slices, dataset_type, output_dir, prefix):
    """
    Plots the image slices, ground truth masks (if available), predicted masks, and overlays.

    Parameters:
    - slices (list of tuples): Each tuple contains (image_slice, mask_slice or None, predicted_mask_slice).
    - dataset_type (str): Type of dataset ('paris' or 'belgium').
    - output_dir (str): Directory to save the plots.
    - prefix (str): Prefix for the plot filename.
    """
    num_samples = len(slices)
    if num_samples == 0:
        print(f"No slices to plot for {dataset_type} dataset.")
        return

    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for idx, (image_slice, mask_slice, predicted_mask_slice) in enumerate(slices):
        # Image
        axes[idx, 0].imshow(image_slice, cmap='gray', aspect='auto')
        axes[idx, 0].set_title(f'{dataset_type.capitalize()} Patient {idx + 1} - Image')
        axes[idx, 0].axis('off')

        if dataset_type.lower() == 'belgium':
            # Ground Truth Mask
            axes[idx, 1].imshow(mask_slice, cmap='gray', aspect='auto')
            axes[idx, 1].set_title(f'{dataset_type.capitalize()} Patient {idx + 1} - Ground Truth Mask')
            axes[idx, 1].axis('off')

            # Predicted Mask
            axes[idx, 2].imshow(predicted_mask_slice, cmap='gray', aspect='auto')
            axes[idx, 2].set_title(f'{dataset_type.capitalize()} Patient {idx + 1} - Predicted Mask')
            axes[idx, 2].axis('off')

            # Overlay
            axes[idx, 3].imshow(image_slice, cmap='gray', aspect='auto')
            overlay = np.zeros((*image_slice.shape, 3))
            # True Positives (Green)
            overlay[(mask_slice == 1) & (predicted_mask_slice == 1)] = [0, 1, 0]
            # False Negatives (Ground Truth Only - Blue)
            overlay[(mask_slice == 1) & (predicted_mask_slice == 0)] = [0, 0, 1]
            # False Positives (Prediction Only - Red)
            overlay[(mask_slice == 0) & (predicted_mask_slice == 1)] = [1, 0, 0]
            axes[idx, 3].imshow(overlay, alpha=0.2, aspect='auto')
            axes[idx, 3].set_title(f'{dataset_type.capitalize()} Patient {idx + 1} - Overlay')
            axes[idx, 3].axis('off')
        else:
            # For Paris dataset (no masks)
            # Ground Truth Mask (All Black)
            black_mask = np.zeros_like(image_slice)
            axes[idx, 1].imshow(black_mask, cmap='gray', aspect='auto')
            axes[idx, 1].set_title(f'{dataset_type.capitalize()} Patient {idx + 1} - Ground Truth Mask')
            axes[idx, 1].axis('off')

            # Predicted Mask
            axes[idx, 2].imshow(predicted_mask_slice, cmap='gray', aspect='auto')
            axes[idx, 2].set_title(f'{dataset_type.capitalize()} Patient {idx + 1} - Predicted Mask')
            axes[idx, 2].axis('off')

            # Overlay (only prediction)
            axes[idx, 3].imshow(image_slice, cmap='gray', aspect='auto')
            overlay = np.zeros((*image_slice.shape, 3))
            # Prediction Only (Red)
            overlay[predicted_mask_slice == 1] = [1, 0, 0]
            axes[idx, 3].imshow(overlay, alpha=0.2, aspect='auto')
            axes[idx, 3].set_title(f'{dataset_type.capitalize()} Patient {idx + 1} - Overlay')
            axes[idx, 3].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{prefix}_predictions.png')
    plt.savefig(plot_path)
    print(f"{dataset_type.capitalize()} predictions plot saved at {plot_path}")
    plt.close(fig)


def main():
    # Define paths (update these paths as necessary)
    CHECKPOINT_PATH = 'outputs/checkpoints/Simple-Unet-voxel-full-fat/best_model.pth.tar'
    TEST_PARIS_DIR = '../data/test_full_paris_data'
    TEST_BELGIUM_DIR = '../data/test_belgium_data/'
    OUTPUT_DIR = 'outputs/predictions/Simple-Unet-voxel-full-fat/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define target spacing and desired slice size
    target_spacing = (3.0, 1.7188, 1.7188)  # (Z, Y, X)
    desired_size = (256, 256)

    # Load the trained model
    model = load_model(CHECKPOINT_PATH, device)

    # ==========================
    # Process Paris Dataset
    # ==========================
    print("\nProcessing Paris dataset...")
    paris_water_paths, paris_fat_paths = get_test_data_paths(
        TEST_PARIS_DIR, dataset_type='paris', has_masks=False)

    # Select a subset of samples (e.g., first 4 patients)
    num_paris_samples = 4
    paris_water_paths = paris_water_paths[:num_paris_samples]
    paris_fat_paths = paris_fat_paths[:num_paris_samples]

    paris_slices = []

    for idx, (w_path, f_path) in enumerate(zip(paris_water_paths, paris_fat_paths)):
        print(f"\nProcessing Paris patient {idx + 1}...")
        water_data, affine = load_nifti_image(w_path, target_spacing)
        fat_data, _ = load_nifti_image(f_path, target_spacing)

        # Predict masks (no ground truth masks)
        predicted_masks = predict_volume(model, water_data, fat_data, device, desired_size=desired_size, threshold=0.5)

        # Save predicted mask
        output_mask_path = os.path.join(OUTPUT_DIR, f'paris_patient_{idx + 1}_pred_mask.nii.gz')
        nib.save(nib.Nifti1Image(predicted_masks.astype(np.float32), affine), output_mask_path)
        print(f"Predicted mask saved at {output_mask_path}")

        # Select middle slice for visualization
        middle_slice_idx = water_data.shape[2] // 2
        image_slice = water_data[:, :, middle_slice_idx]  # Use water channel for visualization
        predicted_mask_slice = predicted_masks[:, :, middle_slice_idx]
        paris_slices.append((image_slice, None, predicted_mask_slice))  # No mask_slice

    # Plot Paris predictions
    plot_predictions(paris_slices, dataset_type='paris', output_dir=OUTPUT_DIR, prefix='paris')

    # ==========================
    # Process Belgium Dataset
    # ==========================
    print("\nProcessing Belgium dataset...")
    belgium_water_paths, belgium_fat_paths, belgium_mask_paths = get_test_data_paths(
        TEST_BELGIUM_DIR, dataset_type='belgium', has_masks=True)

    # Select a subset of samples (e.g., first 4 patients)
    num_belgium_samples = 4
    belgium_water_paths = belgium_water_paths[:num_belgium_samples]
    belgium_fat_paths = belgium_fat_paths[:num_belgium_samples]
    belgium_mask_paths = belgium_mask_paths[:num_belgium_samples]

    belgium_slices = []

    for idx, (w_path, f_path, m_path) in enumerate(zip(belgium_water_paths, belgium_fat_paths, belgium_mask_paths)):
        print(f"\nProcessing Belgium patient {idx + 1}...")
        water_data, affine = load_nifti_image(w_path, target_spacing)
        fat_data, _ = load_nifti_image(f_path, target_spacing)
        mask_data = load_nifti_mask(m_path, target_spacing)

        # Predict masks
        predicted_masks = predict_volume(model, water_data, fat_data, device, desired_size=desired_size, threshold=0.5)

        # Save predicted mask
        output_mask_path = os.path.join(OUTPUT_DIR, f'belgium_patient_{idx + 1}_pred_mask.nii.gz')
        nib.save(nib.Nifti1Image(predicted_masks.astype(np.float32), affine), output_mask_path)
        print(f"Predicted mask saved at {output_mask_path}")

        # Select middle slice for visualization
        middle_slice_idx = water_data.shape[2] // 2
        image_slice = water_data[:, :, middle_slice_idx]
        mask_slice = mask_data[:, :, middle_slice_idx]
        predicted_mask_slice = predicted_masks[:, :, middle_slice_idx]
        belgium_slices.append((image_slice, mask_slice, predicted_mask_slice))

    # Plot Belgium predictions
    plot_predictions(belgium_slices, dataset_type='belgium', output_dir=OUTPUT_DIR, prefix='belgium')

    print("\nAll predictions completed and saved.")


if __name__ == '__main__':
    main()
