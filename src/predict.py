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
        image_file = None
        for fname in [' mDIXON-Quant_BH_v3.nii', ' mDIXON-Quant_BH.nii', ' mDIXON-Quant_BH.nii.gz']:
            if fname in files_in_patient:
                image_file = fname
                break
        if image_file is None:
            print(f"No image file found in {patient_dir}")
            continue
        image_path = os.path.join(patient_dir, image_file)
        image_paths.append(image_path)

        if has_masks:
            mask_file = None
            for mname in ['erector.nii', 'erector.nii.gz']:
                if mname in files_in_patient:
                    mask_file = mname
                    break
            if mask_file is None:
                print(f"No mask file found in {patient_dir}")
                continue
            mask_path = os.path.join(patient_dir, mask_file)
            mask_paths.append(mask_path)

    if has_masks:
        return image_paths, mask_paths
    else:
        return image_paths


def load_nifti_image(nifti_path, target_spacing):
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.float32)
    header = img.header
    affine = img.affine

    voxel_spacing = header.get_zooms()  # (X_spacing, Y_spacing, Z_spacing)
    p975 = np.percentile(data, 99)
    data = np.clip(data, 0, p975) / p975

    zoom_factors = (
        voxel_spacing[0] / target_spacing[0],
        voxel_spacing[1] / target_spacing[1],
        voxel_spacing[2] / target_spacing[2]
    )
    data_resampled = zoom(data, zoom_factors, order=1)

    return data_resampled, affine


def load_nifti_mask(nifti_path, target_spacing):
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.float32)
    header = img.header

    voxel_spacing = header.get_zooms()
    data = (data > 0).astype(np.uint8)

    zoom_factors = (
        voxel_spacing[0] / target_spacing[0],
        voxel_spacing[1] / target_spacing[1],
        voxel_spacing[2] / target_spacing[2]
    )
    data_resampled = zoom(data, zoom_factors, order=0)

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

        image_resized = resize(
            current_slice,
            desired_size,
            mode='reflect',
            anti_aliasing=True
        )

        image_tensor = torch.from_numpy(image_resized).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(image_tensor)

        probability_map = torch.sigmoid(output).cpu().numpy()[0, 0, :, :]
        predicted_mask_resized = (probability_map > threshold).astype(np.uint8)

        predicted_mask_original_size = resize(
            predicted_mask_resized,
            (current_slice.shape[0], current_slice.shape[1]),
            order=0,
            preserve_range=True,
            anti_aliasing=False
        ).astype(np.uint8)

        predicted_masks[:, :, i] = predicted_mask_original_size

    return predicted_masks


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use the chosen best spacing as in training
    target_spacing = (1.75, 1.75, 3.0)
    desired_size = (256, 256)

    CHECKPOINT_PATH = 'outputs/checkpoints/Simple-Unet-voxel-full-99-fine/best_model.pth.tar'
    TEST_PARIS_DIR = '../data/test_paris_data/'
    TEST_BELGIUM_DIR = '../data/test_belgium_data/'
    OUTPUT_DIR = 'outputs/predictions/Simple-Unet-voxel-full-99-fine'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model(CHECKPOINT_PATH, device)

    paris_image_paths, paris_mask_paths = get_test_data_paths(TEST_PARIS_DIR, has_masks=True)
    belgium_image_paths, belgium_mask_paths = get_test_data_paths(TEST_BELGIUM_DIR, has_masks=True)

    # Just an example of choosing first few images
    paris_image_paths = paris_image_paths[:4]
    paris_mask_paths = paris_mask_paths[:4]
    belgium_image_paths = belgium_image_paths[:4]
    belgium_mask_paths = belgium_mask_paths[:4]

    # Process Paris images
    paris_slices = []
    for idx, (image_path, mask_path) in enumerate(zip(paris_image_paths, paris_mask_paths)):
        image_data, affine = load_nifti_image(image_path, target_spacing)
        mask_data = load_nifti_mask(mask_path, target_spacing)
        predicted_masks = predict_volume(model, image_data, device, desired_size=desired_size, threshold=0.5)

        predicted_mask_nifti = nib.Nifti1Image(predicted_masks.astype(np.float32), affine)
        output_mask_path = os.path.join(OUTPUT_DIR, f'paris_patient_{idx + 1}_pred_mask.nii.gz')
        nib.save(predicted_mask_nifti, output_mask_path)

        middle_slice_idx = image_data.shape[2] // 2
        image_slice = image_data[:, :, middle_slice_idx]
        mask_slice = mask_data[:, :, middle_slice_idx]
        predicted_mask_slice = predicted_masks[:, :, middle_slice_idx]
        paris_slices.append((image_slice, mask_slice, predicted_mask_slice))

    # Process Belgium images
    belgium_slices = []
    for idx, (image_path, mask_path) in enumerate(zip(belgium_image_paths, belgium_mask_paths)):
        image_data, affine = load_nifti_image(image_path, target_spacing)
        mask_data = load_nifti_mask(mask_path, target_spacing)
        predicted_masks = predict_volume(model, image_data, device, desired_size=desired_size, threshold=0.5)

        predicted_mask_nifti = nib.Nifti1Image(predicted_masks.astype(np.float32), affine)
        output_mask_path = os.path.join(OUTPUT_DIR, f'belgium_patient_{idx + 1}_pred_mask.nii.gz')
        nib.save(predicted_mask_nifti, output_mask_path)

        middle_slice_idx = image_data.shape[2] // 2
        image_slice = image_data[:, :, middle_slice_idx]
        mask_slice = mask_data[:, :, middle_slice_idx]
        predicted_mask_slice = predicted_masks[:, :, middle_slice_idx]
        belgium_slices.append((image_slice, mask_slice, predicted_mask_slice))

    num_paris = len(paris_slices)
    num_belgium = len(belgium_slices)

    # Plot Paris slices
    fig_paris, axes_paris = plt.subplots(num_paris, 4, figsize=(20, 5 * num_paris))
    for idx, (image_slice, mask_slice, predicted_mask_slice) in enumerate(paris_slices):
        # Original image
        axes_paris[idx, 0].imshow(image_slice, cmap='gray', aspect='auto')
        axes_paris[idx, 0].set_title(f'Paris Patient {idx + 1} - Image')
        axes_paris[idx, 0].axis('off')
        # Ground truth mask
        axes_paris[idx, 1].imshow(mask_slice, cmap='gray', aspect='auto')
        axes_paris[idx, 1].set_title(f'Paris Patient {idx + 1} - Ground Truth Mask')
        axes_paris[idx, 1].axis('off')
        # Predicted mask
        axes_paris[idx, 2].imshow(predicted_mask_slice, cmap='gray', aspect='auto')
        axes_paris[idx, 2].set_title(f'Paris Patient {idx + 1} - Predicted Mask')
        axes_paris[idx, 2].axis('off')
        # Overlay
        axes_paris[idx, 3].imshow(image_slice, cmap='gray', aspect='auto')
        # Create an overlay of ground truth and predicted masks
        overlay = np.zeros((*image_slice.shape, 3))
        # Overlap (both masks) in green
        overlay[(mask_slice == 1) & (predicted_mask_slice == 1)] = [0, 1, 0]
        # Ground truth mask only in blue
        overlay[(mask_slice == 1) & (predicted_mask_slice == 0)] = [0, 0, 1]
        # Predicted mask only in red
        overlay[(mask_slice == 0) & (predicted_mask_slice == 1)] = [1, 0, 0]
        axes_paris[idx, 3].imshow(overlay, alpha=0.22, aspect='auto')
        axes_paris[idx, 3].set_title(f'Paris Patient {idx + 1} - Overlay')
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
        axes_belgium[idx, 0].set_title(f'Belgium Patient {idx + 1} - Image')
        axes_belgium[idx, 0].axis('off')
        # Ground truth mask
        axes_belgium[idx, 1].imshow(mask_slice, cmap='gray', aspect='auto')
        axes_belgium[idx, 1].set_title(f'Belgium Patient {idx + 1} - Ground Truth Mask')
        axes_belgium[idx, 1].axis('off')
        # Predicted mask
        axes_belgium[idx, 2].imshow(predicted_mask_slice, cmap='gray', aspect='auto')
        axes_belgium[idx, 2].set_title(f'Belgium Patient {idx + 1} - Predicted Mask')
        axes_belgium[idx, 2].axis('off')
        # Overlay
        axes_belgium[idx, 3].imshow(image_slice, cmap='gray', aspect='auto')
        # Create an overlay of ground truth and predicted masks
        overlay = np.zeros((*image_slice.shape, 3))
        # Overlap (both masks) in green
        overlay[(mask_slice == 1) & (predicted_mask_slice == 1)] = [0, 1, 0]
        # Ground truth mask only in blue
        overlay[(mask_slice == 1) & (predicted_mask_slice == 0)] = [0, 0, 1]
        # Predicted mask only in red
        overlay[(mask_slice == 0) & (predicted_mask_slice == 1)] = [1, 0, 0]
        axes_belgium[idx, 3].imshow(overlay, alpha=0.22, aspect='auto')
        axes_belgium[idx, 3].set_title(f'Belgium Patient {idx + 1} - Overlay')
        axes_belgium[idx, 3].axis('off')
    plt.tight_layout()
    plot_belgium_path = os.path.join(OUTPUT_DIR, 'belgium_predictions.png')
    plt.savefig(plot_belgium_path)
    print(f"Belgium predictions plot saved at {plot_belgium_path}")


if __name__ == '__main__':
    main()
