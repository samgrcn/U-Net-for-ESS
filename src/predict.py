import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from src.models.unet import UNet3D
from scipy.ndimage import zoom
from skimage.transform import resize

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_test_data_paths_2chan(test_dir, has_masks=True):
    """
    Search for water image, fat image, and optionally a mask
    in each patient folder within test_dir.

    Returns:
      - water_paths: list of water-image paths
      - fat_paths:   list of fat-image paths
      - mask_paths:  list of mask paths (if has_masks=True)
    """
    patient_dirs = [
        os.path.join(test_dir, d)
        for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    ]

    water_paths = []
    fat_paths = []
    mask_paths = []  # optional

    for pdir in patient_dirs:
        files_in_patient = os.listdir(pdir)

        # ---------------------
        # Find water image
        # ---------------------
        water_file = None
        for wname in [' mDIXON-Quant_BH_v3.nii',
                      ' mDIXON-Quant_BH.nii',
                      ' mDIXON-Quant_BH.nii.gz']:
            if wname in files_in_patient:
                water_file = wname
                break
        if water_file is None:
            print(f"No water image found in {pdir}, skipping.")
            continue
        wpath = os.path.join(pdir, water_file)

        # ---------------------
        # Find fat image
        # ---------------------
        fat_file = None
        for fname in files_in_patient:
            # Check if 'fat' in the file name (case-insensitive) and ends with .nii or .nii.gz
            if 'fat' in fname.lower() and (fname.endswith('.nii') or fname.endswith('.nii.gz')):
                fat_file = fname
                break
        if fat_file is None:
            print(f"No fat file found in {pdir}, skipping.")
            continue
        fpath = os.path.join(pdir, fat_file)

        # ---------------------
        # Find mask (optional)
        # ---------------------
        mpath = None
        if has_masks:
            for mname in ['erector.nii', 'erector.nii.gz']:
                if mname in files_in_patient:
                    mpath = os.path.join(pdir, mname)
                    break
            if mpath is None:
                print(f"No mask found in {pdir}, skipping.")
                continue

        water_paths.append(wpath)
        fat_paths.append(fpath)
        if has_masks:
            mask_paths.append(mpath)

    if has_masks:
        return water_paths, fat_paths, mask_paths
    else:
        return water_paths, fat_paths


def load_nifti_2chan(water_path, fat_path, target_spacing, desired_size):
    """
    Loads and resamples water + fat volumes, returning two separate
    (D,H,W) numpy arrays, plus other metadata for restoring geometry.
    """
    # ---------------------
    # WATER
    # ---------------------
    w_img = nib.load(water_path)
    w_data = w_img.get_fdata().astype(np.float32)
    w_affine = w_img.affine
    w_header = w_img.header
    w_spacing = w_header.get_zooms()

    # Basic normalization
    w_p99 = np.percentile(w_data, 99)
    w_data = np.clip(w_data, 0, w_p99) / (w_p99 + 1e-8)

    # Resample
    w_zoom_factors = (
        w_spacing[0] / target_spacing[0],
        w_spacing[1] / target_spacing[1],
        w_spacing[2] / target_spacing[2]
    )
    w_data_resampled = zoom(w_data, w_zoom_factors, order=1)
    # (X',Y',Z') -> (Z',X',Y')
    w_data_resampled = np.transpose(w_data_resampled, (2, 0, 1))

    # Resize
    w_data_resized = resize(
        w_data_resampled,
        desired_size,
        mode='reflect',
        anti_aliasing=True
    )

    # ---------------------
    # FAT
    # ---------------------
    f_img = nib.load(fat_path)
    f_data = f_img.get_fdata().astype(np.float32)
    f_header = f_img.header
    f_spacing = f_header.get_zooms()

    f_p99 = np.percentile(f_data, 99)
    f_data = np.clip(f_data, 0, f_p99) / (f_p99 + 1e-8)

    f_zoom_factors = (
        f_spacing[0] / target_spacing[0],
        f_spacing[1] / target_spacing[1],
        f_spacing[2] / target_spacing[2]
    )
    f_data_resampled = zoom(f_data, f_zoom_factors, order=1)
    f_data_resampled = np.transpose(f_data_resampled, (2, 0, 1))

    f_data_resized = resize(
        f_data_resampled,
        desired_size,
        mode='reflect',
        anti_aliasing=True
    )

    # We'll return both volumes plus metadata.
    return (
        w_data_resized,
        f_data_resized,
        w_affine,
        w_data.shape,  # original shape of WATER
        w_spacing,  # original spacing
        w_data_resampled.shape  # resampled shape of WATER
    )


def load_nifti_mask(mask_path, target_spacing, desired_size):
    img = nib.load(mask_path)
    data = img.get_fdata().astype(np.float32)
    data = (data > 0).astype(np.float32)
    header = img.header
    voxel_spacing = header.get_zooms()

    zoom_factors = (
        voxel_spacing[0] / target_spacing[0],
        voxel_spacing[1] / target_spacing[1],
        voxel_spacing[2] / target_spacing[2]
    )
    mask_resampled = zoom(data, zoom_factors, order=0)
    mask_resampled = np.transpose(mask_resampled, (2, 0, 1))

    mask_resized = resize(
        mask_resampled,
        desired_size,
        order=0,
        preserve_range=True,
        anti_aliasing=False
    ).astype(np.float32)

    return mask_resized


def restore_original_geometry(pred_mask, desired_size, resampled_shape, original_shape, voxel_spacing, target_spacing,
                              affine):
    """
    Same as your old function:
    pred_mask: (D,H,W) = (Z',X',Y')
    Up-sample back to original shape/orientation.
    """
    # Step 1: resize back from desired_size -> resampled_shape
    pred_mask_resampled = resize(
        pred_mask,
        (resampled_shape[2], resampled_shape[0], resampled_shape[1]),
        order=0,
        preserve_range=True,
        anti_aliasing=False
    ).astype(np.uint8)

    # (Z',X',Y') -> (X',Y',Z')
    pred_mask_resampled = np.transpose(pred_mask_resampled, (1, 2, 0))

    # Step 2: Zoom back to original spacing
    inverse_zoom_factors = (
        voxel_spacing[0] / target_spacing[0],
        voxel_spacing[1] / target_spacing[1],
        voxel_spacing[2] / target_spacing[2]
    )
    pred_mask_original = zoom(pred_mask_resampled, inverse_zoom_factors, order=0).astype(np.uint8)

    # Step 3: ensure exact original_shape
    pred_mask_original = resize(
        pred_mask_original,
        original_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False
    ).astype(np.uint8)

    pred_nifti = nib.Nifti1Image(pred_mask_original.astype(np.float32), affine)
    return pred_nifti


def load_model(checkpoint_path, device):
    model = UNet3D(n_channels=2, n_classes=1, bilinear=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model


def predict_volume_2chan(model, water_data, fat_data, device, threshold=0.5):
    """
    Takes 2 separate volumes (water_data, fat_data) each shape (D,H,W).
    Stacks them to shape (1,2,D,H,W) for inference.
    Returns predicted_mask shape (D,H,W).
    """
    with torch.no_grad():
        # shape -> (1,D,H,W) each
        water_tensor = torch.from_numpy(water_data).unsqueeze(0).float()
        fat_tensor = torch.from_numpy(fat_data).unsqueeze(0).float()

        # Combine into 2-channel: (1, 2, D, H, W)
        input_2ch = torch.stack([water_tensor, fat_tensor], dim=1).to(device)

        output = model(input_2ch)  # (1,1,D,H,W)
        prob = torch.sigmoid(output).cpu().numpy()[0, 0]
        predicted_mask = (prob > threshold).astype(np.uint8)
    return predicted_mask


def main():
    """
    Example usage:
    - We load the best model checkpoint.
    - We have 4 Paris patients + 4 Belgium patients (for demonstration).
    - We predict and save the results + plot them.
    """
    CHECKPOINT_PATH = 'outputs/checkpoints/Simple-Unet3D-cropped-augment-99-2channel/best_model.pth.tar'
    TEST_PARIS_DIR = '../data/cropped_test_full_paris_data/'
    TEST_BELGIUM_DIR = '../data/cropped_test_belgium_data/'
    OUTPUT_DIR = 'outputs/predictions/Simple-Unet3D-cropped-augment-99-2channel/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target_spacing = (1.75, 1.75, 3.0)
    desired_size = (32, 128, 128)

    model = load_model(CHECKPOINT_PATH, DEVICE)

    # Get Paris test data
    paris_water_paths, paris_fat_paths, paris_mask_paths = get_test_data_paths_2chan(TEST_PARIS_DIR, has_masks=True)
    # Get Belgium test data
    belgium_water_paths, belgium_fat_paths, belgium_mask_paths = get_test_data_paths_2chan(TEST_BELGIUM_DIR,
                                                                                           has_masks=True)

    # Take first 4 of each for plotting
    paris_water_paths = paris_water_paths[:4]
    paris_fat_paths = paris_fat_paths[:4]
    paris_mask_paths = paris_mask_paths[:4]

    belgium_water_paths = belgium_water_paths[:4]
    belgium_fat_paths = belgium_fat_paths[:4]
    belgium_mask_paths = belgium_mask_paths[:4]

    paris_slices = []
    for i, (w_path, f_path, m_path) in enumerate(zip(paris_water_paths, paris_fat_paths, paris_mask_paths)):
        (wdata, fdata, affine, orig_shape, spacing, resamp_shape) = load_nifti_2chan(w_path, f_path, target_spacing,
                                                                                     desired_size)
        mdata = load_nifti_mask(m_path, target_spacing, desired_size)
        pred_mask = predict_volume_2chan(model, wdata, fdata, DEVICE)

        # Restore geometry and save
        pred_nifti = restore_original_geometry(
            pred_mask, desired_size, resamp_shape, orig_shape, spacing, target_spacing, affine
        )
        nib.save(pred_nifti, os.path.join(OUTPUT_DIR, f'paris_patient_{i + 1}_pred_mask.nii.gz'))

        # Grab mid slice for plotting
        mid_slice = wdata.shape[0] // 2
        water_slice = wdata[mid_slice]
        mask_slice = mdata[mid_slice]
        pred_slice = pred_mask[mid_slice]
        # store for plotting
        paris_slices.append((water_slice, mask_slice, pred_slice))

    belgium_slices = []
    for i, (w_path, f_path, m_path) in enumerate(zip(belgium_water_paths, belgium_fat_paths, belgium_mask_paths)):
        (wdata, fdata, affine, orig_shape, spacing, resamp_shape) = load_nifti_2chan(w_path, f_path, target_spacing,
                                                                                     desired_size)
        mdata = load_nifti_mask(m_path, target_spacing, desired_size)
        pred_mask = predict_volume_2chan(model, wdata, fdata, DEVICE)

        # Restore geometry and save
        pred_nifti = restore_original_geometry(
            pred_mask, desired_size, resamp_shape, orig_shape, spacing, target_spacing, affine
        )
        nib.save(pred_nifti, os.path.join(OUTPUT_DIR, f'belgium_patient_{i + 1}_pred_mask.nii.gz'))

        # Mid-slice
        mid_slice = wdata.shape[0] // 2
        water_slice = wdata[mid_slice]
        mask_slice = mdata[mid_slice]
        pred_slice = pred_mask[mid_slice]
        belgium_slices.append((water_slice, mask_slice, pred_slice))

    # Plot results
    def plot_results(slices, name):
        num = len(slices)
        fig, axes = plt.subplots(num, 4, figsize=(20, 5 * num))
        if num == 1:
            axes = [axes]
        for i, (img_sl, mask_sl, pred_sl) in enumerate(slices):
            axes[i][0].imshow(img_sl, cmap='gray')
            axes[i][0].set_title(f'{name} {i + 1} - Water Slice')
            axes[i][0].axis('off')

            axes[i][1].imshow(mask_sl, cmap='gray')
            axes[i][1].set_title('Ground Truth Mask')
            axes[i][1].axis('off')

            axes[i][2].imshow(pred_sl, cmap='gray')
            axes[i][2].set_title('Predicted Mask')
            axes[i][2].axis('off')

            # Overlay
            overlay = np.zeros((*img_sl.shape, 3))
            overlay[(mask_sl == 1) & (pred_sl == 1)] = [0, 1, 0]
            overlay[(mask_sl == 1) & (pred_sl == 0)] = [0, 0, 1]
            overlay[(mask_sl == 0) & (pred_sl == 1)] = [1, 0, 0]

            axes[i][3].imshow(img_sl, cmap='gray')
            axes[i][3].imshow(overlay, alpha=0.3)
            axes[i][3].set_title('Overlay (green=TP, red=FP, blue=FN)')
            axes[i][3].axis('off')

        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"{name.lower()}_predictions.png")
        plt.savefig(out_path)
        print(f"Saved {name} predictions plot at {out_path}")
        plt.show()

    plot_results(paris_slices, "Paris")
    plot_results(belgium_slices, "Belgium")


if __name__ == '__main__':
    main()
