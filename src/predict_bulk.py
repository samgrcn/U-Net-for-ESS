import os
import torch
import numpy as np
import nibabel as nib
from src.models.unet import UNet3D
from skimage.transform import resize
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_bulk_image_paths_2chan(test_dir):
    """
    Returns a list of tuples:
      [(patient_name, water_path, fat_path), ...]
    for each folder in test_dir that contains both
    the water file (' mDIXON-Quant_BH*') and 'fat.nii.gz'.
    """
    patient_dirs = [
        os.path.join(test_dir, d)
        for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    ]

    paths_2chan = []
    for p_dir in patient_dirs:
        files_in_patient = os.listdir(p_dir)
        patient_name = os.path.basename(p_dir)

        # find water
        water_file = None
        for fname in files_in_patient:
            # Check if 'water' in the file name (case-insensitive) and ends with .nii or .nii.gz
            if 'water' in fname.lower() and (fname.endswith('.nii') or fname.endswith('.nii.gz')):
                water_file = fname
                break
        # find fat
        fat_file = None
        for fname in files_in_patient:
            # Check if 'fat' in the file name (case-insensitive) and ends with .nii or .nii.gz
            if 'fat' in fname.lower() and (fname.endswith('.nii') or fname.endswith('.nii.gz')):
                fat_file = fname
                break
        if water_file is None:
            print(f"No water file found in {p_dir}, skipping.")
            continue
        if fat_file not in files_in_patient:
            print(f"No fat file found in {p_dir}, skipping.")
            continue

        water_path = os.path.join(p_dir, water_file)
        fat_path = os.path.join(p_dir, fat_file)

        paths_2chan.append((patient_name, water_path, fat_path))

    return paths_2chan


def load_model(checkpoint_path, device):
    model = UNet3D(n_channels=2, n_classes=1, bilinear=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Model loaded successfully.")
    return model


def load_nifti_2chan(water_path, fat_path, target_spacing=(1.75, 1.75, 3.0), desired_size=(32, 128, 128)):
    # WATER
    w_img = nib.load(water_path)
    w_data = w_img.get_fdata().astype(np.float32)
    w_header = w_img.header
    w_spacing = w_header.get_zooms()

    w_p99 = np.percentile(w_data, 99)
    w_data = np.clip(w_data, 0, w_p99) / (w_p99 + 1e-8)

    w_zoom = (
        w_spacing[0] / target_spacing[0],
        w_spacing[1] / target_spacing[1],
        w_spacing[2] / target_spacing[2]
    )
    w_data_resampled = zoom(w_data, w_zoom, order=1)
    w_data_resampled = np.transpose(w_data_resampled, (2, 0, 1))
    w_data_resized = resize(
        w_data_resampled,
        desired_size,
        mode='reflect',
        anti_aliasing=True
    )

    # FAT
    f_img = nib.load(fat_path)
    f_data = f_img.get_fdata().astype(np.float32)
    f_header = f_img.header
    f_spacing = f_header.get_zooms()

    f_p99 = np.percentile(f_data, 99)
    f_data = np.clip(f_data, 0, f_p99) / (f_p99 + 1e-8)

    f_zoom = (
        f_spacing[0] / target_spacing[0],
        f_spacing[1] / target_spacing[1],
        f_spacing[2] / target_spacing[2]
    )
    f_data_resampled = zoom(f_data, f_zoom, order=1)
    f_data_resampled = np.transpose(f_data_resampled, (2, 0, 1))
    f_data_resized = resize(
        f_data_resampled,
        desired_size,
        mode='reflect',
        anti_aliasing=True
    )

    # Save metadata for restoring geometry if needed
    affine = w_img.affine  # or f_img.affine
    original_shape = w_data.shape  # water's original shape
    resampled_shape = w_data_resampled.shape

    return (w_data_resized, f_data_resized,
            affine, original_shape, w_spacing, resampled_shape)


def restore_original_geometry(pred_mask, desired_size, resampled_shape,
                              original_shape, voxel_spacing,
                              target_spacing, affine):
    # same function as before
    pred_mask_resampled = resize(
        pred_mask,
        (resampled_shape[2], resampled_shape[0], resampled_shape[1]),
        order=0,
        preserve_range=True,
        anti_aliasing=False
    ).astype(np.uint8)
    pred_mask_resampled = np.transpose(pred_mask_resampled, (1, 2, 0))

    inv_zoom = (
        voxel_spacing[0] / target_spacing[0],
        voxel_spacing[1] / target_spacing[1],
        voxel_spacing[2] / target_spacing[2]
    )
    pred_mask_original = zoom(pred_mask_resampled, inv_zoom, order=0).astype(np.uint8)
    pred_mask_original = resize(
        pred_mask_original,
        original_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False
    ).astype(np.uint8)

    return nib.Nifti1Image(pred_mask_original.astype(np.float32), affine)


def predict_volume_2chan(model, water_data, fat_data, device, threshold=0.5):
    with torch.no_grad():
        w_tensor = torch.from_numpy(water_data).unsqueeze(0).float()
        f_tensor = torch.from_numpy(fat_data).unsqueeze(0).float()
        input_2ch = torch.stack([w_tensor, f_tensor], dim=1).to(device)
        output = model(input_2ch)
        prob = torch.sigmoid(output).cpu().numpy()[0, 0]
        pred_mask = (prob > threshold).astype(np.uint8)
    return pred_mask


def main():
    CHECKPOINT_PATH = 'outputs/checkpoints/Simple-Unet3D-cropped-augment-99-2channel/checkpoint_epoch_50.pth.tar'
    TEST_BULK_DIR = '../data/cropped_test_belgium_bulk/'
    OUTPUT_DIR = 'outputs/predictions/Simple-Unet3D-cropped-augment-99-2channel/bulk/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target_spacing = (1.75, 1.75, 3.0)
    desired_size = (32, 128, 128)

    model = load_model(CHECKPOINT_PATH, DEVICE)
    bulk_paths = get_bulk_image_paths_2chan(TEST_BULK_DIR)
    if not bulk_paths:
        print("No suitable water+fat pairs found in bulk directory.")
        return

    images_list = []
    preds_list = []
    names_list = []

    for (patient_name, w_path, f_path) in bulk_paths:
        print(f"Predicting for patient: {patient_name}")
        (w_data, f_data, affine, orig_shape, spacing, resamp_shape) = load_nifti_2chan(
            w_path, f_path, target_spacing, desired_size
        )
        pred_mask = predict_volume_2chan(model, w_data, f_data, DEVICE)

        pred_nifti = restore_original_geometry(
            pred_mask, desired_size, resamp_shape, orig_shape, spacing, target_spacing, affine
        )

        out_path = os.path.join(OUTPUT_DIR, f'{patient_name}_pred_mask.nii.gz')
        nib.save(pred_nifti, out_path)
        print(f"Saved predicted mask for {patient_name} -> {out_path}")

        images_list.append(w_data)  # or combine w_data/f_data for plotting
        preds_list.append(pred_mask)
        names_list.append(patient_name)

    # Optionally, create overlay plot for up to 15 patients
    num_patients = len(images_list)
    rows, cols = 3, 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
    axes = axes.ravel()

    for i in range(min(num_patients, 15)):
        water_data = images_list[i]
        pred_mask = preds_list[i]
        patient_name = names_list[i]

        mid_slice = water_data.shape[0] // 2
        img_slice = water_data[mid_slice, :, :]
        pm_slice = pred_mask[mid_slice, :, :]

        axes[i].imshow(img_slice, cmap='gray', aspect='auto')
        overlay = np.zeros((*img_slice.shape, 3))
        overlay[pm_slice == 1] = [1, 0, 0]
        axes[i].imshow(overlay, alpha=0.3, aspect='auto')
        axes[i].set_title(patient_name)
        axes[i].axis('off')

    plt.tight_layout()
    bulk_plot_path = os.path.join(OUTPUT_DIR, 'all_patients_overlay.png')
    plt.savefig(bulk_plot_path)
    plt.show()
    print(f"Saved overlay of all patients -> {bulk_plot_path}")


if __name__ == '__main__':
    main()
