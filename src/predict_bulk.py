import os
import torch
import numpy as np
import nibabel as nib
from models.unet import UNet
from skimage.transform import resize
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_bulk_image_paths(test_dir):
    """
    For each patient directory in test_belgium_bulk/, find:
    - A file containing "water" in its name -> water image
    - A file containing "fatfrac" in its name -> fat fraction image

    Returns: A list of tuples (patient_name, water_path, fatfrac_path)
    """
    patient_dirs = [os.path.join(test_dir, d) for d in os.listdir(test_dir)
                    if os.path.isdir(os.path.join(test_dir, d))]
    image_paths = []
    for p_dir in patient_dirs:
        files_in_patient = os.listdir(p_dir)

        # Initialize variables
        water_path = None
        fat_path = None

        # Search for water and fatfrac files
        for f in files_in_patient:
            if 'water' in f and f.endswith('.nii.gz'):
                water_path = os.path.join(p_dir, f)
            if 'fatfrac' in f and f.endswith('.nii.gz'):
                fat_path = os.path.join(p_dir, f)

        if water_path is None or fat_path is None:
            print(f"Missing water or fatfrac image in {p_dir}. Skipping this patient.")
            continue

        patient_name = os.path.basename(p_dir)
        image_paths.append((patient_name, water_path, fat_path))
    return image_paths

def load_nifti_image(nifti_path, target_spacing=(3.0, 1.7188, 1.7188)):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header

    voxel_spacing = header.get_zooms()  # (X, Y, Z)
    voxel_spacing = (voxel_spacing[2], voxel_spacing[1], voxel_spacing[0])

    data = data.astype(np.float32)
    # Normalize
    data = (data - np.min(data)) / (np.ptp(data))

    zoom_factors = (
        voxel_spacing[0] / target_spacing[0],
        voxel_spacing[1] / target_spacing[1],
        voxel_spacing[2] / target_spacing[2]
    )
    data_resampled = zoom(data, zoom_factors, order=1)

    return data_resampled, affine

def load_model(checkpoint_path, device):
    model = UNet(n_channels=2, n_classes=1, bilinear=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Model loaded successfully.")
    return model

def predict_volume(model, water_data, fat_data, device, desired_size=(256, 256), threshold=0.5):
    H, W, D = water_data.shape
    predicted_masks = np.zeros((H, W, D), dtype=np.uint8)

    for i in range(D):
        water_slice = water_data[:, :, i]
        fat_slice = fat_data[:, :, i]

        water_resized = resize(water_slice, desired_size, mode='reflect', anti_aliasing=True)
        fat_resized = resize(fat_slice, desired_size, mode='reflect', anti_aliasing=True)

        image_tensor = np.stack([water_resized, fat_resized], axis=0)  # (2, H, W)
        image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).float().to(device)  # (1,2,H,W)

        with torch.no_grad():
            output = model(image_tensor)

        probability_map = torch.sigmoid(output).cpu().numpy()[0, 0, :, :]
        predicted_mask_resized = (probability_map > threshold).astype(np.uint8)

        predicted_mask_original_size = resize(
            predicted_mask_resized,
            (H, W),
            order=0,
            preserve_range=True,
            anti_aliasing=False
        ).astype(np.uint8)

        predicted_masks[:, :, i] = predicted_mask_original_size

        if (i + 1) % 10 == 0 or (i + 1) == D:
            print(f"Processed slice {i+1}/{D}")

    return predicted_masks

def main():
    # Update these paths as necessary
    CHECKPOINT_PATH = 'outputs/checkpoints/Simple-Unet-voxel-full-fat/checkpoint_epoch_30.pth.tar'
    TEST_BULK_DIR = '../data/test_belgium_bulk/'
    OUTPUT_DIR = 'outputs/predictions/Simple-Unet-voxel-full-fat/bulk/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model(CHECKPOINT_PATH, device)

    bulk_images = get_bulk_image_paths(TEST_BULK_DIR)
    if not bulk_images:
        print("No valid patients found in the bulk directory.")
        return

    images_list = []
    preds_list = []
    names_list = []

    for patient_name, water_path, fat_path in bulk_images:
        print(f"Predicting for patient: {patient_name}")
        water_data, affine = load_nifti_image(water_path)
        fat_data, _ = load_nifti_image(fat_path)

        predicted_masks = predict_volume(model, water_data, fat_data, device, desired_size=(256, 256), threshold=0.5)

        output_mask_path = os.path.join(OUTPUT_DIR, f'{patient_name}_pred_mask.nii.gz')
        nib.save(nib.Nifti1Image(predicted_masks.astype(np.float32), affine), output_mask_path)
        print(f"Predicted mask saved at {output_mask_path}")

        images_list.append(water_data)  # use water channel for visualization
        preds_list.append(predicted_masks)
        names_list.append(patient_name)

    # Plot all patients in one figure (up to 15 if you have 15 patients)
    num_patients = len(images_list)
    rows = 3
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
    axes = axes.ravel()

    for i in range(num_patients):
        if i >= rows * cols:
            break
        image_data = images_list[i]
        predicted_masks = preds_list[i]
        patient_name = names_list[i]

        slice_idx = image_data.shape[2] // 2
        img_slice = image_data[:, :, slice_idx]
        pred_slice = predicted_masks[:, :, slice_idx]

        axes[i].imshow(img_slice, cmap='gray', aspect='auto')
        overlay = np.zeros((*img_slice.shape, 3))
        overlay[pred_slice == 1] = [1, 0, 0]  # Red for predicted mask
        axes[i].imshow(overlay, alpha=0.3, aspect='auto')
        axes[i].set_title(f'{patient_name}')
        axes[i].axis('off')

    # Hide unused subplots if fewer than 15 patients
    for j in range(i+1, rows*cols):
        axes[j].axis('off')

    plt.tight_layout()
    bulk_plot_path = os.path.join(OUTPUT_DIR, 'all_patients_overlay.png')
    plt.savefig(bulk_plot_path)
    print(f"All patients overlay plot saved at {bulk_plot_path}")

    plt.show()

if __name__ == '__main__':
    main()
