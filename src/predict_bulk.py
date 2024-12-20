import os
import torch
import numpy as np
import nibabel as nib
from models.unet import UNet
from skimage.transform import resize
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

def get_bulk_image_paths(test_dir):
    """
    Scan through 'test_belgium_bulk/' directory.
    For each patient directory, find a NIfTI image containing 'water' in its filename.
    Returns: A list of tuples (patient_name, image_path)
    """
    patient_dirs = [os.path.join(test_dir, d) for d in os.listdir(test_dir)
                    if os.path.isdir(os.path.join(test_dir, d))]
    image_paths = []
    for p_dir in patient_dirs:
        files_in_patient = os.listdir(p_dir)
        water_image = None
        for f in files_in_patient:
            if 'water' in f and (f.endswith('.nii') or f.endswith('.nii.gz')):
                water_image = f
                break
        if water_image is None:
            print(f"No 'water' image found in {p_dir}. Skipping this patient.")
            continue
        image_path = os.path.join(p_dir, water_image)
        patient_name = os.path.basename(p_dir)
        image_paths.append((patient_name, image_path))
    return image_paths

# Given (3.0, 1.7188, 1.7188) in Z,X,Y, convert to X,Y,Z = (1.7188, 1.7188, 3.0)
def load_nifti_image(nifti_path, target_spacing=(1.7188, 1.7188, 3.0)):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header

    # voxel_spacing = (X_spacing, Y_spacing, Z_spacing)
    voxel_spacing = header.get_zooms()  # no reordering

    data = data.astype(np.float32)
    # Normalize to [0, 1]
    p975 = np.percentile(data, 99)
    data = np.clip(data, 0, p975)
    data = data / p975

    # Resample volume to target_spacing (X, Y, Z)
    zoom_factors = (
        voxel_spacing[0] / target_spacing[0],
        voxel_spacing[1] / target_spacing[1],
        voxel_spacing[2] / target_spacing[2]
    )
    data_resampled = zoom(data, zoom_factors, order=0)

    return data_resampled, affine

def load_model(checkpoint_path, device):
    # Loading a 1-channel model
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

        # Resize the current slice to desired_size
        image_resized = resize(
            current_slice,
            desired_size,
            mode='reflect',
            anti_aliasing=True
        )

        # Add batch and channel dimensions for single-channel input
        image_tensor = torch.from_numpy(image_resized).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(image_tensor)

        probability_map = torch.sigmoid(output).cpu().numpy()[0, 0, :, :]
        predicted_mask_resized = (probability_map > threshold).astype(np.uint8)

        # Resize mask back to original slice size
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Update these paths accordingly:
    CHECKPOINT_PATH = 'outputs/checkpoints/Simple-Unet-voxel-full-994/best_model.pth.tar'
    TEST_BULK_DIR = '../data/test_belgium_bulk/'
    OUTPUT_DIR = 'outputs/predictions/Simple-Unet-voxel-full-994/bulk/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model(CHECKPOINT_PATH, device)

    bulk_images = get_bulk_image_paths(TEST_BULK_DIR)
    if not bulk_images:
        print("No 'water' images found in the bulk directory.")
        return

    images_list = []
    preds_list = []
    names_list = []

    for patient_name, image_path in bulk_images:
        print(f"Predicting for patient: {patient_name}, image: {image_path}")
        image_data, affine = load_nifti_image(image_path)

        # Perform prediction
        predicted_masks = predict_volume(model, image_data, device, desired_size=(256, 256), threshold=0.5)

        # Save predicted mask as NIfTI
        output_mask_path = os.path.join(OUTPUT_DIR, f'{patient_name}_pred_mask.nii.gz')
        predicted_mask_nifti = nib.Nifti1Image(predicted_masks.astype(np.float32), affine)
        nib.save(predicted_mask_nifti, output_mask_path)
        print(f"Predicted mask saved at {output_mask_path}")

        images_list.append(image_data)
        preds_list.append(predicted_masks)
        names_list.append(patient_name)

    num_patients = len(images_list)
    if num_patients < 15:
        print(f"Warning: Expected 15 patients, but got {num_patients}. We'll only plot what we have.")

    rows = 3
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
    axes = axes.ravel()

    for i in range(num_patients):
        image_data = images_list[i]
        predicted_masks = preds_list[i]
        patient_name = names_list[i]

        # Middle slice index
        slice_idx = image_data.shape[2] // 2
        img_slice = image_data[:, :, slice_idx]
        pred_slice = predicted_masks[:, :, slice_idx]

        axes[i].imshow(img_slice, cmap='gray', aspect='auto')
        overlay = np.zeros((*img_slice.shape, 3))
        overlay[pred_slice == 1] = [1, 0, 0]  # predicted mask in red
        axes[i].imshow(overlay, alpha=0.3, aspect='auto')
        axes[i].set_title(f'{patient_name}')
        axes[i].axis('off')

    plt.tight_layout()
    bulk_plot_path = os.path.join(OUTPUT_DIR, 'all_15_patients_overlay.png')
    plt.savefig(bulk_plot_path)
    print(f"All 15 patients overlay plot saved at {bulk_plot_path}")

    plt.show()

if __name__ == '__main__':
    main()
