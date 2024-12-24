import os
import torch
import numpy as np
import nibabel as nib
from src.models.unet import UNet3D
from skimage.transform import resize
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

def get_bulk_image_paths(test_dir):
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

def load_nifti_image(nifti_path, target_spacing=(1.75, 1.75, 3.0), desired_size=(32, 128, 128)):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header

    original_shape = data.shape  # (X,Y,Z)
    voxel_spacing = header.get_zooms()

    data = data.astype(np.float32)
    p975 = np.percentile(data, 99)
    data = np.clip(data, 0, p975) / p975

    zoom_factors = (
        voxel_spacing[0] / target_spacing[0],
        voxel_spacing[1] / target_spacing[1],
        voxel_spacing[2] / target_spacing[2]
    )
    data_resampled = zoom(data, zoom_factors, order=1)
    resampled_shape = data_resampled.shape  # (X',Y',Z')

    # (X',Y',Z') -> (Z',X',Y')
    data_resampled = np.transpose(data_resampled, (2,0,1))

    data_resized = resize(
        data_resampled,
        desired_size,
        mode='reflect',
        anti_aliasing=True
    )
    return data_resized, affine, original_shape, voxel_spacing, resampled_shape

def load_model(checkpoint_path, device):
    model = UNet3D(n_channels=1, n_classes=1, bilinear=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Model loaded successfully.")
    return model

def predict_volume(model, image_data, device, threshold=0.5):
    with torch.no_grad():
        image_tensor = torch.from_numpy(image_data).unsqueeze(0).unsqueeze(0).float().to(device)
        output = model(image_tensor)
        prob = torch.sigmoid(output).cpu().numpy()[0,0]
        predicted_mask = (prob > threshold).astype(np.uint8)
    return predicted_mask

def restore_original_geometry(pred_mask, desired_size, resampled_shape, original_shape, voxel_spacing, target_spacing, affine):
    # Undo the steps as in predict.py
    # pred_mask: (D,H,W) = (Z',X',Y')
    pred_mask_resampled = resize(
        pred_mask,
        (resampled_shape[2], resampled_shape[0], resampled_shape[1]),
        order=0,
        preserve_range=True,
        anti_aliasing=False
    ).astype(np.uint8)

    # (Z',X',Y') -> (X',Y',Z')
    pred_mask_resampled = np.transpose(pred_mask_resampled, (1, 2, 0))

    # Inverse zoom
    inverse_zoom_factors = (
        voxel_spacing[0] / target_spacing[0],
        voxel_spacing[1] / target_spacing[1],
        voxel_spacing[2] / target_spacing[2]
    )
    pred_mask_original = zoom(pred_mask_resampled, inverse_zoom_factors, order=0).astype(np.uint8)

    # Ensure exact original shape
    pred_mask_original = resize(
        pred_mask_original,
        original_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False
    ).astype(np.uint8)

    pred_nifti = nib.Nifti1Image(pred_mask_original.astype(np.float32), affine)
    return pred_nifti

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    CHECKPOINT_PATH = 'outputs/checkpoints/Simple-Unet3D-cropped-augment/best_model.pth.tar'
    TEST_BULK_DIR = '../data/cropped_test_belgium_bulk/'
    OUTPUT_DIR = 'outputs/predictions/Simple-Unet3D-cropped-augment/bulk/'
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
        image_data, affine, original_shape, voxel_spacing, resampled_shape = load_nifti_image(image_path)
        predicted_masks = predict_volume(model, image_data, device, threshold=0.5)

        pred_nifti = restore_original_geometry(
            predicted_masks, (32, 128, 128), resampled_shape, original_shape, voxel_spacing, (1.75,1.75,3.0), affine
        )

        output_mask_path = os.path.join(OUTPUT_DIR, f'{patient_name}_pred_mask.nii.gz')
        nib.save(pred_nifti, output_mask_path)
        print(f"Predicted mask saved at {output_mask_path}")

        images_list.append(image_data)
        preds_list.append(predicted_masks)
        names_list.append(patient_name)

    num_patients = len(images_list)
    if num_patients < 15:
        print(f"Warning: Expected around 15 patients, got {num_patients}.")

    rows = 3
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
    axes = axes.ravel()

    for i in range(min(num_patients, 15)):
        image_data = images_list[i]
        predicted_masks = preds_list[i]
        patient_name = names_list[i]

        slice_idx = image_data.shape[0] // 2
        img_slice = image_data[slice_idx,:,:]
        pred_slice = predicted_masks[slice_idx,:,:]

        axes[i].imshow(img_slice, cmap='gray', aspect='auto')
        overlay = np.zeros((*img_slice.shape, 3))
        overlay[pred_slice == 1] = [1,0,0]
        axes[i].imshow(overlay, alpha=0.3, aspect='auto')
        axes[i].set_title(f'{patient_name}')
        axes[i].axis('off')

    plt.tight_layout()
    bulk_plot_path = os.path.join(OUTPUT_DIR, 'all_patients_overlay.png')
    plt.savefig(bulk_plot_path)
    print(f"Bulk overlay plot saved at {bulk_plot_path}")

    plt.show()

if __name__ == '__main__':
    main()
