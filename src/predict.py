import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from src.models.unet import UNet3D
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

def load_nifti_image(nifti_path, target_spacing, desired_size):
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.float32)
    affine = img.affine
    header = img.header

    original_shape = data.shape  # (X,Y,Z)
    voxel_spacing = header.get_zooms()  # Original voxel spacing

    p975 = np.percentile(data, 99)
    data = np.clip(data, 0, p975) / p975

    zoom_factors = (
        voxel_spacing[0] / target_spacing[0],
        voxel_spacing[1] / target_spacing[1],
        voxel_spacing[2] / target_spacing[2]
    )
    data_resampled = zoom(data, zoom_factors, order=1)
    resampled_shape = data_resampled.shape

    data_resampled = np.transpose(data_resampled, (2, 0, 1))
    data_resized = resize(
        data_resampled,
        desired_size,
        mode='reflect',
        anti_aliasing=True
    )

    return data_resized, affine, original_shape, voxel_spacing, resampled_shape

def load_nifti_mask(nifti_path, target_spacing, desired_size):
    img = nib.load(nifti_path)
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

def load_model(checkpoint_path, device):
    model = UNet3D(n_channels=1, n_classes=1, bilinear=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Model loaded successfully.")
    return model

def predict_volume(model, image_data, device, threshold=0.5):
    # image_data: (D,H,W)
    with torch.no_grad():
        image_tensor = torch.from_numpy(image_data).unsqueeze(0).unsqueeze(0).float().to(device)
        output = model(image_tensor)
        prob = torch.sigmoid(output).cpu().numpy()[0,0]
        predicted_mask = (prob > threshold).astype(np.uint8)
    return predicted_mask

def restore_original_geometry(pred_mask, desired_size, resampled_shape, original_shape, voxel_spacing, target_spacing, affine):
    pred_mask_resampled = resize(
        pred_mask,
        (resampled_shape[2], resampled_shape[0], resampled_shape[1]),
        order=0,
        preserve_range=True,
        anti_aliasing=False
    ).astype(np.uint8)

    pred_mask_resampled = np.transpose(pred_mask_resampled, (1, 2, 0))

    inverse_zoom_factors = (
        voxel_spacing[0] / target_spacing[0],
        voxel_spacing[1] / target_spacing[1],
        voxel_spacing[2] / target_spacing[2]
    )

    pred_mask_original = zoom(pred_mask_resampled, inverse_zoom_factors, order=0).astype(np.uint8)

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

    target_spacing = (1.75, 1.75, 3.0)
    desired_size = (32, 128, 128)

    CHECKPOINT_PATH = 'outputs/checkpoints/Simple-Unet3D-cropped-augment/best_model.pth.tar'
    TEST_PARIS_DIR = '../data/cropped_test_paris_data/'
    TEST_BELGIUM_DIR = '../data/cropped_test_belgium_data/'
    OUTPUT_DIR = 'outputs/predictions/Simple-Unet3D-cropped-augment'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model(CHECKPOINT_PATH, device)

    paris_image_paths, paris_mask_paths = get_test_data_paths(TEST_PARIS_DIR, has_masks=True)
    belgium_image_paths, belgium_mask_paths = get_test_data_paths(TEST_BELGIUM_DIR, has_masks=True)

    paris_image_paths = paris_image_paths[:4]
    paris_mask_paths = paris_mask_paths[:4]
    belgium_image_paths = belgium_image_paths[:4]
    belgium_mask_paths = belgium_mask_paths[:4]

    paris_slices = []
    for idx, (image_path, mask_path) in enumerate(zip(paris_image_paths, paris_mask_paths)):
        image_data, affine, original_shape, voxel_spacing, resampled_shape = load_nifti_image(image_path, target_spacing, desired_size)
        mask_data = load_nifti_mask(mask_path, target_spacing, desired_size)
        predicted_mask = predict_volume(model, image_data, device, threshold=0.5)

        pred_nifti = restore_original_geometry(
            predicted_mask, desired_size, resampled_shape, original_shape, voxel_spacing, target_spacing, affine
        )
        nib.save(pred_nifti, os.path.join(OUTPUT_DIR, f'paris_patient_{idx+1}_pred_mask.nii.gz'))

        mid_slice = image_data.shape[0] // 2
        image_slice = image_data[mid_slice,:,:]
        mask_slice = mask_data[mid_slice,:,:]
        pred_slice = predicted_mask[mid_slice,:,:]
        paris_slices.append((image_slice, mask_slice, pred_slice))

    belgium_slices = []
    for idx, (image_path, mask_path) in enumerate(zip(belgium_image_paths, belgium_mask_paths)):
        image_data, affine, original_shape, voxel_spacing, resampled_shape = load_nifti_image(image_path, target_spacing, desired_size)
        mask_data = load_nifti_mask(mask_path, target_spacing, desired_size)
        predicted_mask = predict_volume(model, image_data, device, threshold=0.5)

        pred_nifti = restore_original_geometry(
            predicted_mask, desired_size, resampled_shape, original_shape, voxel_spacing, target_spacing, affine
        )
        nib.save(pred_nifti, os.path.join(OUTPUT_DIR, f'belgium_patient_{idx+1}_pred_mask.nii.gz'))

        mid_slice = image_data.shape[0] // 2
        image_slice = image_data[mid_slice,:,:]
        mask_slice = mask_data[mid_slice,:,:]
        pred_slice = predicted_mask[mid_slice,:,:]
        belgium_slices.append((image_slice, mask_slice, pred_slice))

    # Plotting results
    def plot_results(slices, name):
        num = len(slices)
        fig, axes = plt.subplots(num, 4, figsize=(20, 5 * num))
        if num == 1:
            axes = [axes]
        for i, (img_sl, mask_sl, pred_sl) in enumerate(slices):
            axes[i][0].imshow(img_sl, cmap='gray', aspect='auto')
            axes[i][0].set_title(f'{name} Patient {i+1} - Image')
            axes[i][0].axis('off')

            axes[i][1].imshow(mask_sl, cmap='gray', aspect='auto')
            axes[i][1].set_title(f'{name} Patient {i+1} - Ground Truth Mask')
            axes[i][1].axis('off')

            axes[i][2].imshow(pred_sl, cmap='gray', aspect='auto')
            axes[i][2].set_title(f'{name} Patient {i+1} - Predicted Mask')
            axes[i][2].axis('off')

            overlay = np.zeros((*img_sl.shape, 3))
            overlay[(mask_sl==1) & (pred_sl==1)] = [0,1,0]
            overlay[(mask_sl==1) & (pred_sl==0)] = [0,0,1]
            overlay[(mask_sl==0) & (pred_sl==1)] = [1,0,0]

            axes[i][3].imshow(img_sl, cmap='gray', aspect='auto')
            axes[i][3].imshow(overlay, alpha=0.22, aspect='auto')
            axes[i][3].set_title(f'{name} Patient {i+1} - Overlay')
            axes[i][3].axis('off')
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f'{name.lower()}_predictions.png')
        plt.savefig(plot_path)
        print(f"{name} predictions plot saved at {plot_path}")

    plot_results(paris_slices, "Paris")
    plot_results(belgium_slices, "Belgium")

if __name__ == '__main__':
    main()
