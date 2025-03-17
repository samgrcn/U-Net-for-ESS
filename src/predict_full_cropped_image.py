import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Directories
BASE_DIR = '../data/'
TEST_BULK_DIR = os.path.join(BASE_DIR, 'test_belgium_bulk')
CROPPED_TEST_BELGIUM_DIR = os.path.join(BASE_DIR, 'cropped_test_belgium_bulk')

PRED_DIR = 'outputs/predictions/Unet-3ch-voxel-box/bulk'

def get_bulk_image_paths(test_dir):
    """
    Returns a list of tuples (patient_name, image_path) for all water images in test_belgium_bulk.
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

def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    return data, affine

def read_crop_info(patient_id):
    """Read the crop_info.txt for this patient from the cropped directory."""
    crop_info_path = os.path.join(CROPPED_TEST_BELGIUM_DIR, patient_id, 'crop_info.txt')
    if not os.path.exists(crop_info_path):
        raise FileNotFoundError(f"crop_info.txt not found for patient {patient_id}")

    with open(crop_info_path, 'r') as f:
        lines = f.readlines()

    orig_shape_line = lines[0].strip().split(":")[1].strip().split()
    X, Y, Z = int(orig_shape_line[0]), int(orig_shape_line[1]), int(orig_shape_line[2])

    x_line = lines[1].strip().split(',')
    x_start = int(x_line[0].split(':')[1])
    x_end = int(x_line[1].split(':')[1])

    y_line = lines[2].strip().split(',')
    y_start = int(y_line[0].split(':')[1])
    y_end = int(y_line[1].split(':')[1])

    z_line = lines[3].strip().split(',')
    z_start = int(z_line[0].split(':')[1])
    z_end = int(z_line[1].split(':')[1])

    return (X, Y, Z, x_start, x_end, y_start, y_end, z_start, z_end)

def main():
    bulk_images = get_bulk_image_paths(TEST_BULK_DIR)

    if not bulk_images:
        print("No images found in test_belgium_bulk.")
        return

    bulk_images.sort(key=lambda x: x[0])

    images_list = []
    preds_list = []
    names_list = []

    for patient_name, image_path in bulk_images:
        full_image_data, affine = load_nifti(image_path)

        pred_mask_name = f'{patient_name}_pred_mask.nii.gz'
        pred_mask_path = os.path.join(PRED_DIR, pred_mask_name)

        if not os.path.exists(pred_mask_path):
            print(f"No predicted mask found for {patient_name} at {pred_mask_path}. Skipping.")
            continue

        cropped_pred_data, _ = load_nifti(pred_mask_path)

        X, Y, Z, x_start, x_end, y_start, y_end, z_start, z_end = read_crop_info(patient_name)

        full_pred_mask = np.zeros((X, Y, Z), dtype=np.float32)
        full_pred_mask[x_start:x_end+1, y_start:y_end+1, z_start:z_end+1] = cropped_pred_data

        images_list.append(full_image_data)
        preds_list.append(full_pred_mask)
        names_list.append(patient_name)

    num_patients = len(images_list)
    if num_patients == 0:
        print("No patients to plot.")
        return

    rows = 3
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
    axes = axes.ravel()

    for i in range(num_patients):
        image_data = images_list[i]
        predicted_masks = preds_list[i]
        patient_name = names_list[i]

        slice_idx = image_data.shape[2] // 2
        img_slice = image_data[:, :, slice_idx]
        pred_slice = predicted_masks[:, :, slice_idx]

        axes[i].imshow(img_slice, cmap='gray', aspect='auto')
        overlay = np.zeros((*img_slice.shape, 3), dtype=np.float32)
        overlay[pred_slice > 0.5] = [1, 0, 0]
        axes[i].imshow(overlay, alpha=0.3, aspect='auto')
        axes[i].set_title(f'{patient_name}')
        axes[i].axis('off')

    for j in range(num_patients, rows*cols):
        axes[j].axis('off')

    plt.tight_layout()

    output_overlay_path = os.path.join(PRED_DIR, 'all_15_patients_full_overlay.png')
    plt.savefig(output_overlay_path)
    print(f"All patients full-size overlay plot saved at {output_overlay_path}")
    plt.show()

if __name__ == '__main__':
    main()
