import os
import nibabel as nib
import numpy as np

# Directories
BASE_DIR = '../data/'
PARIS_DIR = os.path.join(BASE_DIR, 'paris_data')
TEST_PARIS_DIR = os.path.join(BASE_DIR, 'test_paris_data')
TEST_BELGIUM_DIR = os.path.join(BASE_DIR, 'test_belgium_data')
TEST_BELGIUM_BULK_DIR = os.path.join(BASE_DIR, 'test_belgium_bulk')

CROPPED_PARIS_DIR = os.path.join(BASE_DIR, 'cropped_paris_data')
CROPPED_TEST_PARIS_DIR = os.path.join(BASE_DIR, 'cropped_test_paris_data')
CROPPED_TEST_BELGIUM_DIR = os.path.join(BASE_DIR, 'cropped_test_belgium_data')
CROPPED_TEST_BELGIUM_BULK_DIR = os.path.join(BASE_DIR, 'cropped_test_belgium_bulk')

os.makedirs(CROPPED_PARIS_DIR, exist_ok=True)
os.makedirs(CROPPED_TEST_PARIS_DIR, exist_ok=True)
os.makedirs(CROPPED_TEST_BELGIUM_DIR, exist_ok=True)
os.makedirs(CROPPED_TEST_BELGIUM_BULK_DIR, exist_ok=True)

def get_subdirs(directory):
    """Returns a list of full paths to subdirectories in a directory."""
    return [os.path.join(directory, d) for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))]

def find_image_file(files, require_mask=True):
    """
    Find the appropriate image file in the folder.
    If require_mask is True, we look for the Paris/Belgium main image (' mDIXON-Quant_BH*.nii').
    If require_mask is False, this likely is the test_belgium_bulk, so we look for 'water' in the filename.
    """
    if require_mask:
        for fname in [' mDIXON-Quant_BH_v3.nii', ' mDIXON-Quant_BH.nii']:
            if fname in files:
                return fname
        return None
    else:
        for f in files:
            if 'water' in f and (f.endswith('.nii') or f.endswith('.nii.gz')):
                return f
        return None

def find_mask_file(files):
    """Find the mask file in the folder ('erector.nii' or 'erector.nii.gz')."""
    for fname in ['erector.nii', 'erector.nii.gz']:
        if fname in files:
            return fname
    return None

def load_nifti(path):
    """Load a NIfTI image and return the data, affine, and header."""
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine, img.header

def save_nifti(data, affine, out_path):
    """Save data as NIfTI using the given affine."""
    new_img = nib.Nifti1Image(data, affine)
    nib.save(new_img, out_path)

def crop_image_fixed(image_data):
    """
    Crop the image according to the fixed rules:
    - Keep from 20% to 80% along the X dimension.
    - Keep from Y/2 to the end along Y dimension.
    - Keep full Z dimension.
    """
    X, Y, Z = image_data.shape

    x_start = int(X*0.25)
    x_end = int(X*0.75) - 1

    y_start = int(Y//2)
    y_end = Y - 1

    z_start = 0
    z_end = Z - 1

    cropped_image = image_data[x_start:x_end+1, y_start:y_end+1, z_start:z_end+1]

    return cropped_image, (X, Y, Z, x_start, x_end, y_start, y_end, z_start, z_end)

def process_directory(input_dir, output_dir, require_mask=True):
    """
    For each patient folder in input_dir:
    - If require_mask=True, load image and mask.
      If not found, skip.
    - If require_mask=False, only load the 'water' image.
    - Crop the image using fixed rules.
    - If a mask is available (require_mask=True), crop and save it too.
    - Save cropping indices to a file.
    """
    patient_dirs = get_subdirs(input_dir)
    for p_dir in patient_dirs:
        files = os.listdir(p_dir)
        img_name = find_image_file(files, require_mask=require_mask)

        if img_name is None:
            print(f"No valid image found in {p_dir}. Skipping.")
            continue

        img_path = os.path.join(p_dir, img_name)
        image_data, affine, header = load_nifti(img_path)

        cropped_image, crop_info = crop_image_fixed(image_data)
        _, (X, Y, Z, x_start, x_end, y_start, y_end, z_start, z_end) = (None, crop_info)

        patient_id = os.path.basename(p_dir)
        out_patient_dir = os.path.join(output_dir, patient_id)
        os.makedirs(out_patient_dir, exist_ok=True)

        out_img_path = os.path.join(out_patient_dir, img_name)

        save_nifti(cropped_image, affine, out_img_path)

        if require_mask:
            mask_name = find_mask_file(files)
            if mask_name is None:
                print(f"No mask found in {p_dir}. Skipping.")
                continue
            mask_path = os.path.join(p_dir, mask_name)
            mask_data, _, _ = load_nifti(mask_path)
            cropped_mask = mask_data[x_start:x_end+1, y_start:y_end+1, z_start:z_end+1]

            out_mask_path = os.path.join(out_patient_dir, 'erector.nii')
            save_nifti(cropped_mask, affine, out_mask_path)

        crop_txt_path = os.path.join(out_patient_dir, 'crop_info.txt')
        with open(crop_txt_path, 'w') as f:
            f.write("Original Shape: {} {} {}\n".format(X, Y, Z))
            f.write("x_start: {}, x_end: {}\n".format(x_start, x_end))
            f.write("y_start: {}, y_end: {}\n".format(y_start, y_end))
            f.write("z_start: {}, z_end: {}\n".format(z_start, z_end))

        print(f"Cropped data saved to {out_patient_dir}")

if __name__ == "__main__":
    print("Cropping Paris data...")
    process_directory(PARIS_DIR, CROPPED_PARIS_DIR, require_mask=True)

    print("Cropping Test Paris data...")
    process_directory(TEST_PARIS_DIR, CROPPED_TEST_PARIS_DIR, require_mask=True)

    print("Cropping Test Belgium data...")
    process_directory(TEST_BELGIUM_DIR, CROPPED_TEST_BELGIUM_DIR, require_mask=True)

    print("Cropping Test Belgium Bulk data (no mask)...")
    process_directory(TEST_BELGIUM_BULK_DIR, CROPPED_TEST_BELGIUM_BULK_DIR, require_mask=False)

    print("Cropping completed.")
