import os
import nibabel as nib
import numpy as np
from jupyter_server.transutils import base_dir


def flip_nifti_image(input_path):
    """
    Loads a NIfTI image, flips it left to right, and saves the flipped image,
    overwriting the original file.

    Parameters:
    - input_path: Path to the input NIfTI file.
    """
    try:
        img = nib.load(input_path)
        data = img.get_fdata()
        flipped_data = np.flip(data, axis=1)

        flipped_img = nib.Nifti1Image(flipped_data, img.affine, img.header)

        nib.save(flipped_img, input_path)
        print(f"Successfully flipped and overwritten: {input_path}")

    except Exception as e:
        print(f"Failed to process {input_path}: {e}")

def main():
    dir = 'full_paris_data'
    base_dir = os.path.join(dir)


    for subfolder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, subfolder)
        if not os.path.isdir(folder_path):
            print(f"Folder {folder_path} does not exist. Skipping.")
            continue

        image_filename = " mDIXON-Quant_BH.nii.gz"
        mask_filename = "erector.nii"

        image_path = os.path.join(folder_path, image_filename)
        mask_path = os.path.join(folder_path, mask_filename)

        missing_files = False
        if not os.path.isfile(image_path):
            print(f"Image file {image_path} not found. Skipping this file.")
            missing_files = True
        if not os.path.isfile(mask_path):
            print(f"Mask file {mask_path} not found. Skipping this file.")
            missing_files = True
        if missing_files:
            continue

        print(f"\nProcessing folder: {subfolder}")
        flip_nifti_image(image_path)
        flip_nifti_image(mask_path)

    print("\nAll specified folders have been processed.")

if __name__ == "__main__":
    main()
