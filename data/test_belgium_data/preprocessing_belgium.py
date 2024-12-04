import nibabel as nib
import numpy as np
import os

def mirror_nifti(input_path):
    """
    Mirrors a NIfTI file along the left-right axis and overwrites the original file.

    Args:
        input_path (str): Path to the input NIfTI file (.nii or .nii.gz).
    """
    try:
        nifti_img = nib.load(input_path)
        nifti_data = nifti_img.get_fdata()
        mirrored_data = np.flip(nifti_data, axis=0)  # Flip along the left-right axis
        mirrored_img = nib.Nifti1Image(mirrored_data, nifti_img.affine, nifti_img.header)
        nib.save(mirrored_img, input_path)
        print(f"Mirrored file: {input_path}")
    except Exception as e:
        print(f"Error processing file {input_path}: {e}")

def process_folders(base_folder):
    """
    Processes all subfolders in the base folder, mirroring NIfTI files in place.

    Args:
        base_folder (str): Path to the base folder (e.g., test_belgium_data).
    """
    for subfolder in sorted(os.listdir(base_folder)):
        subfolder_path = os.path.join(base_folder, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Processing folder: {subfolder_path}")
            for filename in os.listdir(subfolder_path):
                if filename.endswith(".nii") or filename.endswith(".nii.gz"):
                    file_path = os.path.join(subfolder_path, filename)
                    mirror_nifti(file_path)

if __name__ == "__main__":
    # Set the base folder where the script is placed
    base_folder = os.path.abspath(os.path.dirname(__file__))
    process_folders(base_folder)
