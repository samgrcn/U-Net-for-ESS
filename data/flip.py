import os
import nibabel as nib
import numpy as np

def flip_nifti_image(input_path):
    """
    Loads a NIfTI image, flips it left to right, and saves the flipped image,
    overwriting the original file.

    Parameters:
    - input_path: Path to the input NIfTI file.
    """
    try:
        # Load the NIfTI image
        img = nib.load(input_path)
        data = img.get_fdata()

        # Flip the data along the left-right axis (commonly axis=1, adjust if needed)
        flipped_data = np.flip(data, axis=1)

        # Create a new NIfTI image with the flipped data
        flipped_img = nib.Nifti1Image(flipped_data, img.affine, img.header)

        # Save the flipped image, overwriting the original file
        nib.save(flipped_img, input_path)
        print(f"Successfully flipped and overwritten: {input_path}")

    except Exception as e:
        print(f"Failed to process {input_path}: {e}")

def main():
    # Define the base directory
    base_dir = 'test_full_paris_data'

    # Iterate through subfolders in the base directory
    for subfolder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, subfolder)
        if not os.path.isdir(folder_path):
            continue

        print(f"\nProcessing folder: {subfolder}")

        # Process files containing 'fatfrac' in their names
        for file_name in os.listdir(folder_path):
            if ' mDIXON-Quant_BH' in file_name and file_name.endswith('.nii.gz'):
                file_path = os.path.join(folder_path, file_name)

                # Flip the file
                flip_nifti_image(file_path)

    print("\nAll specified files have been processed.")

if __name__ == "__main__":
    main()
