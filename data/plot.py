import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


def load_and_view_middle_slice(nifti_path):
    """
    Load a NIfTI image and display the middle slice of the volume.

    Parameters:
        nifti_path (str): Path to the NIfTI image file.
    """
    try:
        nifti_image = nib.load(nifti_path)
        header = nifti_image.header
        print(nifti_image.shape)

        image_data = nifti_image.get_fdata()

        middle_index = [dim // 2 for dim in image_data.shape[:3]]

        plt.figure(figsize=(8, 8))
        plt.imshow(image_data[:, :, middle_index[2]], cmap="gray")
        plt.title(f"Middle Slice of {nifti_path}")
        plt.axis("off")
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

nifti_path = 'paris_data/5/ mDIXON-Quant_BH.nii'
load_and_view_middle_slice(nifti_path)
