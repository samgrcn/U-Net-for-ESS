import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib  # For NIfTI files
import pydicom

class SliceDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        """
        Args:
            image_paths (list): List of paths to image files (NIfTI or DICOM directories).
            mask_paths (list): List of paths to mask files (NIfTI or DICOM directories).
        """
        self.image_slices = []
        self.mask_slices = []

        for img_path, mask_path in zip(image_paths, mask_paths):
            # Load image slices
            image_slices = self.load_slices(img_path, is_mask=False)
            # Load mask slices
            mask_slices = self.load_slices(mask_path, is_mask=True)
            # Ensure the number of slices match
            assert len(image_slices) == len(mask_slices), f"Mismatch in number of slices between image and mask at {img_path}"
            # Append slices to the dataset lists
            self.image_slices.extend(image_slices)
            self.mask_slices.extend(mask_slices)

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):
        image = self.image_slices[idx]
        mask = self.mask_slices[idx]

        # Normalize image to [0, 1]
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        # Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0).float()  # Shape: [1, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask

    def load_slices(self, path, is_mask=False):
        """
        Loads slices from a NIfTI file or DICOM directory.
        Args:
            path (str): Path to NIfTI file or DICOM directory.
            is_mask (bool): Whether the path is for a mask.
        Returns:
            List of 2D numpy arrays.
        """
        slices = []
        if os.path.isdir(path):
            # Load DICOM slices
            slices = self.load_dicom_slices(path)
        elif path.endswith('.nii') or path.endswith('.nii.gz'):
            # Load NIfTI slices
            slices = self.load_nifti_slices(path, is_mask=is_mask)
        else:
            raise ValueError(f"Unsupported file format: {path}")
        return slices

    def load_dicom_slices(self, dicom_dir):
        # Get all DICOM file paths
        dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
        # Sort files based on InstanceNumber or SliceLocation
        dicom_files.sort(key=lambda x: int(pydicom.dcmread(x).InstanceNumber))
        # Load slices
        slices = []
        for file in dicom_files:
            ds = pydicom.dcmread(file)
            image = ds.pixel_array.astype(np.float32)
            slices.append(image)
        return slices  # List of 2D arrays

    def load_nifti_slices(self, nifti_path, is_mask=False):
        img = nib.load(nifti_path)
        data = img.get_fdata()
        # print(f"Original data shape: {data.shape}")  # Debug print

        # For masks, create binary masks if necessary
        if is_mask:
            # Assuming liver label is 1
            data = (data == 1).astype(np.float32)
        else:
            data = data.astype(np.float32)
            # Normalize image data
            data = (data - np.min(data)) / (np.max(data) - np.min(data))

        # Slicing along the first dimension (Z-axis)
        slices = [data[:, :, i] for i in range(data.shape[2])]
        return slices  # List of 2D arrays