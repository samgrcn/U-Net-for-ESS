import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib  # For NIfTI files
import SimpleITK as sitk  # For DICOM files

class MuscleDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        """
        Args:
            image_paths (list): List of paths to image files.
            mask_paths (list): List of paths to mask files.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask paths
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image and mask data
        image = self.load_image(image_path)
        mask = self.load_mask(mask_path)

        # Ensure that image and mask have the same shape
        assert image.shape == mask.shape, f"Image and mask shapes do not match: {image.shape} vs {mask.shape}"

        # Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0).float()  # Shape: [1, D, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0).float()    # Shape: [1, D, H, W]

        return image, mask

    def load_image(self, path):
        if path.endswith('.nii') or path.endswith('.nii.gz'):
            return self.load_nifti(path)
        else:
            return self.load_dicom(path)

    def load_nifti(self, path):
        img = nib.load(path)
        img_data = img.get_fdata()
        # Normalize
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
        # Transpose to [D, H, W] if needed
        if img_data.ndim == 3:
            img_data = img_data.transpose(2, 0, 1)
        return img_data.astype(np.float32)

    def load_dicom(self, directory):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(directory)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        img_array = sitk.GetArrayFromImage(image)  # Shape: [num_slices, H, W]
        # Normalize
        img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
        return img_array.astype(np.float32)  # Shape: [D, H, W]

    def load_mask(self, path):
        if path.endswith('.nii') or path.endswith('.nii.gz'):
            mask = nib.load(path)
            mask_data = mask.get_fdata()
            # Transpose to [D, H, W] if needed
            if mask_data.ndim == 3:
                mask_data = mask_data.transpose(2, 0, 1)
        else:
            # Handle other mask formats if needed
            raise NotImplementedError("Mask format not supported.")
        # Ensure binary mask
        mask_data = (mask_data > 0).astype(np.float32)
        return mask_data