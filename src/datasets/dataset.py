# datasets/dataset.py
import os
import torch
from ..utils.utils import normalize
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import pydicom
from skimage.transform import resize
from scipy.ndimage import zoom

class SliceDataset(Dataset):
    def __init__(self, image_paths, mask_paths, desired_size=(256, 256), target_spacing=(3.0, 1.7188, 1.7188)):
        self.image_slices = []
        self.mask_slices = []
        self.desired_size = desired_size
        self.target_spacing = target_spacing  # (Z, Y, X)

        for img_path, mask_path in zip(image_paths, mask_paths):
            image_slices = self.load_nifti_slices(img_path, is_mask=False)
            mask_slices = self.load_nifti_slices(mask_path, is_mask=True)
            assert len(image_slices) == len(mask_slices), f"Mismatch in number of slices between image and mask at {img_path}"

            self.image_slices.extend(image_slices)
            self.mask_slices.extend(mask_slices)

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):
        image = self.image_slices[idx]
        mask = self.mask_slices[idx]

        # Resize image and mask to desired_size
        image = resize(
            image,
            self.desired_size,
            mode='reflect',
            anti_aliasing=True
        )
        mask = resize(
            mask,
            self.desired_size,
            order=0,
            preserve_range=True,
            anti_aliasing=False
        )

        # Add channel dimension
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        return image, mask

    def load_nifti_slices(self, nifti_path, is_mask=False):
        img = nib.load(nifti_path)
        data = img.get_fdata()
        header = img.header
        affine = img.affine

        # Get voxel spacing
        voxel_spacing = header.get_zooms()  # (X, Y, Z)
        voxel_spacing = (voxel_spacing[2], voxel_spacing[1], voxel_spacing[0])  # Convert to (Z, Y, X)

        if is_mask:
            data = data.astype(np.float32)
            data = (data > 0).astype(np.float32)
        else:
            data = normalize(data)

        # Resample volume to target_spacing
        zoom_factors = (
            voxel_spacing[0] / self.target_spacing[0],
            voxel_spacing[1] / self.target_spacing[1],
            voxel_spacing[2] / self.target_spacing[2]
        )
        data_resampled = zoom(data, zoom_factors, order=1 if not is_mask else 0)

        # Extract slices along Z-axis
        slices = [data_resampled[:, :, i] for i in range(data_resampled.shape[2])]
        return slices
