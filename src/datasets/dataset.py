# datasets/dataset.py
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from skimage.transform import resize
from scipy.ndimage import zoom

class SliceDataset(Dataset):
    def __init__(self, water_image_paths, fat_image_paths, mask_paths, desired_size=(256, 256), target_spacing=(3.0, 1.7188, 1.7188)):
        self.water_slices = []
        self.fat_slices = []
        self.mask_slices = []
        self.desired_size = desired_size
        self.target_spacing = target_spacing

        for water_path, fat_path, mask_path in zip(water_image_paths, fat_image_paths, mask_paths):
            water_slices = self.load_nifti_slices(water_path, is_mask=False)
            fat_slices = self.load_nifti_slices(fat_path, is_mask=False)
            mask_slices = self.load_nifti_slices(mask_path, is_mask=True)

            # Ensure they all have the same number of slices
            assert len(water_slices) == len(fat_slices) == len(mask_slices), f"Slice count mismatch in {water_path}"

            self.water_slices.extend(water_slices)
            self.fat_slices.extend(fat_slices)
            self.mask_slices.extend(mask_slices)

        # Now we have parallel lists: water_slices, fat_slices, mask_slices

    def __len__(self):
        return len(self.water_slices)

    def __getitem__(self, idx):
        water = self.water_slices[idx]
        fat = self.fat_slices[idx]
        mask = self.mask_slices[idx]

        # Resize images and mask
        water = resize(
            water,
            self.desired_size,
            mode='reflect',
            anti_aliasing=True
        )
        fat = resize(
            fat,
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

        # Stack water and fat along channel dimension: shape (2, H, W)
        image = np.stack([water, fat], axis=0)  # (2, H, W)

        mask = np.expand_dims(mask, axis=0)  # (1, H, W)

        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        return image, mask

    def load_nifti_slices(self, nifti_path, is_mask=False):
        img = nib.load(nifti_path)
        data = img.get_fdata()
        header = img.header

        # Original voxel spacing (X, Y, Z)
        voxel_spacing = header.get_zooms()
        # Reorder to (Z, Y, X)
        voxel_spacing = (voxel_spacing[2], voxel_spacing[1], voxel_spacing[0])

        data = data.astype(np.float32)

        if is_mask:
            data = (data > 0).astype(np.float32)
            order = 0
        else:
            # Normalize image to [0,1]
            data = (data - np.min(data)) / (np.ptp(data))
            order = 1

        # Resample volume to target_spacing
        zoom_factors = (
            voxel_spacing[0] / self.target_spacing[0],
            voxel_spacing[1] / self.target_spacing[1],
            voxel_spacing[2] / self.target_spacing[2]
        )
        data_resampled = zoom(data, zoom_factors, order=order)

        # Extract slices along Z-axis
        slices = [data_resampled[:, :, i] for i in range(data_resampled.shape[2])]
        return slices
